import sys
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import tkinter as tk
from tkinter import filedialog

# -----------------------------
# Config
# -----------------------------

FEATURE_KEYS = [
    "cp_stdev",
    "cp_mes_delta",
    "do_stdev",
    "do_mes_delta",
    "pr_stdev",
    "pr_mes_delta",
]

WINDOW = 10          # timesteps per sample (try 20 / 50 / 100)
STRIDE = 1           # sliding step
TRAIN_FRAC = 0.80    # time split: [0..80%) train, [80%..100%) test

SPOOFED_RATIO = 0.50  # desired fraction of spoofed windows (y=1). 0.5=balanced, 0.99=mostly spoofed
APPLY_RATIO_TO_TEST = True  # set True to enforce the same ratio on the test set (downsample only)
APPLY_RATIO_TO_TRAIN = False  # set True to enforce the same ratio on the test set (downsample only)

BATCH_SIZE = 8
EPOCHS = 30
SEED = 66

np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Helpers
# -----------------------------
def _to_2d(a):
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got {a.shape}")
    return a

def _mode_ignore_zeros(row):
    row = np.asarray(row).astype(np.int32)
    row = row[row != 0]
    if row.size == 0:
        return 0
    vals, cnts = np.unique(row, return_counts=True)
    return int(vals[np.argmax(cnts)])

def extract_sv_per_sat(sv_id_14x838):
    """
    sv_id is (sat_count, time_len) = (14, 838)
    Return one PRN per satellite row using mode ignoring zeros.
    """
    sv = _to_2d(sv_id_14x838)
    return np.array([_mode_ignore_zeros(sv[i, :]) for i in range(sv.shape[0])], dtype=np.int32)

def make_windows_for_range(series_TF, label, window, stride, start_idx, end_idx, sv_row=None):
    """
    series_TF: (time_len, F)
    windows fully inside [start_idx, end_idx)
    If sv_row is provided (shape: (time_len,)), any window that contains sv_id == 0 is skipped.
    """
    time_len, F = series_TF.shape
    X_list, y_list = [], []

    last_start = min(end_idx - window, time_len - window)
    for s in range(start_idx, last_start + 1, stride):
        e = s + window
        if e <= end_idx:
            # If sv_row provided, skip windows containing any zero sv_id entries
            if sv_row is not None:
                if s == 0: 
                    continue
                if sv_row[s-1] == 0:
                    continue
                # ensure slice in bounds and check zeros
                if s < 0 or e > sv_row.shape[0]:
                    continue
                if np.any(sv_row[s:e] == 0):
                    continue
            X_list.append(series_TF[s:e, :])
            y_list.append(label)

    if not X_list:
        return np.zeros((0, window, F), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int32)

def standardize_fit(X_train, eps=1e-6):
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)

def standardize_apply(X, mean, std):
    return ((X - mean) / std).astype(np.float32)

def compute_class_weights(y):
    n = len(y)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        return None
    return {0: n / (2.0 * n0), 1: n / (2.0 * n1)}

def apply_spoof_ratio(X, y, spoof_ratio, seed=0):
    """
    Downsample (no replacement) to enforce a desired class ratio.
    spoof_ratio: fraction of spoofed samples (y==1) in the returned dataset.

    Examples:
      SPOOFED_RATIO=0.50 -> equal spoofed/genuine
      SPOOFED_RATIO=0.99 -> mostly spoofed (very few genuine)

    Note: This ONLY downsamples; it never oversamples.
    """
    if spoof_ratio is None:
        return X, y
    spoof_ratio = float(spoof_ratio)
    if not (0.0 < spoof_ratio < 1.0):
        raise ValueError("SPOOFED_RATIO must be between 0 and 1 (exclusive).")

    idx_s = np.where(y == 1)[0]
    idx_g = np.where(y == 0)[0]
    n_s, n_g = len(idx_s), len(idx_g)

    # If one class is missing, nothing to balance
    if n_s == 0 or n_g == 0:
        return X, y

    # Maximum total size achievable without replacement given the desired ratio
    total_max_from_s = int(np.floor(n_s / spoof_ratio))
    total_max_from_g = int(np.floor(n_g / (1.0 - spoof_ratio)))
    total = min(total_max_from_s, total_max_from_g)

    if total <= 0:
        raise RuntimeError("Cannot satisfy SPOOFED_RATIO with the available data (one class too small).")

    n_s_des = int(np.floor(total * spoof_ratio))
    n_g_des = total - n_s_des

    # Guard against rounding producing empty class
    n_s_des = min(n_s_des, n_s)
    n_g_des = min(n_g_des, n_g)
    if n_s_des == 0 or n_g_des == 0:
        raise RuntimeError("SPOOFED_RATIO resulted in zero samples for one class. Adjust SPOOFED_RATIO.")

    rng = np.random.default_rng(seed)
    pick_s = rng.choice(idx_s, size=n_s_des, replace=False)
    pick_g = rng.choice(idx_g, size=n_g_des, replace=False)

    pick = np.concatenate([pick_s, pick_g])
    rng.shuffle(pick)

    return X[pick], y[pick]


# -----------------------------
# Load .mat
# -----------------------------
try:
    root = tk.Tk()
    root.withdraw()
    spoofed_path = filedialog.askopenfilename(
        title='Select SPOOFED dataset (.mat file)',
        filetypes=[('MAT files', '*.mat'), ('All files', '*.*')]
    )
    if not spoofed_path:
        print('No spoofed file selected.')
        sys.exit(1)
    
    genuine_path = filedialog.askopenfilename(
        title='Select GENUINE dataset (.mat file)',
        filetypes=[('MAT files', '*.mat'), ('All files', '*.*')]
    )
    if not genuine_path:
        print('No genuine file selected.')
        sys.exit(1)
except Exception:
    if len(sys.argv) < 3:
        print('Usage: python lstm.py path/to/spoofed.mat path/to/genuine.mat')
        sys.exit(1)
    spoofed_path = sys.argv[1]
    genuine_path = sys.argv[2]

mat_spoofed = loadmat(spoofed_path)
mat_genuine = loadmat(genuine_path)

def load_and_validate(mat_data, dataset_name):
    """Load features from .mat file and validate shapes."""
    feat0 = _to_2d(mat_data[FEATURE_KEYS[0]]).astype(np.float32)
    sat_count, time_len = feat0.shape
    
    for k in FEATURE_KEYS:
        a = _to_2d(mat_data[k])
        if a.shape != (sat_count, time_len):
            raise ValueError(f"{dataset_name}: {k} shape {a.shape} != ({sat_count},{time_len})")
    
    sv_id = _to_2d(mat_data["sv_id"])
    if sv_id.shape != (sat_count, time_len):
        raise ValueError(f"{dataset_name}: sv_id shape {sv_id.shape} != ({sat_count},{time_len})")
    
    features = []
    for k in FEATURE_KEYS:
        a = _to_2d(mat_data[k]).astype(np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(a)
    
    sv_per_sat = extract_sv_per_sat(sv_id)
    sv_matrix = sv_id.astype(np.int32)  # keep full per-time sv_id information
    print(f"{dataset_name} - sat_count: {sat_count}, time_len: {time_len}")
    print(f"{dataset_name} - sv_per_sat: {sv_per_sat.tolist()}")
    
    return features, sat_count, time_len, sv_per_sat, sv_matrix

spoofed_features, spoofed_sat_count, spoofed_time_len, spoofed_sv_per_sat, spoofed_sv_matrix = load_and_validate(mat_spoofed, "SPOOFED")
genuine_features, genuine_sat_count, genuine_time_len, genuine_sv_per_sat, genuine_sv_matrix = load_and_validate(mat_genuine, "GENUINE")

F = len(FEATURE_KEYS)
spoofed_split = int(spoofed_time_len * TRAIN_FRAC)
genuine_split = int(genuine_time_len * TRAIN_FRAC)

# -----------------------------
# Build dataset (windows per satellite)
# Time-based split to reduce leakage
# -----------------------------
X_train_all, y_train_all = [], []
X_test_all, y_test_all = [], []

# collect genuine windows for correlation analysis
genuine_windows = []

# Process spoofed dataset (label=1)
for i in range(spoofed_sat_count):
    series = np.stack([spoofed_features[f_idx][i, :] for f_idx in range(F)], axis=-1)
    sv_row = spoofed_sv_matrix[i, :]
    Xtr, ytr = make_windows_for_range(series, 1, WINDOW, STRIDE, 0, spoofed_split, sv_row=sv_row)
    Xte, yte = make_windows_for_range(series, 1, WINDOW, STRIDE, spoofed_split, spoofed_time_len, sv_row=sv_row)
    if len(Xtr) > 0:
        X_train_all.append(Xtr); y_train_all.append(ytr)
    if len(Xte) > 0:
        X_test_all.append(Xte); y_test_all.append(yte)

# Process genuine dataset (label=0)
for i in range(genuine_sat_count):
    series = np.stack([genuine_features[f_idx][i, :] for f_idx in range(F)], axis=-1)
    sv_row = genuine_sv_matrix[i, :]
    Xtr, ytr = make_windows_for_range(series, 0, WINDOW, STRIDE, 0, genuine_split, sv_row=sv_row)
    Xte, yte = make_windows_for_range(series, 0, WINDOW, STRIDE, genuine_split, genuine_time_len, sv_row=sv_row)
    if len(Xtr) > 0:
        X_train_all.append(Xtr); y_train_all.append(ytr)
        genuine_windows.append(Xtr)   # collect genuine train windows
    if len(Xte) > 0:
        X_test_all.append(Xte); y_test_all.append(yte)
        genuine_windows.append(Xte)   # collect genuine test windows

X_train = np.concatenate(X_train_all, axis=0) if X_train_all else np.zeros((0, WINDOW, F), np.float32)
y_train = np.concatenate(y_train_all, axis=0) if y_train_all else np.zeros((0,), np.int32)

X_test = np.concatenate(X_test_all, axis=0) if X_test_all else np.zeros((0, WINDOW, F), np.float32)
y_test = np.concatenate(y_test_all, axis=0) if y_test_all else np.zeros((0,), np.int32)


# -----------------------------
# Optionally enforce spoofed/genuine ratio (downsample only)
# -----------------------------
if APPLY_RATIO_TO_TRAIN:
    X_train, y_train = apply_spoof_ratio(X_train, y_train, SPOOFED_RATIO, seed=SEED)
if APPLY_RATIO_TO_TEST:
    X_test, y_test = apply_spoof_ratio(X_test, y_test, SPOOFED_RATIO, seed=SEED + 1)

print("X_train:", X_train.shape, "spoofed%:", float(y_train.mean() * 100) if len(y_train) else 0.0)
print("X_test :", X_test.shape,  "spoofed%:", float(y_test.mean() * 100)  if len(y_test) else 0.0)

# --- new: compute genuine feature correlations and print table ---
if genuine_windows:
    genuine_all = np.concatenate(genuine_windows, axis=0)  # (N_windows, WINDOW, F)
    if genuine_all.shape[0] > 0:
        # per-window feature means -> shape (N_windows, F)
        per_win_means = genuine_all.mean(axis=1)
        # Pearson correlation matrix across features (F x F)
        corr = np.corrcoef(per_win_means, rowvar=False)
        # Replace NaNs (e.g., zero variance features) with 0.0 and clip to [-1,1]
        corr = np.where(np.isnan(corr), 0.0, corr)
        corr = np.clip(corr, -1.0, 1.0)

        print("\nGenuine feature correlations (-1..1):")
        # header
        header = "".ljust(16) + " ".join(f"{k[:10]:>8}" for k in FEATURE_KEYS)
        print(header)
        for i, k in enumerate(FEATURE_KEYS):
            row_vals = "  ".join(f"{v:8.2f}" for v in corr[i])
            print(f"{k[:14]:<16}{row_vals}")
else:
    print("\nNo genuine windows collected for correlation analysis.")

if len(X_train) == 0 or len(X_test) == 0:
    raise RuntimeError("No windows were created. Reduce WINDOW or adjust STRIDE/TRAIN_FRAC.")

# -----------------------------
# Normalize (fit on train only)
# -----------------------------
mean, std = standardize_fit(X_train)
X_train = standardize_apply(X_train, mean, std)
X_test = standardize_apply(X_test, mean, std)

# Save normalization params for inference later
np.savez("norm_params.npz", mean=mean, std=std, window=WINDOW, feature_keys=np.array(FEATURE_KEYS))

# -----------------------------
# tf.data
# -----------------------------
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(min(len(X_train), 5000), seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# LSTM model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW, F)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
             tf.keras.metrics.AUC(name="auc")]
)

class_weights = compute_class_weights(y_train)
print("class_weights:", class_weights)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)
]

model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print("\nEvaluation:")
model.evaluate(test_ds, verbose=1)

# -----------------------------
# Export to TFLite
# -----------------------------
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic range quant
# tflite_model = converter.convert()

# with open("lstm_spoof_detector.tflite", "wb") as f:
#     f.write(tflite_model)

# print("\nSaved: lstm_spoof_detector.tflite")
print("Saved: norm_params.npz")
