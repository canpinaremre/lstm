import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

# Optional GUI file picker (desktop). If not available, use CLI args.
try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover
    tk = None
    filedialog = None


@dataclass(frozen=True)
class Config:
    # Feature names to read from the MAT files (each is sat_count x time_len)
    feature_keys: Tuple[str, ...] = (
        "cp_stdev",
        "cp_mes_delta",
        "do_stdev",
        "do_mes_delta",
        "pr_stdev",
        "pr_mes_delta",
    )

    # Sliding-window parameters
    window: int = 20
    stride: int = 1

    # Random split ratios AFTER building all windows
    train_frac: float = 0.70
    val_frac: float = 0.15  # test = 1 - train_frac - val_frac

    # Class ratio in the final dataset (downsample only, no oversampling)
    spoof_ratio: float = 0.50

    # Training parameters
    batch_size: int = 32
    epochs: int = 30
    seed: int = 58
    shuffle_buffer: int = 5000

    # Train all epochs, but keep best weights using ModelCheckpoint
    monitor_metric: str = "val_auc"  # or "val_prauc"

    # Window validity rules (based on sv_id)
    drop_if_sv_contains_zero: bool = True
    require_prev_sample_valid: bool = True


# -----------------------------
# Helpers
# -----------------------------

def _to_2d(a) -> np.ndarray:
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")
    return a


def _mode_ignore_zeros(row: np.ndarray) -> int:
    row = np.asarray(row).astype(np.int32)
    row = row[row != 0]
    if row.size == 0:
        return 0
    vals, cnts = np.unique(row, return_counts=True)
    return int(vals[np.argmax(cnts)])


def extract_sv_per_sat(sv_id_2d: np.ndarray) -> np.ndarray:
    """Return one PRN per satellite row using mode ignoring zeros."""
    sv = _to_2d(sv_id_2d)
    return np.array([_mode_ignore_zeros(sv[i, :]) for i in range(sv.shape[0])], dtype=np.int32)


def standardize_fit(X_train: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def make_windows_for_series(
    series_TF: np.ndarray,
    label: int,
    window: int,
    stride: int,
    sv_row: Optional[np.ndarray],
    drop_if_sv_contains_zero: bool,
    require_prev_sample_valid: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    series_TF: (time_len, F)
    sv_row: (time_len,) int sv_id values for this satellite row

    Window rules:
      - If drop_if_sv_contains_zero: discard windows containing any sv_id == 0.
      - If require_prev_sample_valid: skip windows where s==0 or sv_row[s-1]==0.

    Returns:
      X: (N, window, F) float32
      y: (N,) int32
    """
    time_len, F = series_TF.shape
    if time_len < window:
        return np.zeros((0, window, F), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    X_list: List[np.ndarray] = []
    last_start = time_len - window

    for s in range(0, last_start + 1, stride):
        e = s + window

        if sv_row is not None and drop_if_sv_contains_zero:
            if require_prev_sample_valid:
                if s == 0:
                    continue
                if sv_row[s - 1] == 0:
                    continue
            if np.any(sv_row[s:e] == 0):
                continue

        X_list.append(series_TF[s:e, :])

    if not X_list:
        return np.zeros((0, window, F), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    X = np.stack(X_list).astype(np.float32)
    y = np.full((X.shape[0],), int(label), dtype=np.int32)
    return X, y


def pick_paths_from_user() -> Tuple[str, str]:
    """Try GUI picker, else expect CLI: python lstm.py spoofed.mat genuine.mat"""
    if tk is not None and filedialog is not None:
        try:
            root = tk.Tk()
            root.withdraw()

            spoofed_path = filedialog.askopenfilename(
                title="Select SPOOFED dataset (.mat file)",
                filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
            )
            if not spoofed_path:
                raise RuntimeError("No spoofed file selected.")

            genuine_path = filedialog.askopenfilename(
                title="Select GENUINE dataset (.mat file)",
                filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
            )
            if not genuine_path:
                raise RuntimeError("No genuine file selected.")

            return spoofed_path, genuine_path
        except Exception:
            pass

    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python lstm.py path/to/spoofed.mat path/to/genuine.mat")

    return sys.argv[1], sys.argv[2]


def load_and_validate(
    mat_data: dict,
    dataset_name: str,
    feature_keys: Tuple[str, ...],
) -> Tuple[List[np.ndarray], int, int, np.ndarray, np.ndarray]:
    """Load features from a MAT dict and validate shapes."""
    feat0 = _to_2d(mat_data[feature_keys[0]]).astype(np.float32)
    sat_count, time_len = feat0.shape

    for k in feature_keys:
        a = _to_2d(mat_data[k])
        if a.shape != (sat_count, time_len):
            raise ValueError(f"{dataset_name}: {k} shape {a.shape} != ({sat_count},{time_len})")

    sv_id = _to_2d(mat_data["sv_id"])
    if sv_id.shape != (sat_count, time_len):
        raise ValueError(f"{dataset_name}: sv_id shape {sv_id.shape} != ({sat_count},{time_len})")

    features: List[np.ndarray] = []
    for k in feature_keys:
        a = _to_2d(mat_data[k]).astype(np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(a)

    sv_per_sat = extract_sv_per_sat(sv_id)
    sv_matrix = sv_id.astype(np.int32)

    print(f"{dataset_name} - sat_count: {sat_count}, time_len: {time_len}")
    print(f"{dataset_name} - sv_per_sat: {sv_per_sat.tolist()}")

    return features, sat_count, time_len, sv_per_sat, sv_matrix


def build_all_windows(
    features: List[np.ndarray],
    sat_count: int,
    sv_matrix: np.ndarray,
    label: int,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build windows for all satellite rows and concatenate."""
    F = len(features)
    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for i in range(sat_count):
        series = np.stack([features[f_idx][i, :] for f_idx in range(F)], axis=-1)  # (time_len, F)
        sv_row = sv_matrix[i, :]

        X, y = make_windows_for_series(
            series_TF=series,
            label=label,
            window=cfg.window,
            stride=cfg.stride,
            sv_row=sv_row,
            drop_if_sv_contains_zero=cfg.drop_if_sv_contains_zero,
            require_prev_sample_valid=cfg.require_prev_sample_valid,
        )

        if X.shape[0] > 0:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        return (
            np.zeros((0, cfg.window, F), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


def stratified_sample_and_split(
    X_spoof: np.ndarray,
    X_genuine: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    - Enforce cfg.spoof_ratio by downsampling (no oversampling).
    - Random split AFTER windowing (no time split), stratified by class.

    NOTE: This can introduce temporal leakage across splits if adjacent windows
    are highly correlated. You requested this behavior for experimentation.
    """
    rng = np.random.default_rng(cfg.seed)

    n_s = int(X_spoof.shape[0])
    n_g = int(X_genuine.shape[0])

    if n_s == 0 or n_g == 0:
        raise RuntimeError(f"Insufficient windows. spoof={n_s}, genuine={n_g}")

    r = float(cfg.spoof_ratio)
    if not (0.0 < r < 1.0):
        raise ValueError("spoof_ratio must be between 0 and 1 (exclusive).")

    train_frac = float(cfg.train_frac)
    val_frac = float(cfg.val_frac)
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    # Max achievable total size without replacement for desired ratio
    total_max_from_s = int(np.floor(n_s / r))
    total_max_from_g = int(np.floor(n_g / (1.0 - r)))
    total = min(total_max_from_s, total_max_from_g)
    if total <= 0:
        raise RuntimeError("Cannot satisfy spoof_ratio with available windows.")

    n_s_des = int(np.floor(total * r))
    n_g_des = total - n_s_des
    if n_s_des <= 0 or n_g_des <= 0:
        raise RuntimeError("spoof_ratio produced empty class after downsampling.")

    idx_s = rng.choice(n_s, size=n_s_des, replace=False)
    idx_g = rng.choice(n_g, size=n_g_des, replace=False)
    rng.shuffle(idx_s)
    rng.shuffle(idx_g)

    def _split_counts(n: int) -> Tuple[int, int, int]:
        n_train = int(np.floor(n * train_frac))
        n_val = int(np.floor(n * val_frac))
        n_test = n - n_train - n_val
        if n_test <= 0:
            # Move one from train or val to test if possible
            if n_train > 1:
                n_train -= 1
                n_test += 1
            elif n_val > 1:
                n_val -= 1
                n_test += 1
            else:
                raise RuntimeError("Not enough samples to create train/val/test splits.")
        return n_train, n_val, n_test

    s_tr, s_va, s_te = _split_counts(n_s_des)
    g_tr, g_va, g_te = _split_counts(n_g_des)

    s_train = idx_s[:s_tr]
    s_val = idx_s[s_tr:s_tr + s_va]
    s_test = idx_s[s_tr + s_va:s_tr + s_va + s_te]

    g_train = idx_g[:g_tr]
    g_val = idx_g[g_tr:g_tr + g_va]
    g_test = idx_g[g_tr + g_va:g_tr + g_va + g_te]

    X_train = np.concatenate([X_spoof[s_train], X_genuine[g_train]], axis=0)
    y_train = np.concatenate([
        np.ones((len(s_train),), np.int32),
        np.zeros((len(g_train),), np.int32),
    ], axis=0)

    X_val = np.concatenate([X_spoof[s_val], X_genuine[g_val]], axis=0)
    y_val = np.concatenate([
        np.ones((len(s_val),), np.int32),
        np.zeros((len(g_val),), np.int32),
    ], axis=0)

    X_test = np.concatenate([X_spoof[s_test], X_genuine[g_test]], axis=0)
    y_test = np.concatenate([
        np.ones((len(s_test),), np.int32),
        np.zeros((len(g_test),), np.int32),
    ], axis=0)

    def _shuffle_pair(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.arange(len(y))
        rng.shuffle(idx)
        return X[idx], y[idx]

    X_train, y_train = _shuffle_pair(X_train, y_train)
    X_val, y_val = _shuffle_pair(X_val, y_val)
    X_test, y_test = _shuffle_pair(X_test, y_test)

    print("\nDataset sizes after ratio + random stratified split:")
    print(f"  Train: {X_train.shape} spoof%={y_train.mean() * 100:.1f}")
    print(f"  Val  : {X_val.shape} spoof%={y_val.mean() * 100:.1f}")
    print(f"  Test : {X_test.shape} spoof%={y_test.mean() * 100:.1f}")

    np.savez(
        "dataset_params.npz",
        seed=np.array([cfg.seed], dtype=np.int32),
        window=np.array([cfg.window], dtype=np.int32),
        stride=np.array([cfg.stride], dtype=np.int32),
        train_frac=np.array([cfg.train_frac], dtype=np.float32),
        val_frac=np.array([cfg.val_frac], dtype=np.float32),
        spoof_ratio=np.array([cfg.spoof_ratio], dtype=np.float32),
        n_spoof_raw=np.array([n_s], dtype=np.int64),
        n_genuine_raw=np.array([n_g], dtype=np.int64),
        n_spoof_used=np.array([n_s_des], dtype=np.int64),
        n_genuine_used=np.array([n_g_des], dtype=np.int64),
        train_size=np.array([len(y_train)], dtype=np.int64),
        val_size=np.array([len(y_val)], dtype=np.int64),
        test_size=np.array([len(y_test)], dtype=np.int64),
    )
    print("Saved: dataset_params.npz")

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_tf_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = (
        train_ds
        .shuffle(min(len(X_train), cfg.shuffle_buffer), seed=cfg.seed, reshuffle_each_iteration=True)
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_model(cfg: Config, num_features: int) -> tf.keras.Model:
    # Regularizers affect training only; they do not complicate inference graphs.
    reg = tf.keras.regularizers.l2(1e-4)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(cfg.window, num_features)),
        tf.keras.layers.LSTM(64, kernel_regularizer=reg, recurrent_regularizer=reg),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prauc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def print_confusion(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    title: str = "",
) -> None:
    y_prob = model.predict(X, batch_size=256, verbose=0).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = int(np.sum((y == 1) & (y_pred == 1)))
    tn = int(np.sum((y == 0) & (y_pred == 0)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-9)

    if title:
        print(f"{title}")
    print(
        f"Confusion @thr={threshold:.2f}: TP={tp} FP={fp} TN={tn} FN={fn} "
        f"| precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}"
    )


def main() -> None:
    cfg = Config()

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    spoofed_path, genuine_path = pick_paths_from_user()

    mat_spoofed = loadmat(spoofed_path)
    mat_genuine = loadmat(genuine_path)

    spoofed_features, spoofed_sat_count, _, _, spoofed_sv_matrix = load_and_validate(
        mat_spoofed, "SPOOFED", cfg.feature_keys
    )
    genuine_features, genuine_sat_count, _, _, genuine_sv_matrix = load_and_validate(
        mat_genuine, "GENUINE", cfg.feature_keys
    )

    # 1) Build ALL windows first (no time split)
    X_spoof, _ = build_all_windows(spoofed_features, spoofed_sat_count, spoofed_sv_matrix, label=1, cfg=cfg)
    X_genuine, _ = build_all_windows(genuine_features, genuine_sat_count, genuine_sv_matrix, label=0, cfg=cfg)

    print("\nRaw windows (before ratio/split):")
    print(f"  spoof  : {X_spoof.shape}")
    print(f"  genuine: {X_genuine.shape}")

    if X_spoof.shape[0] == 0 or X_genuine.shape[0] == 0:
        raise RuntimeError("No windows created. Reduce window, adjust stride, or check sv_id filtering.")

    # 2) Enforce spoof_ratio, then random stratified train/val/test split
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_sample_and_split(X_spoof, X_genuine, cfg)

    # 3) Normalize (fit on train only)
    mean, std = standardize_fit(X_train)
    X_train = standardize_apply(X_train, mean, std)
    X_val = standardize_apply(X_val, mean, std)
    X_test = standardize_apply(X_test, mean, std)

    np.savez(
        "norm_params.npz",
        mean=mean,
        std=std,
        window=np.array([cfg.window], dtype=np.int32),
        feature_keys=np.array(cfg.feature_keys),
    )
    print("Saved: norm_params.npz")

    # 4) tf.data
    train_ds, val_ds, test_ds = build_tf_datasets(X_train, y_train, X_val, y_val, X_test, y_test, cfg)

    # 5) Model + checkpoint (train full epochs, keep best weights)
    model = build_model(cfg, num_features=len(cfg.feature_keys))
    best_weights_path = "best_weights.weights.h5"

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_weights_path,
        monitor=cfg.monitor_metric,
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=[ckpt],
        verbose=1,
    )

    # 6) Load best weights and evaluate
    try:
        model.load_weights(best_weights_path)
        print(f"Loaded best weights: {best_weights_path} (monitor={cfg.monitor_metric})")
    except Exception as e:
        print(f"Warning: could not load best weights ({e}). Using last epoch weights.")

    print("\nEvaluation (VAL):")
    model.evaluate(val_ds, verbose=1)

    print("\nEvaluation (TEST):")
    model.evaluate(test_ds, verbose=1)

    # Confusion matrices at default threshold 0.5
    print_confusion(model, X_val, y_val, threshold=0.5, title="VAL confusion")
    print_confusion(model, X_test, y_test, threshold=0.5, title="TEST confusion")

    # 7) Save final model (with best weights loaded)
    model.save("lstm_spoof_detector.keras")
    model.save_weights("lstm_weights.weights.h5")
    print("Saved: lstm_spoof_detector.keras")
    print("Saved: lstm_weights.weights.h5")

    # Save training history for later plotting
    np.savez("train_history.npz", **{k: np.array(v) for k, v in history.history.items()})
    print("Saved: train_history.npz")


if __name__ == "__main__":
    main()
