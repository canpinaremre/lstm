#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np

# Try scipy first, then fallback to h5py for MATLAB v7.3 files
try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None

try:
    import h5py
except ImportError:
    h5py = None


FEATURE_KEYS: Tuple[str, ...] = (
    "cp_stdev",
    "cp_mes_delta",
    "do_stdev",
    "do_mes_delta",
    "pr_stdev",
)


def load_mat_file(mat_path: str) -> Dict[str, Any]:
    """
    Load MATLAB .mat file.
    Supports:
      - old style MAT via scipy.io.loadmat
      - v7.3 HDF5 MAT via h5py
    """
    path = Path(mat_path)
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    # First try scipy for classic .mat
    if loadmat is not None:
        try:
            data = loadmat(mat_path)
            return data
        except NotImplementedError:
            pass
        except Exception:
            pass

    # Fallback to h5py for v7.3
    if h5py is not None:
        try:
            f = h5py.File(mat_path, "r")
            return {"__h5py_file__": f}
        except Exception as e:
            raise RuntimeError(
                f"Could not read MAT file with scipy or h5py: {e}"
            ) from e

    raise RuntimeError(
        "Unable to read MAT file. Install scipy for classic MAT files "
        "or h5py for MATLAB v7.3 files."
    )


def close_mat_file(data: Dict[str, Any]) -> None:
    f = data.get("__h5py_file__")
    if f is not None:
        f.close()


def get_var(data: Dict[str, Any], key: str) -> np.ndarray:
    """
    Read variable by name from loaded MAT content.
    """
    # h5py-backed MAT
    if "__h5py_file__" in data:
        f = data["__h5py_file__"]
        if key not in f:
            raise KeyError(f"Variable '{key}' not found in MAT file.")
        arr = np.array(f[key])
        # h5py datasets from MATLAB are often transposed vs scipy expectations
        arr = np.array(arr)
        return np.squeeze(arr)

    # scipy-backed MAT
    if key not in data:
        raise KeyError(f"Variable '{key}' not found in MAT file.")
    return np.squeeze(np.array(data[key]))


def extract_series(arr: np.ndarray, start_index: int, sv_id: int, window: int) -> np.ndarray:
    """
    Extract a 1D window for one feature.

    Supported shapes:
      - 1D: [N]
      - 2D: [N, num_sv] or [num_sv, N]

    Heuristic:
      Prefer [time, sv] if it fits.
      Otherwise try [sv, time].
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 0:
        raise ValueError("Feature array is scalar, expected 1D or 2D.")

    if arr.ndim == 1:
        if start_index + window > arr.shape[0]:
            raise IndexError(
                f"Requested range [{start_index}:{start_index + window}] "
                f"exceeds 1D array length {arr.shape[0]}."
            )
        return arr[start_index:start_index + window].astype(np.float32)

    if arr.ndim == 2:
        rows, cols = arr.shape

        # Case 1: [time, sv]
        if rows >= start_index + window and cols > sv_id:
            return arr[start_index:start_index + window, sv_id].astype(np.float32)

        # Case 2: [sv, time]
        if cols >= start_index + window and rows > sv_id:
            return arr[sv_id, start_index:start_index + window].astype(np.float32)

        raise IndexError(
            f"Could not interpret array shape {arr.shape} with "
            f"start_index={start_index}, sv_id={sv_id}, window={window}."
        )

    raise ValueError(
        f"Unsupported array ndim={arr.ndim}. Expected 1D or 2D."
    )


def build_feature_matrix(
    data: Dict[str, Any],
    feature_keys: Tuple[str, ...],
    start_index: int,
    sv_id: int,
    window: int,
) -> np.ndarray:
    """
    Output shape: [window, num_features]
    """
    cols = []
    for key in feature_keys:
        arr = get_var(data, key)
        vec = extract_series(arr, start_index, sv_id, window)
        cols.append(vec)

    # Stack columns into [window, num_features]
    mat = np.column_stack(cols).astype(np.float32)
    return mat


def format_c_array(matrix: np.ndarray, var_name: str = "features") -> str:
    rows, cols = matrix.shape
    lines = [f"float {var_name}[{rows}][{cols}] = {{"]

    for r in range(rows):
        vals = ", ".join(f"{float(v):.8f}" for v in matrix[r])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read MATLAB features and export a C/C++ float array text file."
    )
    parser.add_argument("mat_file", help="Input .mat file path")
    parser.add_argument("start_index", type=int, help="Start index")
    parser.add_argument("sv_id", type=int, help="SV id / column index")
    parser.add_argument(
        "-o", "--output", default="features.txt", help="Output txt file path"
    )
    parser.add_argument(
        "-w", "--window", type=int, default=20, help="Window length (default: 20)"
    )
    parser.add_argument(
        "--var-name", default="features", help="C array variable name"
    )

    args = parser.parse_args()

    data = None
    try:
        data = load_mat_file(args.mat_file)

        mat = build_feature_matrix(
            data=data,
            feature_keys=FEATURE_KEYS,
            start_index=args.start_index,
            sv_id=args.sv_id,
            window=args.window,
        )

        text = format_c_array(mat, var_name=args.var_name)
        Path(args.output).write_text(text, encoding="utf-8")

        print(f"Exported: {args.output}")
        print(f"Shape: {mat.shape}")
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    finally:
        if data is not None:
            close_mat_file(data)


if __name__ == "__main__":
    sys.exit(main())