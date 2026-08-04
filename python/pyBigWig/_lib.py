"""ctypes loading for the Mojo interval kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYBIGWIG_LIB") or os.path.join(ROOT, "dist", "libmojo-pybigwig.so")
I = ctypes.c_int64


def _build() -> str:
    if os.path.exists(LIB):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
        if not os.path.exists(pixi):
            raise RuntimeError("Mojo compiler not found; run through pixi")
        command = [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
    else:
        command = ["bash", os.path.join(ROOT, "build", "build.sh")]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_handle: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        _handle = ctypes.CDLL(_build())
        _handle.mpbw_values.argtypes = [I] * 7
        _handle.mpbw_values.restype = None
        _handle.mpbw_bin_accumulate.argtypes = [I] * 8
        _handle.mpbw_bin_accumulate.restype = None
    return _handle


def _array_address(array: np.ndarray, dtype: np.dtype, size: int, *, writable: bool = False) -> int:
    """Validate a NumPy buffer before passing its address to native code."""
    if (not isinstance(array, np.ndarray) or array.dtype != dtype or
            not array.flags.c_contiguous or array.size < size or
            (writable and not array.flags.writeable)):
        raise ValueError("invalid NumPy buffer for Mojo kernel")
    address = int(array.ctypes.data)
    if not address:
        raise ValueError("null NumPy buffer for Mojo kernel")
    return address


def values(starts: np.ndarray, ends: np.ndarray, values: np.ndarray,
           query_start: int, query_end: int, result: np.ndarray) -> None:
    """Call the values kernel with exact, contiguous NumPy buffers."""
    n = starts.size
    if ends.size != n or values.size != n or query_end < query_start:
        raise ValueError("inconsistent interval buffers")
    length = query_end - query_start
    starts_addr = _array_address(starts, np.dtype(np.int64), n)
    ends_addr = _array_address(ends, np.dtype(np.int64), n)
    values_addr = _array_address(values, np.dtype(np.float64), n)
    result_addr = _array_address(result, np.dtype(np.float64), length, writable=True)
    if n:
        lib().mpbw_values(starts_addr, ends_addr, values_addr, n, query_start, query_end, result_addr)


def bin_accumulate(starts: np.ndarray, ends: np.ndarray, values: np.ndarray,
                   query_start: int, query_end: int, bins: int, acc: np.ndarray) -> None:
    """Call the stats kernel after checking every native buffer extent."""
    n = starts.size
    if ends.size != n or values.size != n or bins < 1 or query_end <= query_start:
        raise ValueError("inconsistent interval buffers")
    starts_addr = _array_address(starts, np.dtype(np.int64), n)
    ends_addr = _array_address(ends, np.dtype(np.int64), n)
    values_addr = _array_address(values, np.dtype(np.float64), n)
    acc_addr = _array_address(acc, np.dtype(np.float64), bins * 5, writable=True)
    if n:
        lib().mpbw_bin_accumulate(starts_addr, ends_addr, values_addr, n,
                                  query_start, query_end, bins, acc_addr)
