"""Serialization backends for cache entries.

Each backend is a pair of (save_fn, load_fn, extension).  The default
backend is ``npz`` (NumPy compressed archive), which is fast for
array-heavy scientific data and produces portable files.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ── Backend type ──────────────────────────────────────────────────────────
# A backend is a (save_fn, load_fn, file_extension) tuple.
Backend = tuple[Callable[[Path, Any], None], Callable[[Path], Any], str]


# ── NumPy .npz ───────────────────────────────────────────────────────────

def _npz_save(path: Path, data: Any) -> None:
    """Save to .npz.  Dicts become named arrays; scalars/objects become
    a single ``data`` entry; arbitrary objects fall back to pickle-in-npz."""
    if isinstance(data, dict):
        npz_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                npz_data[k] = v
            elif isinstance(v, (list, tuple)):
                arr = np.asarray(v)
                if arr.dtype == object:
                    npz_data[k] = np.void(pickle.dumps(v))
                else:
                    npz_data[k] = arr
            elif isinstance(v, dict):
                npz_data[k] = np.void(pickle.dumps(v))
            else:
                npz_data[k] = v
        np.savez_compressed(path, **npz_data)
    elif isinstance(data, np.ndarray):
        np.savez_compressed(path, data=data)
    else:
        np.savez_compressed(path, __pickle__=np.void(pickle.dumps(data)))


def _npz_load(path: Path) -> Any:
    """Load from .npz.  Restores dict, array, or pickled object."""
    raw = np.load(path, allow_pickle=True)
    if "__pickle__" in raw:
        return pickle.loads(raw["__pickle__"].tobytes())
    if "data" in raw and len(raw) == 1:
        return raw["data"]
    # Return as dict, converting 0-d arrays back to scalars
    result = {}
    for k in raw:
        val = raw[k]
        if isinstance(val, np.ndarray) and val.ndim == 0:
            result[k] = val.item()
        else:
            result[k] = val
    return result


# ── Pickle ───────────────────────────────────────────────────────────────

def _pickle_save(path: Path, data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_load(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Registry ─────────────────────────────────────────────────────────────

BACKENDS: dict[str, Backend] = {
    "npz": (_npz_save, _npz_load, ".npz"),
    "pickle": (_pickle_save, _pickle_load, ".pkl"),
}

# Optional HDF5 backend
try:
    import h5py  # noqa: F401

    def _h5_save(path: Path, data: Any) -> None:
        import h5py

        with h5py.File(path, "w") as f:
            if isinstance(data, dict):
                for k, v in data.items():
                    f.create_dataset(k, data=np.asarray(v), compression="gzip")
            elif isinstance(data, np.ndarray):
                f.create_dataset("data", data=data, compression="gzip")
            else:
                f.attrs["pickle"] = np.void(pickle.dumps(data))

    def _h5_load(path: Path) -> Any:
        import h5py

        with h5py.File(path, "r") as f:
            if "pickle" in f.attrs:
                return pickle.loads(f.attrs["pickle"].tobytes())
            if "data" in f and len(f) == 1:
                return f["data"][:]
            return {k: f[k][:] for k in f}

    BACKENDS["hdf5"] = (_h5_save, _h5_load, ".h5")
except ImportError:
    pass


def get_backend(name: str = "npz") -> Backend:
    """Look up a backend by name.

    Raises
    ------
    ValueError
        If the backend is unknown.
    """
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {list(BACKENDS)}"
        )
    return BACKENDS[name]
