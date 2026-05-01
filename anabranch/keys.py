"""Content-addressable cache keys from Python function arguments.

Generates deterministic, reproducible keys from (args, kwargs, function
source) using SHA-256.  NumPy arrays are hashed by shape + dtype +
first 256 bytes of data for speed.
"""

from __future__ import annotations

import hashlib
import json
import inspect
from typing import Any, Callable

import numpy as np


def _canonicalize(obj: Any) -> Any:
    """Convert an arbitrary Python object to a JSON-serializable canonical form.

    Rules
    -----
    - NumPy arrays → {__ndarray__: sha256(shape + dtype + sample)}
    - Primitive types (int, float, str, bool, None) → themselves
    - list/tuple → JSON array
    - dict → JSON object (sorted keys)
    - Everything else → str(type) + repr (best-effort)
    """
    if isinstance(obj, np.ndarray):
        h = hashlib.sha256()
        h.update(str(obj.shape).encode())
        h.update(str(obj.dtype).encode())
        flat = obj.ravel()
        sample = flat[: min(256, flat.size)].tobytes()
        h.update(sample)
        return {"__ndarray__": h.hexdigest()}

    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]

    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items())}

    if isinstance(obj, (set, frozenset)):
        return [_canonicalize(v) for v in sorted(obj, key=str)]

    # Best-effort for other types
    try:
        return repr(obj)
    except Exception:
        return str(type(obj).__name__)


def make_key(*args: Any, **kwargs: Any) -> str:
    """Generate a 32-char hex key from function arguments.

    Parameters
    ----------
    *args, **kwargs :
        The same arguments passed to the decorated function.

    Returns
    -------
    str
        A 32-character hexadecimal SHA-256 prefix, deterministic given
        identical arguments.
    """
    payload = {"args": [_canonicalize(a) for a in args],
               "kwargs": _canonicalize(kwargs)}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def make_source_hash(fn: Callable) -> str:
    """Hash the source code of a function for cache invalidation.

    When the function body changes, previously cached results are
    automatically invalidated.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = fn.__name__
    return hashlib.sha256(src.encode()).hexdigest()[:16]
