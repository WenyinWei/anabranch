"""Non-intrusive memoization decorator for scientific workflows.

The core API::

    from anabranch import memoize

    @memoize
    def run_simulation(params):
        # hours of computation ...
        return results

    # First call: runs simulation, caches to disk
    r = run_simulation(my_params)

    # Second call: instant cache hit, no recomputation
    r = run_simulation(my_params)

The decorator is **source-aware**: if you modify the function body, stale
cache entries are automatically invalidated.

For cross-project organization, use a session::

    from anabranch import memoize, Session

    with Session("project_alpha"):
        @memoize
        def build_matrix(n):
            ...

Session is a reentrant context manager — nested sessions compose cleanly.
"""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from anabranch.keys import make_key, make_source_hash
from anabranch.store import CacheStore

# ── Global default cache ──────────────────────────────────────────────────

_DEFAULT_ROOT = Path.home() / ".cache" / "anabranch"

_global_cache: CacheStore | None = None
_cache_lock = threading.Lock()

# ── Session stack ─────────────────────────────────────────────────────────

_session_stack: list[str] = []
_session_lock = threading.Lock()


def get_cache() -> CacheStore:
    """Return the global (thread-safe, lazy-init) cache instance."""
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = CacheStore(
                    _DEFAULT_ROOT, backend="npz", max_size_gb=20, async_write=True
                )
    return _global_cache


def set_cache_root(path: str | Path) -> None:
    """Change the default cache directory.

    Must be called before the first ``@memoize`` usage (or call
    ``clear_global_cache()`` first).
    """
    global _global_cache, _DEFAULT_ROOT
    _DEFAULT_ROOT = Path(path)
    _global_cache = None


def clear_global_cache() -> None:
    """Reset the global cache instance.  Does NOT delete cached files."""
    global _global_cache
    _global_cache = None


# ── Session (namespace) context manager ───────────────────────────────────

@contextmanager
def Session(namespace: str):
    """Context manager that prefixes all ``@memoize`` keys within its scope.

    Parameters
    ----------
    namespace : str
        A short identifier for the project or analysis phase, e.g.
        ``"east_efit"``, ``"parameter_scan_v2"``.

    Example
    -------
    >>> with Session("tokamak_response"):
    ...     @memoize
    ...     def compute_plasma_response(shot, time):
    ...         ...
    """
    _session_lock.acquire()
    _session_stack.append(namespace)
    _session_lock.release()
    try:
        yield
    finally:
        _session_lock.acquire()
        _session_stack.pop()
        _session_lock.release()


def get_session_namespace() -> str:
    """Return the current namespace (composed from the session stack)."""
    with _session_lock:
        return ":".join(_session_stack) if _session_stack else ""


# ── Memoize decorator ─────────────────────────────────────────────────────

def memoize(
    fn: Callable | None = None,
    *,
    namespace: str | None = None,
    ignore: list[str] | None = None,
    cache: CacheStore | None = None,
    source_aware: bool = True,
) -> Callable:
    """Disk-backed memoization decorator.  Zero configuration needed.

    Usage (decorator form)::

        @memoize
        def my_func(a, b, c=3):
            return expensive_work(a, b, c)

    Usage (explicit form)::

        @memoize(namespace="project_x", ignore=["verbose"])
        def my_func(data, verbose=False):
            return expensive_work(data)

    Parameters
    ----------
    fn : callable or None
        The function to decorate.  When called as ``@memoize`` (no
        parentheses), ``fn`` is the function.  When called with keyword
        arguments like ``@memoize(namespace=...)``, ``fn`` is None and
        we return the actual decorator.
    namespace : str or None
        Explicit namespace prefix.  Overrides the session stack.
        If None, uses the active session (via ``with Session(...)``).
    ignore : list of str or None
        Keyword argument names to exclude from cache key generation.
        Useful for verbosity flags, loggers, etc. that don't affect output.
    cache : CacheStore or None
        Use a specific cache instance instead of the global default.
    source_aware : bool
        If True (default), the function's source code is included in the
        cache key, so editing the function automatically invalidates
        stale cache entries.

    Returns
    -------
    callable
        The wrapped function with identical signature and return type.
    """
    # Support both @memoize and @memoize(...)
    if fn is not None:
        return _memoize_impl(fn, namespace, ignore, cache, source_aware)

    def decorator(f):
        return _memoize_impl(f, namespace, ignore, cache, source_aware)

    return decorator


def _memoize_impl(
    fn: Callable,
    namespace: str | None,
    ignore: list[str] | None,
    cache: CacheStore | None,
    source_aware: bool,
) -> Callable:
    """Internal: build the memoized wrapper."""
    cache = cache or get_cache()
    ignore_set = set(ignore or [])
    src_hash = make_source_hash(fn) if source_aware else ""
    
    # In-memory stash: when async_write is on, recently written results live
    # here so the next get() within the same process finds them instantly
    # even before the background thread finishes the disk write.
    _stash: dict[str, Any] = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Filter out ignored kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ignore_set}

        # Build cache key
        ns = namespace or get_session_namespace()
        arg_key = make_key(*args, **clean_kwargs)
        full_key = f"{ns}:{fn.__qualname__}:{src_hash}:{arg_key}" if ns else \
                   f"{fn.__qualname__}:{src_hash}:{arg_key}"

        # Check in-memory stash first (for async writes not yet on disk)
        if full_key in _stash:
            return _stash[full_key]

        # Try disk cache
        cached = cache.get(full_key)
        if cached is not None:
            _stash[full_key] = cached
            return cached

        # Compute and cache
        result = fn(*args, **kwargs)
        _stash[full_key] = result
        cache.put(full_key, result)
        return result

    # Expose cache control on the wrapper
    wrapper._anabranch_key_fn = fn.__qualname__  # type: ignore[attr-defined]
    wrapper._anabranch_clear = lambda: _clear_fn_entries(cache, fn.__qualname__, src_hash)  # type: ignore[attr-defined]

    return wrapper


def _clear_fn_entries(cache: CacheStore, qualname: str, src_hash: str) -> int:
    """Clear all cache entries for a given function.  Returns count removed."""
    prefix = f"{qualname}:{src_hash}:"
    count = 0
    with cache._lock:
        for key in list(cache._index):
            if prefix in key:
                cache._evict(key)
                count += 1
    return count


# ── Public re-exports for convenience ─────────────────────────────────────

__all__ = [
    "memoize",
    "Session",
    "get_cache",
    "set_cache_root",
    "clear_global_cache",
    "CacheStore",
]
