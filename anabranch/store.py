"""Disk-backed cache store with async writes and LRU eviction.

CacheStore
----------
The core cache engine.  Manages a directory of serialized cache entries
with content-addressable keys, optional TTL, and size-based eviction.

Features
--------
- **Async writes**: cache ``put()`` returns immediately; writes happen in
  a daemon thread so I/O never blocks the main computation.
- **LRU eviction**: when the cache exceeds ``max_size_gb``, oldest entries
  are removed automatically.
- **TTL support**: entries can expire after a configurable time-to-live.
- **Thread-safe**: all index operations are protected by a lock.
- **Crash-resilient**: writes are atomic (write to temp file, then rename).

Usage
-----
>>> from anabranch import CacheStore
>>> cache = CacheStore("/tmp/my_cache", max_size_gb=5)
>>> cache.put("experiment_42", large_result_dict)
>>> data = cache.get("experiment_42")  # instant on second call
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anabranch.backends import get_backend, Backend


@dataclass
class _Entry:
    key: str
    path: Path
    created: float
    size_bytes: int
    ttl: float | None = None

    @property
    def expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created) > self.ttl

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


class CacheStore:
    """Filesystem cache with async writes and LRU eviction.

    Parameters
    ----------
    root : str or Path
        Cache directory.  Created if it doesn't exist.
    backend : str
        Serialization format: ``"npz"`` (default), ``"pickle"``, ``"hdf5"``.
    max_size_gb : float or None
        Maximum total cache size in GB.  Oldest entries evicted when
        exceeded.  ``None`` = unlimited.
    ttl_seconds : float or None
        Default time-to-live for new entries.  ``None`` = never expire.
    async_write : bool
        If True (default), writes happen in background threads.
    """

    def __init__(
        self,
        root: str | Path,
        backend: str = "npz",
        max_size_gb: float | None = 20.0,
        ttl_seconds: float | None = None,
        async_write: bool = True,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_size_gb = max_size_gb
        self.ttl_seconds = ttl_seconds
        self.async_write = async_write

        self._save_fn, self._load_fn, self._ext = get_backend(backend)
        self._backend_name = backend

        self._index: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()

        # Load existing entries from disk
        self._scan()

    # ── Public API ────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Read from cache.

        Returns
        -------
        Any or None
            The cached data, or ``None`` if the key is not found or the
            entry has expired.
        """
        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                return None
            if entry.expired:
                self._evict(key)
                return None
            if not entry.path.exists():
                self._evict(key)
                return None
            # Promote to most-recently-used
            self._index.move_to_end(key)

        try:
            return self._load_fn(entry.path)
        except Exception:
            with self._lock:
                self._evict(key)
            return None

    def put(self, key: str, data: Any) -> None:
        """Store data in cache.

        When ``async_write=True`` (default), returns immediately and
        writes in a background thread.
        """
        if self.async_write:
            t = threading.Thread(
                target=self._put_sync, args=(key, data), daemon=True
            )
            t.start()
        else:
            self._put_sync(key, data)

    def _put_sync(self, key: str, data: Any) -> None:
        """Synchronous write (used by async thread or direct call)."""
        # Write to temp file first, then rename → atomic on most filesystems
        final_path = self.root / f"{key}{self._ext}"
        tmp_base = self.root / f".{key}.tmp"
        self._save_fn(tmp_base, data)
        # numpy.savez_compressed appends .npz → look for the actual output file
        actual_tmp = tmp_base
        for candidate in [tmp_base, tmp_base.with_suffix(self._ext),
                          Path(str(tmp_base) + self._ext)]:
            if candidate.exists():
                actual_tmp = candidate
                break
        if not actual_tmp.exists():
            raise FileNotFoundError(f"Save did not produce output: {tmp_base}")
        os.replace(actual_tmp, final_path)

        with self._lock:
            self._index[key] = _Entry(
                key=key,
                path=final_path,
                created=time.time(),
                size_bytes=final_path.stat().st_size,
                ttl=self.ttl_seconds,
            )
            self._index.move_to_end(key)
            self._maybe_evict()

    def contains(self, key: str) -> bool:
        """Check if a key exists and is valid (not expired, file exists)."""
        with self._lock:
            entry = self._index.get(key)
            if entry is None or entry.expired:
                return False
            return entry.path.exists()

    def remove(self, key: str) -> None:
        """Remove a single entry (index + disk)."""
        with self._lock:
            self._evict(key)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            for key in list(self._index):
                self._evict(key)

    # ── Info ──────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return cache statistics as a dict."""
        with self._lock:
            entries = len(self._index)
            total_bytes = sum(e.size_bytes for e in self._index.values())
        return {
            "entries": entries,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "root": str(self.root),
            "backend": self._backend_name,
            "async_write": self.async_write,
        }

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"CacheStore({s['entries']} entries, {s['total_mb']:.1f} MB, "
            f"root={s['root']})"
        )

    def __len__(self) -> int:
        return self.stats["entries"]

    # ── Internal ──────────────────────────────────────────────────────

    def _scan(self) -> None:
        """Scan cache directory for existing entries on startup."""
        for path in sorted(self.root.glob(f"*{self._ext}")):
            key = path.stem
            self._index[key] = _Entry(
                key=key,
                path=path,
                created=path.stat().st_mtime,
                size_bytes=path.stat().st_size,
                ttl=self.ttl_seconds,
            )

    def _evict(self, key: str) -> None:
        """Remove one entry from index and disk."""
        entry = self._index.pop(key, None)
        if entry and entry.path.exists():
            try:
                entry.path.unlink()
            except OSError:
                pass

    def _maybe_evict(self) -> None:
        """Evict oldest entries if total size exceeds max_size_gb."""
        if self.max_size_gb is None:
            return
        max_bytes = self.max_size_gb * 1024**3
        total = sum(e.size_bytes for e in self._index.values())
        while total > max_bytes and self._index:
            _, oldest = self._index.popitem(last=False)
            total -= oldest.size_bytes
            if oldest.path.exists():
                try:
                    oldest.path.unlink()
                except OSError:
                    pass
