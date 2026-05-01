"""anabranch — Non-intrusive disk cache for scientific computing workflows.

Usage
-----
>>> from anabranch import memoize
>>>
>>> @memoize
>>> def expensive_simulation(params):
...     # hours of number-crunching ...
...     return results

That's it.  The first call runs the function and caches to disk.
Subsequent calls with the same arguments return the cached result
instantly.  If you edit the function body, stale cache entries are
automatically invalidated.

For multi-project organization::

>>> from anabranch import Session, memoize
>>>
>>> with Session("tokamak_analysis"):
...     @memoize
...     def fetch_equilibrium(shot, time):
...         ...

Public API
----------
- ``@memoize`` — disk-backed memoization decorator
- ``Session`` — context manager for cache key namespacing
- ``CacheStore`` — low-level cache engine (advanced usage)
- ``set_cache_root`` — change default cache directory
"""

from anabranch.store import CacheStore
from anabranch.memoize import (
    memoize,
    Session,
    set_cache_root,
    clear_global_cache,
    get_cache,
)

__version__ = "0.1.0"
__all__ = [
    "memoize",
    "Session",
    "CacheStore",
    "set_cache_root",
    "clear_global_cache",
    "get_cache",
]
