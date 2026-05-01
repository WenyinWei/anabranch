# anabranch

> Non-intrusive disk cache for scientific computing workflows — async branching for instant recall.

**One line. No boilerplate.**

```python
from anabranch import memoize

@memoize
def run_efit_reconstruction(shot, time):
    # 30 minutes of computation...
    return equilibrium
```

First call: runs the function, caches to disk.
Second call: instant cache hit. Edit the function? Old cache auto-invalidates.

## Why?

Scientific computing has a universal pain point: **re-running from scratch**
every time you tweak a downstream analysis step.  `anabranch` makes it
trivial to checkpoint expensive intermediate results so you can iterate
at the speed of a cache hit instead of the speed of your cluster.

### What makes it "non-intrusive"

- **Decorator-based**: one `@memoize` line, no code restructuring
- **Source-aware**: editing a function automatically invalidates its cache
- **Zero config**: sensible defaults out of the box
- **Type-transparent**: wrapped function preserves signature, type hints, and return type
- **Async I/O**: cache writes never block your computation
- **Sync-friendly**: cache directory can be synced via 坚果云, Dropbox, rsync

## Install

```bash
pip install anabranch
```

For HDF5 backend support:

```bash
pip install anabranch[hdf5]
```

## Quick start

### Basic memoization

```python
import numpy as np
from anabranch import memoize

@memoize
def compute_heavy_matrix(n, m):
    """This only runs once per (n, m) pair."""
    print(f"Computing {n}×{m} matrix...")
    return np.random.randn(n, m) @ np.random.randn(m, n)

# First call: prints "Computing..."
A = compute_heavy_matrix(1000, 500)

# Second call: instant (no print)
A = compute_heavy_matrix(1000, 500)

# Different args: new computation (prints again)
B = compute_heavy_matrix(2000, 500)
```

### Project namespacing with Session

```python
from anabranch import memoize, Session

with Session("east_campaign_2024"):
    @memoize
    def fetch_equilibrium(shot, time):
        return pull_from_mdsplus(shot, time)

with Session("parameter_scan"):
    @memoize
    def evaluate_config(params):
        return run_simulation(params)
```

### Ignoring non-semantic arguments

```python
@memoize(ignore=["verbose", "logger"])
def analyze_data(data, verbose=False, logger=None):
    # verbose and logger don't affect output → excluded from cache key
    return result
```

### Custom cache location

```python
import anabranch
anabranch.set_cache_root("/mnt/data/my_cache")
```

## How it works

1. **Key generation**: function arguments are hashed (SHA-256) to produce
   a deterministic 32-char hex key.  NumPy arrays are hashed by shape +
   dtype + data sample.
2. **Source hashing**: the function's source code is included in the key,
   so editing the function invalidates stale cache.
3. **Disk storage**: results are serialized via NumPy `.npz` (default),
   pickle, or HDF5 and written atomically (temp file → rename).
4. **Async writes**: `put()` returns immediately; the actual file I/O
   runs in a daemon thread.
5. **LRU eviction**: when the cache exceeds `max_size_gb`, the oldest
   entries are removed.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `root` | `~/.cache/anabranch` | Cache directory |
| `backend` | `"npz"` | Serialization format (`npz`, `pickle`, `hdf5`) |
| `max_size_gb` | `20.0` | Max cache size (None = unlimited) |
| `ttl_seconds` | `None` | Entry time-to-live (None = forever) |
| `async_write` | `True` | Background-thread writes |

## License

MIT
