"""Tests for anabranch — non-intrusive disk cache."""

import tempfile
import time
import numpy as np
import pytest
from pathlib import Path

from anabranch.store import CacheStore


class TestCacheStore:
    def test_put_get_roundtrip(self):
        store = CacheStore(tempfile.mkdtemp(), async_write=False)
        data = {"a": np.array([1, 2, 3]), "b": "hello"}
        store.put("test1", data)
        result = store.get("test1")
        assert np.array_equal(result["a"], data["a"])
        assert result["b"] == "hello"

    def test_get_miss_returns_none(self):
        store = CacheStore(tempfile.mkdtemp(), async_write=False)
        assert store.get("nonexistent") is None

    def test_contains(self):
        store = CacheStore(tempfile.mkdtemp(), async_write=False)
        store.put("foo", [1, 2, 3])
        time.sleep(0.1)
        assert store.contains("foo")
        assert not store.contains("bar")

    def test_remove(self):
        store = CacheStore(tempfile.mkdtemp(), async_write=False)
        store.put("rm_me", 42)
        time.sleep(0.1)
        assert store.contains("rm_me")
        store.remove("rm_me")
        assert not store.contains("rm_me")

    def test_clear(self):
        store = CacheStore(tempfile.mkdtemp(), async_write=False)
        store.put("a", 1)
        store.put("b", 2)
        time.sleep(0.1)
        assert len(store) == 2
        store.clear()
        assert len(store) == 0

    def test_stats(self):
        store = CacheStore(tempfile.mkdtemp(), async_write=False)
        store.put("data", np.random.randn(500, 500))
        time.sleep(0.1)
        s = store.stats
        assert s["entries"] == 1
        # Random data doesn't compress → > 0 MB
        assert s["total_mb"] > 0
        assert "root" in s

    def test_numpy_array_roundtrip(self):
        store = CacheStore(tempfile.mkdtemp(), async_write=False)
        arr = np.random.randn(50, 30)
        store.put("arr", arr)
        time.sleep(0.1)
        result = store.get("arr")
        assert np.allclose(result, arr)

    def test_nested_dict_roundtrip(self):
        store = CacheStore(tempfile.mkdtemp(), backend="pickle", async_write=False)
        data = {
            "level1": {
                "arr": np.eye(3),
                "scalar": 3.14,
                "list": [1, 2, 3],
            }
        }
        store.put("nested", data)
        time.sleep(0.1)
        result = store.get("nested")
        assert np.allclose(result["level1"]["arr"], np.eye(3))
        assert result["level1"]["scalar"] == 3.14

    def test_repr(self):
        store = CacheStore(tempfile.mkdtemp())
        r = repr(store)
        assert "CacheStore" in r
        assert "MB" in r or "entries" in r

    def test_max_size_eviction(self):
        store = CacheStore(
            tempfile.mkdtemp(), async_write=False, max_size_gb=0.001  # 1 MB
        )
        # 200x200 float64 = 320KB raw, ~30KB compressed. 40 entries ≈ 1.2MB
        for i in range(40):
            store.put(f"big_{i}", np.random.randn(200, 200))
        time.sleep(0.1)
        # Should have evicted oldest entries
        assert store.get("big_0") is None or len(store) <= 25

    def test_persistence_across_instances(self):
        d = tempfile.mkdtemp()
        s1 = CacheStore(d, async_write=False)
        s1.put("persist", {"val": 99})
        time.sleep(0.1)

        s2 = CacheStore(d, async_write=False)
        assert s2.contains("persist")
        assert s2.get("persist")["val"] == 99
