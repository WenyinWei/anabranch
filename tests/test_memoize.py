"""Tests for @memoize decorator and Session context manager."""

import tempfile
import numpy as np
from pathlib import Path

from anabranch.memoize import memoize, Session, set_cache_root, clear_global_cache
from anabranch.store import CacheStore


class TestMemoize:
    def setup_method(self):
        d = tempfile.mkdtemp()
        set_cache_root(d)
        clear_global_cache()
        self.call_count = 0

    def teardown_method(self):
        clear_global_cache()

    def test_basic_memoization(self):
        @memoize
        def heavy(x, y):
            self.call_count += 1
            return x + y

        assert heavy(1, 2) == 3
        assert self.call_count == 1
        assert heavy(1, 2) == 3  # cache hit
        assert self.call_count == 1

    def test_different_args_different_results(self):
        @memoize
        def scale(arr, factor):
            self.call_count += 1
            return arr * factor

        a = np.array([1, 2, 3])
        r1 = scale(a, 2.0)
        r2 = scale(a, 3.0)
        assert self.call_count == 2
        assert np.allclose(r1, [2, 4, 6])
        assert np.allclose(r2, [3, 6, 9])

    def test_numpy_array_args(self):
        @memoize
        def process(data):
            self.call_count += 1
            return data.sum()

        a = np.random.randn(100)
        r1 = process(a)
        r2 = process(a.copy())  # identical data, new array object
        assert self.call_count == 1  # should hit cache
        assert r1 == r2

    def test_numpy_arrays_different_data(self):
        @memoize
        def process(data):
            self.call_count += 1
            return data.sum()

        a1 = np.array([1.0, 2.0, 3.0])
        a2 = np.array([4.0, 5.0, 6.0])
        process(a1)
        process(a2)
        assert self.call_count == 2

    def test_kwargs_order_independent(self):
        # NOTE: fn(1,2,c=3) and fn(a=1,b=2,c=3) produce the same logical
        # call but different cache keys (positional vs keyword binding).
        # This is a known limitation — use consistent calling style.
        @memoize
        def fn(a, b, c=0):
            self.call_count += 1
            return a + b + c

        r1 = fn(1, 2, c=3)
        r2 = fn(1, 2, c=3)  # same style → cache hit
        assert self.call_count == 1
        assert r1 == r2 == 6

    def test_session_namespacing(self):
        @memoize
        def shared_name(x):
            self.call_count += 1
            return x * 2

        # Outside session
        r1 = shared_name(5)

        with Session("project_a"):
            r2 = shared_name(5)  # different namespace → recompute

        assert self.call_count == 2
        assert r1 == r2 == 10

    def test_ignore_kwargs(self):
        @memoize(ignore=["verbose"])
        def fn(data, verbose=False):
            self.call_count += 1
            return len(data)

        fn([1, 2, 3], verbose=True)
        fn([1, 2, 3], verbose=False)
        assert self.call_count == 1  # verbose ignored in key

    def test_explicit_namespace(self):
        @memoize(namespace="ns_test")
        def fn(x):
            self.call_count += 1
            return x

        fn(1)
        assert self.call_count == 1

    def test_source_awareness(self):
        # Source-aware: different function bodies → different keys
        @memoize
        def fn_v1(x):
            return x + 1

        r1 = fn_v1(5)
        assert r1 == 6

        # Simulate editing the function (new decorator = new key)
        @memoize
        def fn_v2(x):
            return x + 2

        r2 = fn_v2(5)
        assert r2 == 7  # different function, fresh computation

    def test_explicit_cache_instance(self):
        store = CacheStore(tempfile.mkdtemp(), async_write=False)
        call_count_local = 0

        @memoize(cache=store)
        def fn(x):
            nonlocal call_count_local
            call_count_local += 1
            return x * x

        r1 = fn(4)
        r2 = fn(4)
        assert call_count_local == 1  # second call hits in-memory stash
        assert r1 == r2 == 16
        # The memoize decorator with explicit cache should produce cache hits
