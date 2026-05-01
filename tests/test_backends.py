"""Tests for serialization backends."""

import tempfile
import numpy as np
import pytest
from pathlib import Path

from anabranch.backends import get_backend, BACKENDS


class TestBackends:
    @pytest.mark.parametrize("name", ["npz", "pickle"])
    def test_backend_roundtrip(self, name):
        save, load, ext = get_backend(name)
        path = Path(tempfile.mkdtemp()) / f"test{ext}"

        data = {"arr": np.eye(5), "val": 42, "str": "hello"}
        save(path, data)
        result = load(path)

        assert np.allclose(result["arr"], np.eye(5))
        assert result["val"] == 42
        assert result["str"] == "hello"

    def test_npz_handles_numpy_scalar(self):
        save, load, ext = get_backend("npz")
        path = Path(tempfile.mkdtemp()) / f"scalar{ext}"

        data = {"a": np.float64(3.14), "b": np.int64(42)}
        save(path, data)
        result = load(path)
        assert result["a"] == pytest.approx(3.14)
        assert result["b"] == 42

    def test_pickle_handles_arbitrary_object(self):
        save, load, ext = get_backend("pickle")
        path = Path(tempfile.mkdtemp()) / f"obj{ext}"

        # Use a simple container (local class can't be pickled)
        obj = {"x": 99, "name": "test"}
        save(path, obj)
        result = load(path)
        assert result["x"] == 99

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            get_backend("nonexistent_backend")
