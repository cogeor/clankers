"""Tests for clanker_gym.spaces."""

import numpy as np
import pytest

from clanker_gym.spaces import Box, Discrete, space_from_dict


class TestBox:
    def test_creation(self):
        space = Box(low=[-1.0, -2.0], high=[1.0, 2.0])
        assert space.shape == (2,)
        assert space.dim == 2

    def test_from_numpy(self):
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        space = Box(low=low, high=high)
        assert space.shape == (2,)

    def test_contains(self):
        space = Box(low=[0.0, 0.0], high=[1.0, 1.0])
        assert space.contains(np.array([0.5, 0.5]))
        assert space.contains(np.array([0.0, 1.0]))
        assert not space.contains(np.array([-0.1, 0.5]))
        assert not space.contains(np.array([0.5, 1.1]))

    def test_contains_wrong_shape(self):
        space = Box(low=[0.0, 0.0], high=[1.0, 1.0])
        assert not space.contains(np.array([0.5]))
        assert not space.contains(np.array([0.5, 0.5, 0.5]))

    def test_sample(self):
        space = Box(low=[-1.0, -2.0], high=[1.0, 2.0])
        rng = np.random.default_rng(42)
        for _ in range(100):
            s = space.sample(rng)
            assert space.contains(s)
            assert s.dtype == np.float32

    def test_sample_default_rng(self):
        space = Box(low=[0.0], high=[1.0])
        s = space.sample()
        assert s.shape == (1,)

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Box(low=[0.0, 0.0], high=[1.0])

    def test_from_dict(self):
        data = {"low": [-1.0, -1.0], "high": [1.0, 1.0]}
        space = Box.from_dict(data)
        assert space.shape == (2,)

    def test_frozen(self):
        space = Box(low=[0.0], high=[1.0])
        with pytest.raises(AttributeError):
            space.low = np.array([2.0])  # type: ignore[misc]


class TestDiscrete:
    def test_creation(self):
        space = Discrete(n=5)
        assert space.n == 5
        assert space.dim == 1
        assert space.shape == ()

    def test_contains(self):
        space = Discrete(n=3)
        assert space.contains(0)
        assert space.contains(2)
        assert not space.contains(3)
        assert not space.contains(-1)

    def test_contains_numpy_int(self):
        space = Discrete(n=5)
        assert space.contains(np.int64(3))

    def test_sample(self):
        space = Discrete(n=4)
        rng = np.random.default_rng(42)
        for _ in range(100):
            s = space.sample(rng)
            assert space.contains(s)

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be"):
            Discrete(n=0)

    def test_from_dict(self):
        space = Discrete.from_dict({"n": 10})
        assert space.n == 10


class TestSpaceFromDict:
    def test_box_from_dict(self):
        data = {"low": [0.0], "high": [1.0]}
        space = space_from_dict(data)
        assert isinstance(space, Box)

    def test_discrete_from_dict(self):
        data = {"n": 5}
        space = space_from_dict(data)
        assert isinstance(space, Discrete)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot determine"):
            space_from_dict({"foo": "bar"})
