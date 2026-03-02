"""Tests for clankers.spaces."""

import numpy as np
import pytest

from clankers.spaces import Box, Dict, Discrete, space_from_dict


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


class TestDict:
    def test_creation(self):
        space = Dict(
            {
                "obs": Box(low=[0.0], high=[1.0]),
                "goal": Box(low=[-1.0, -1.0], high=[1.0, 1.0]),
            }
        )
        assert "obs" in space.spaces
        assert "goal" in space.spaces
        assert isinstance(space.spaces["obs"], Box)

    def test_creation_with_discrete(self):
        space = Dict(
            {
                "continuous": Box(low=[0.0], high=[1.0]),
                "discrete": Discrete(n=5),
            }
        )
        assert isinstance(space.spaces["continuous"], Box)
        assert isinstance(space.spaces["discrete"], Discrete)

    def test_sample(self):
        space = Dict(
            {
                "obs": Box(low=[0.0, 0.0], high=[1.0, 1.0]),
                "action": Discrete(n=3),
            }
        )
        rng = np.random.default_rng(42)
        for _ in range(50):
            s = space.sample(rng)
            assert isinstance(s, dict)
            assert set(s.keys()) == {"obs", "action"}
            assert space.contains(s)

    def test_sample_default_rng(self):
        space = Dict({"x": Box(low=[0.0], high=[1.0])})
        s = space.sample()
        assert isinstance(s, dict)
        assert "x" in s

    def test_contains_valid(self):
        space = Dict(
            {
                "obs": Box(low=[0.0], high=[1.0]),
                "goal": Box(low=[0.0], high=[1.0]),
            }
        )
        assert space.contains(
            {
                "obs": np.array([0.5], dtype=np.float32),
                "goal": np.array([0.8], dtype=np.float32),
            }
        )

    def test_contains_invalid_type(self):
        space = Dict({"obs": Box(low=[0.0], high=[1.0])})
        assert not space.contains("not a dict")

    def test_contains_wrong_keys(self):
        space = Dict({"obs": Box(low=[0.0], high=[1.0])})
        assert not space.contains({"wrong_key": np.array([0.5])})

    def test_contains_missing_key(self):
        space = Dict(
            {
                "obs": Box(low=[0.0], high=[1.0]),
                "goal": Box(low=[0.0], high=[1.0]),
            }
        )
        assert not space.contains({"obs": np.array([0.5])})

    def test_contains_extra_key(self):
        space = Dict({"obs": Box(low=[0.0], high=[1.0])})
        assert not space.contains(
            {
                "obs": np.array([0.5]),
                "extra": np.array([0.5]),
            }
        )

    def test_contains_out_of_bounds(self):
        space = Dict({"obs": Box(low=[0.0], high=[1.0])})
        assert not space.contains({"obs": np.array([2.0])})

    def test_repr(self):
        space = Dict({"obs": Box(low=[0.0], high=[1.0])})
        r = repr(space)
        assert "Dict(" in r
        assert "'obs'" in r

    def test_eq(self):
        s1 = Dict({"a": Box(low=[0.0], high=[1.0])})
        s2 = Dict({"a": Box(low=[0.0], high=[1.0])})
        assert s1 == s2

    def test_neq_different_keys(self):
        s1 = Dict({"a": Box(low=[0.0], high=[1.0])})
        s2 = Dict({"b": Box(low=[0.0], high=[1.0])})
        assert s1 != s2

    def test_neq_different_spaces(self):
        s1 = Dict({"a": Box(low=[0.0], high=[1.0])})
        s2 = Dict({"a": Discrete(n=5)})
        assert s1 != s2

    def test_eq_not_dict(self):
        s1 = Dict({"a": Box(low=[0.0], high=[1.0])})
        assert s1 != "not a dict"

    def test_from_dict(self):
        data = {
            "spaces": {
                "obs": {"Box": {"low": [0.0], "high": [1.0]}},
                "action": {"Discrete": {"n": 3}},
            }
        }
        space = Dict.from_dict(data)
        assert isinstance(space.spaces["obs"], Box)
        assert isinstance(space.spaces["action"], Discrete)

    def test_nested_dict(self):
        space = Dict(
            {
                "outer": Dict(
                    {
                        "inner": Box(low=[0.0], high=[1.0]),
                    }
                ),
                "flat": Discrete(n=2),
            }
        )
        rng = np.random.default_rng(42)
        s = space.sample(rng)
        assert isinstance(s["outer"], dict)
        assert "inner" in s["outer"]
        assert space.contains(s)

    def test_nested_dict_from_dict(self):
        data = {
            "spaces": {
                "outer": {
                    "Dict": {
                        "spaces": {
                            "inner": {"Box": {"low": [0.0], "high": [1.0]}},
                        }
                    }
                },
                "flat": {"Discrete": {"n": 2}},
            }
        }
        space = Dict.from_dict(data)
        assert isinstance(space.spaces["outer"], Dict)
        assert isinstance(space.spaces["outer"].spaces["inner"], Box)


class TestSpaceFromDict:
    def test_box_from_dict(self):
        data = {"low": [0.0], "high": [1.0]}
        space = space_from_dict(data)
        assert isinstance(space, Box)

    def test_discrete_from_dict(self):
        data = {"n": 5}
        space = space_from_dict(data)
        assert isinstance(space, Discrete)

    def test_dict_from_dict_serde(self):
        data = {
            "Dict": {
                "spaces": {
                    "obs": {"Box": {"low": [0.0, 0.0], "high": [1.0, 1.0]}},
                    "goal": {"Discrete": {"n": 4}},
                }
            }
        }
        space = space_from_dict(data)
        assert isinstance(space, Dict)
        assert isinstance(space.spaces["obs"], Box)
        assert isinstance(space.spaces["goal"], Discrete)

    def test_dict_from_dict_flat(self):
        data = {
            "spaces": {
                "obs": {"Box": {"low": [0.0], "high": [1.0]}},
            }
        }
        space = space_from_dict(data)
        assert isinstance(space, Dict)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot determine"):
            space_from_dict({"foo": "bar"})
