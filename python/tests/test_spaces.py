"""Tests for clankers.spaces."""

import numpy as np
import pytest

from clankers.spaces import (
    Box,
    Dict,
    Discrete,
    Image,
    MultiBinary,
    MultiDiscrete,
    space_from_dict,
)


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


# ---------------------------------------------------------------------------
# P1.5 — MultiDiscrete / MultiBinary / Image
# ---------------------------------------------------------------------------


class TestMultiDiscrete:
    def test_creation(self):
        space = MultiDiscrete(nvec=[3, 4, 5])
        assert space.shape == (3,)
        assert space.dim == 3

    def test_contains(self):
        space = MultiDiscrete(nvec=[2, 3])
        assert space.contains(np.array([0, 0]))
        assert space.contains(np.array([1, 2]))
        assert not space.contains(np.array([2, 0]))  # 2 == nvec[0]
        assert not space.contains(np.array([0, 3]))  # 3 == nvec[1]
        assert not space.contains(np.array([-1, 0]))
        # Wrong shape.
        assert not space.contains(np.array([0]))
        # Float values are not integers.
        assert not space.contains(np.array([0.5, 1.5]))

    def test_sample(self):
        space = MultiDiscrete(nvec=[3, 5, 2])
        rng = np.random.default_rng(42)
        for _ in range(100):
            s = space.sample(rng)
            assert space.contains(s)

    def test_invalid_nvec(self):
        with pytest.raises(ValueError, match=">= 1"):
            MultiDiscrete(nvec=[0, 2])
        with pytest.raises(ValueError, match="1-D"):
            MultiDiscrete(nvec=[[1, 2], [3, 4]])

    def test_from_dict_flat(self):
        space = space_from_dict({"nvec": [3, 4]})
        assert isinstance(space, MultiDiscrete)
        assert space.shape == (2,)

    def test_from_dict_serde(self):
        space = space_from_dict({"MultiDiscrete": {"nvec": [3, 4]}})
        assert isinstance(space, MultiDiscrete)


class TestMultiBinary:
    def test_creation(self):
        space = MultiBinary(n=8)
        assert space.shape == (8,)
        assert space.dim == 8

    def test_contains(self):
        space = MultiBinary(n=4)
        assert space.contains(np.array([0, 1, 0, 1]))
        assert space.contains(np.array([0, 0, 0, 0]))
        assert space.contains(np.array([1, 1, 1, 1]))
        assert not space.contains(np.array([0, 1, 2, 0]))
        # Wrong shape.
        assert not space.contains(np.array([0, 1]))

    def test_sample(self):
        space = MultiBinary(n=10)
        rng = np.random.default_rng(42)
        for _ in range(100):
            s = space.sample(rng)
            assert space.contains(s)

    def test_invalid_n(self):
        with pytest.raises(ValueError, match=">= 1"):
            MultiBinary(n=0)

    def test_from_dict_serde(self):
        space = space_from_dict({"MultiBinary": {"n": 4}})
        assert isinstance(space, MultiBinary)
        assert space.n == 4


class TestImage:
    def test_creation(self):
        space = Image(height=64, width=128, channels=3)
        assert space.shape == (64, 128, 3)
        assert space.dim == 64 * 128 * 3

    def test_contains(self):
        space = Image(height=8, width=8, channels=3)
        ok = np.zeros((8, 8, 3), dtype=np.uint8)
        assert space.contains(ok)
        # Wrong dtype.
        assert not space.contains(np.zeros((8, 8, 3), dtype=np.float32))
        # Wrong shape.
        assert not space.contains(np.zeros((8, 8), dtype=np.uint8))

    def test_sample(self):
        space = Image(height=4, width=4, channels=1)
        rng = np.random.default_rng(42)
        s = space.sample(rng)
        assert space.contains(s)
        assert s.dtype == np.uint8

    def test_invalid_dims(self):
        with pytest.raises(ValueError, match=">= 1"):
            Image(height=0, width=4, channels=3)

    def test_from_dict_flat(self):
        space = space_from_dict({"height": 16, "width": 16, "channels": 3})
        assert isinstance(space, Image)
        assert space.shape == (16, 16, 3)

    def test_from_dict_serde(self):
        space = space_from_dict({"Image": {"height": 4, "width": 4, "channels": 1}})
        assert isinstance(space, Image)


class TestAllVariantsRoundtrip:
    """P1.5: every Rust space variant deserialises through space_from_dict."""

    def test_box(self):
        assert isinstance(space_from_dict({"Box": {"low": [-1.0], "high": [1.0]}}), Box)

    def test_discrete(self):
        assert isinstance(space_from_dict({"Discrete": {"n": 3}}), Discrete)

    def test_multi_discrete(self):
        assert isinstance(space_from_dict({"MultiDiscrete": {"nvec": [2, 3]}}), MultiDiscrete)

    def test_multi_binary(self):
        assert isinstance(space_from_dict({"MultiBinary": {"n": 4}}), MultiBinary)

    def test_image(self):
        assert isinstance(
            space_from_dict({"Image": {"height": 4, "width": 4, "channels": 3}}),
            Image,
        )

    def test_dict_nests_multi(self):
        space = space_from_dict(
            {
                "Dict": {
                    "spaces": {
                        "categorical": {"MultiDiscrete": {"nvec": [2, 3]}},
                        "rgb": {"Image": {"height": 8, "width": 8, "channels": 3}},
                        "flags": {"MultiBinary": {"n": 4}},
                    }
                }
            }
        )
        assert isinstance(space, Dict)
        assert isinstance(space.spaces["categorical"], MultiDiscrete)
        assert isinstance(space.spaces["rgb"], Image)
        assert isinstance(space.spaces["flags"], MultiBinary)
