"""Tests for clankers.joint_encoder."""

from __future__ import annotations

import json

import numpy as np
import pytest

from clankers.joint_encoder import JointEncoder


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_alphabetic_ordering():
    enc = JointEncoder(["wrist", "elbow", "shoulder"])
    assert enc.names == ("elbow", "shoulder", "wrist")
    assert enc.dof == 3


def test_empty_names_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        JointEncoder([])


def test_duplicate_names_raises():
    with pytest.raises(ValueError, match="duplicates"):
        JointEncoder(["a", "b", "a"])


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def test_from_dict():
    enc = JointEncoder.from_dict({"z_joint": 1.0, "a_joint": 2.0, "m_joint": 0.5})
    assert enc.names == ("a_joint", "m_joint", "z_joint")


def test_from_scene_spec(tmp_path):
    spec = {
        "robot": {"joint_names": ["wrist", "elbow", "shoulder"]},
        "simulation": {},
    }
    p = tmp_path / "scene.json"
    p.write_text(json.dumps(spec))
    enc = JointEncoder.from_scene_spec(p)
    assert enc.names == ("elbow", "shoulder", "wrist")


def test_from_scene_spec_missing_names(tmp_path):
    p = tmp_path / "scene.json"
    p.write_text(json.dumps({"robot": {}}))
    with pytest.raises(ValueError, match="No robot.joint_names"):
        JointEncoder.from_scene_spec(p)


# ---------------------------------------------------------------------------
# Encode / Decode
# ---------------------------------------------------------------------------


def test_encode_round_trip():
    enc = JointEncoder(["c", "a", "b"])
    original = {"a": 1.0, "b": 2.0, "c": 3.0}
    vec = enc.encode(original)
    assert vec.dtype == np.float32
    assert vec.shape == (3,)
    # Alphabetic order: a=1.0, b=2.0, c=3.0
    np.testing.assert_array_almost_equal(vec, [1.0, 2.0, 3.0])
    restored = enc.decode(vec)
    assert restored == pytest.approx(original)


def test_encode_ignores_extra_keys():
    enc = JointEncoder(["a", "b"])
    vec = enc.encode({"a": 1.0, "b": 2.0, "extra": 99.0})
    np.testing.assert_array_almost_equal(vec, [1.0, 2.0])


def test_encode_missing_key_raises():
    enc = JointEncoder(["a", "b"])
    with pytest.raises(KeyError, match="Missing joint.*'b'"):
        enc.encode({"a": 1.0})


def test_decode_wrong_length_raises():
    enc = JointEncoder(["a", "b", "c"])
    with pytest.raises(ValueError, match="Expected vector of length 3"):
        enc.decode(np.array([1.0, 2.0]))


def test_encode_batch():
    enc = JointEncoder(["x", "y"])
    dicts = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
    batch = enc.encode_batch(dicts)
    assert batch.shape == (2, 2)
    np.testing.assert_array_almost_equal(batch, [[1.0, 2.0], [3.0, 4.0]])


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_json_round_trip():
    enc = JointEncoder(["knee", "hip", "ankle"])
    s = enc.to_json()
    restored = JointEncoder.from_json(s)
    assert enc == restored


def test_save_load(tmp_path):
    enc = JointEncoder(["z", "a", "m"])
    p = tmp_path / "encoder.json"
    enc.save(p)
    loaded = JointEncoder.load(p)
    assert enc == loaded


def test_to_json_contains_version():
    enc = JointEncoder(["a"])
    data = json.loads(enc.to_json())
    assert "version" in data
    assert data["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Equality / Repr
# ---------------------------------------------------------------------------


def test_equality():
    a = JointEncoder(["b", "a"])
    b = JointEncoder(["a", "b"])
    assert a == b


def test_inequality():
    a = JointEncoder(["a", "b"])
    b = JointEncoder(["a", "c"])
    assert a != b


def test_repr():
    enc = JointEncoder(["b", "a"])
    r = repr(enc)
    assert "dof=2" in r
    assert "('a', 'b')" in r
