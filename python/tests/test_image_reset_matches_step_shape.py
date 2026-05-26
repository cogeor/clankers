"""Verify image observations from ``reset()`` and ``step()`` agree on shape/dtype.

Pins the WS4-plan finding-#2 contract: image envs now satisfy the
Gymnasium space contract on ``reset()`` (server-side parity landed in
W4 PR1 / loop 03 via ``Response::Reset.obs_encoding``).  This loop
(W4 PR2) closes the wire-format break window by renaming the
client-side dispatcher tag from ``"RawU8"`` to ``"RawU8Image"``; the
test below pins the consumer-side contract end-to-end through
``ClankerEnv``.

Per the loop-04 directive GPU is temporarily off limits, so this test
uses ``MagicMock`` for the ``GymClient`` rather than a real Bevy
server.  The Rust over-the-wire path is covered separately by
``crates/clankers-gym/tests/protocol_image_reset.rs``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from clankers.env import ClankerEnv
from clankers.spaces import Box


def test_image_reset_matches_step_shape() -> None:
    """Image observation from ``reset()`` matches the one from ``step()``."""
    env = ClankerEnv.__new__(ClankerEnv)
    env._client = MagicMock()
    env._connected = True
    # Image schema cannot yet be parsed by ``space_from_dict`` (see
    # ``env.py`` POLISH-TODO in connect()).  ``_validate_obs`` therefore
    # short-circuits via the ``schema is None`` no-op, and the
    # ``_image_obs`` branch in reset()/step() returns the decoded ndarray.
    env._schema = None
    env._validated_this_episode = False
    env.observation_space = None
    env.action_space = Box(low=[-1.0], high=[1.0])

    pixels_reset = np.full((64, 64, 3), 255, dtype=np.uint8)
    pixels_step = np.full((64, 64, 3), 128, dtype=np.uint8)

    env._client.send.side_effect = [
        # reset response — image-on-reset (W4 PR1 server behaviour, loop 03)
        {
            "type": "reset",
            "observation": {"data": []},  # empty sentinel per from_reset_binary
            "info": {"seed": 0},
            "obs_encoding": {
                "type": "RawU8Image",
                "width": 64,
                "height": 64,
                "channels": 3,
                "layout": "Hwc",
            },
            "_image_obs": pixels_reset,
        },
        # step response — same image shape (Gymnasium contract)
        {
            "type": "step",
            "observation": {"data": []},
            "terminated": False,
            "truncated": False,
            "info": {},
            "obs_encoding": {
                "type": "RawU8Image",
                "width": 64,
                "height": 64,
                "channels": 3,
                "layout": "Hwc",
            },
            "_image_obs": pixels_step,
        },
    ]

    obs_reset, _ = env.reset()
    obs_step, _, _, _ = env.step(np.array([0.0], dtype=np.float32))

    assert obs_reset.shape == (64, 64, 3)
    assert obs_reset.dtype == np.uint8
    assert obs_reset.shape == obs_step.shape
    assert obs_reset.dtype == obs_step.dtype


def test_validate_obs_short_circuits_when_schema_is_none() -> None:
    """``_validate_obs`` is a no-op when the schema was never cached."""
    from clankers.env import _validate_obs

    # Image response with no schema → no raise.
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    _validate_obs({"_image_obs": pixels}, schema=None)
    # Flat response with no schema → no raise.
    _validate_obs({"observation": {"data": [0.0, 1.0, 2.0]}}, schema=None)


def test_validate_obs_rejects_shape_mismatch() -> None:
    """``_validate_obs`` raises ProtocolError when arr.shape != Box.shape."""
    import pytest

    from clankers import ProtocolError
    from clankers.env import _validate_obs

    schema = Box(low=[-1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0])  # shape (3,)
    with pytest.raises(ProtocolError, match="observation shape"):
        _validate_obs({"observation": {"data": [0.0, 1.0]}}, schema=schema)


def test_validate_obs_accepts_matching_box_shape() -> None:
    """``_validate_obs`` passes silently when shape and dtype match."""
    from clankers.env import _validate_obs

    schema = Box(low=[-1.0, -1.0], high=[1.0, 1.0])  # shape (2,)
    _validate_obs({"observation": {"data": [0.0, 1.0]}}, schema=schema)


def test_validate_obs_rejects_image_dtype_mismatch() -> None:
    """``_validate_obs`` raises when ``_image_obs`` dtype is not uint8."""
    import pytest

    from clankers import ProtocolError
    from clankers.env import _validate_obs

    # Synthesise a "schema" whose ``.shape`` matches; only dtype fails.
    pixels = np.zeros((8, 8, 3), dtype=np.float32)

    class _ShapeOnly:
        shape = (8, 8, 3)

    with pytest.raises(ProtocolError, match="image observation dtype"):
        _validate_obs({"_image_obs": pixels}, schema=_ShapeOnly())  # type: ignore[arg-type]
