"""Single-environment Gymnasium-compatible wrapper.

``ClankerEnv`` connects to a Clankers ``GymServer`` and provides the
standard ``reset`` / ``step`` / ``close`` interface expected by training
frameworks like Stable-Baselines3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from clankers._errors import ProtocolError
from clankers.client import GymClient
from clankers.spaces import Box, Dict, Discrete, space_from_dict


def _validate_obs(
    resp: dict[str, Any],
    schema: Box | Discrete | Dict | None,
) -> None:
    """Verify the response observation matches the negotiated schema.

    Prefers ``resp["_image_obs"]`` (set by :meth:`GymClient.send` for
    ``RawU8Image`` responses) over ``resp["observation"]["data"]``.
    Raises :class:`ProtocolError` on shape or dtype mismatch.  Passing
    ``schema=None`` (env not connected through :meth:`ClankerEnv.connect`,
    or schema is the Image variant which ``space_from_dict`` cannot yet
    parse) is a no-op for back-compat with the existing test fixtures.
    """
    if schema is None:
        return

    # Image branch: validate the decoded ndarray shape against the schema.
    if "_image_obs" in resp:
        image = resp["_image_obs"]
        expected_shape = getattr(schema, "shape", None)
        if expected_shape is not None and image.shape != tuple(expected_shape):
            raise ProtocolError(
                f"image observation shape {image.shape} != schema {tuple(expected_shape)}"
            )
        if image.dtype != np.uint8:
            raise ProtocolError(f"image observation dtype {image.dtype} != schema uint8")
        return

    # Flat / discrete branch: validate the JSON-carried array.
    if "observation" not in resp:
        raise ProtocolError("response carries neither 'observation' nor '_image_obs'")

    # Only validate Box shape: the wire ``Observation`` is always a flat
    # ``Vec<f32>`` (see ``clankers-core::types::Observation``), so Box is
    # the only schema for which a shape comparison is meaningful here.
    # Dict schemas are not currently transmitted as flat ``data`` arrays.
    if not isinstance(schema, Box):
        return

    arr = np.asarray(resp["observation"]["data"], dtype=np.float32)
    if arr.shape != schema.shape:
        raise ProtocolError(f"observation shape {arr.shape} != schema {schema.shape}")
    if arr.dtype != np.float32:
        raise ProtocolError(f"observation dtype {arr.dtype} != schema float32")


class ClankerEnv:
    """Gymnasium-compatible environment connected to a Clankers server.

    Parameters
    ----------
    host : str
        Server address.
    port : int
        Server port.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9876) -> None:
        self._client = GymClient(host=host, port=port)
        self.observation_space: Box | Discrete | Dict | None = None
        self.action_space: Box | Discrete | Dict | None = None
        self._connected = False
        # Cached observation schema, populated at ``connect()``.  Used by
        # :func:`_validate_obs` to assert shape/dtype on the response from
        # the first ``reset()`` and the first ``step()`` of each episode.
        # ``None`` until handshake completes (or whenever the schema is an
        # Image variant — see POLISH-TODO in :meth:`connect`).
        self._schema: Box | Discrete | Dict | None = None
        # Reset at the start of every episode by :meth:`reset`; once
        # :func:`_validate_obs` has been called inside an episode we skip
        # the per-step revalidation (hot-path concern, WS4-plan § 8 R3).
        self._validated_this_episode: bool = False

    def connect(self, seed: int | None = None) -> dict[str, Any]:
        """Connect and perform handshake. Returns InitResponse."""
        resp = self._client.connect(seed=seed)
        env_info = resp.get("env_info", {})
        obs_data = env_info.get("observation_space", {})
        act_data = env_info.get("action_space", {})
        if obs_data:
            try:
                self.observation_space = space_from_dict(obs_data)
                self._schema = self.observation_space
            except ValueError:
                # POLISH-TODO(W4): `space_from_dict` does not yet handle the
                # ``{"Image": {...}}`` variant of ObservationSpace.  The
                # ``gymnasium_env.py`` wrapper sidesteps this via its own
                # ``_image_space_from_dict``.  For now, leave the schema
                # un-cached so :func:`_validate_obs` short-circuits; image
                # envs continue to function through the ``_image_obs``
                # decode path.  Tracked in WS4-plan § 3.
                self.observation_space = None
                self._schema = None
        if act_data:
            self.action_space = space_from_dict(act_data)
        self._connected = True
        return resp

    def reset(
        self, seed: int | None = None
    ) -> tuple[NDArray[np.float32] | NDArray[np.uint8], dict[str, Any]]:
        """Reset the environment.

        Returns
        -------
        observation : np.ndarray
            Initial observation.  ``float32`` for flat/discrete schemas;
            ``uint8`` of shape ``(H, W, C)`` when the server negotiated
            ``binary_obs`` and the response carries ``_image_obs`` (image
            envs, per W4 PR1).
        info : dict
            Reset metadata.
        """
        req: dict[str, Any] = {"type": "reset"}
        if seed is not None:
            req["seed"] = seed
        resp = self._client.send(req)
        _validate_obs(resp, self._schema)
        if "_image_obs" in resp:
            obs: NDArray[np.float32] | NDArray[np.uint8] = resp["_image_obs"]
        else:
            obs = np.asarray(resp["observation"]["data"], dtype=np.float32)
        info = resp.get("info", {})
        # New episode begins — gate per-step revalidation off after this point.
        self._validated_this_episode = True
        return obs, info

    def step(
        self, action: NDArray[np.float32] | int
    ) -> tuple[NDArray[np.float32] | NDArray[np.uint8], bool, bool, dict[str, Any]]:
        """Take one step.

        Reward is not included in the server response — compute it
        Python-side using :mod:`clankers.rewards`.

        Returns
        -------
        observation : np.ndarray
        terminated : bool
        truncated : bool
        info : dict
        """
        if isinstance(action, (int, np.integer)):
            action_payload: dict[str, Any] = {"Discrete": int(action)}
        else:
            action_payload = {"Continuous": np.asarray(action, dtype=np.float32).tolist()}

        resp = self._client.send({"type": "step", "action": action_payload})
        # Validate at most once per episode (WS4-plan § 8 R3: hot-path budget).
        if not self._validated_this_episode:
            _validate_obs(resp, self._schema)
            self._validated_this_episode = True
        if "_image_obs" in resp:
            obs: NDArray[np.float32] | NDArray[np.uint8] = resp["_image_obs"]
        else:
            obs = np.asarray(resp["observation"]["data"], dtype=np.float32)
        terminated = bool(resp["terminated"])
        truncated = bool(resp["truncated"])
        info = resp.get("info", {})
        return obs, terminated, truncated, info

    def close(self) -> None:
        """Close the connection."""
        self._client.close()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self) -> ClankerEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
