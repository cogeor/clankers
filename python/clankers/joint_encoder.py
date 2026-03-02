"""Generic robot joint position encoder for DL applications.

Provides deterministic alphabetic-order encoding of joint dictionaries to
fixed-length numpy vectors, and decoding back to named dicts.  Robot-agnostic:
works with any joint set.

Example::

    encoder = JointEncoder.from_dict({"elbow": 1.2, "shoulder": 0.5, "wrist": -0.3})
    vec = encoder.encode({"shoulder": 0.5, "elbow": 1.2, "wrist": -0.3})
    # vec = [1.2, 0.5, -0.3]  (alphabetic: elbow, shoulder, wrist)
    restored = encoder.decode(vec)
    # {"elbow": 1.2, "shoulder": 0.5, "wrist": -0.3}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_VERSION = "1.0.0"


class JointEncoder:
    """Deterministic alphabetic-order joint position encoder.

    Joint names are sorted alphabetically at construction time.  The resulting
    order is used for all encode/decode operations, ensuring consistent vector
    layout regardless of input dict ordering.
    """

    def __init__(self, joint_names: list[str]) -> None:
        if not joint_names:
            raise ValueError("joint_names must not be empty")
        sorted_names = sorted(set(joint_names))
        if len(sorted_names) != len(joint_names):
            raise ValueError("joint_names contains duplicates")
        self._names: tuple[str, ...] = tuple(sorted_names)
        self._index: dict[str, int] = {n: i for i, n in enumerate(self._names)}

    # -- Factories -----------------------------------------------------------

    @classmethod
    def from_dict(cls, joint_dict: dict[str, float]) -> JointEncoder:
        """Create encoder from a joint dictionary (keys become joint names)."""
        return cls(list(joint_dict.keys()))

    @classmethod
    def from_scene_spec(cls, path: str | Path) -> JointEncoder:
        """Create encoder from a SceneSpec JSON file (robot.joint_names)."""
        data = json.loads(Path(path).read_text())
        names = data.get("robot", {}).get("joint_names", [])
        if not names:
            raise ValueError(f"No robot.joint_names found in {path}")
        return cls(names)

    @classmethod
    def from_json(cls, s: str) -> JointEncoder:
        """Deserialize encoder from JSON metadata string."""
        data = json.loads(s)
        return cls(data["joint_names"])

    @classmethod
    def load(cls, path: str | Path) -> JointEncoder:
        """Load encoder metadata from a JSON file."""
        return cls.from_json(Path(path).read_text())

    # -- Properties ----------------------------------------------------------

    @property
    def dof(self) -> int:
        """Number of degrees of freedom (joints)."""
        return len(self._names)

    @property
    def names(self) -> tuple[str, ...]:
        """Sorted joint names (immutable)."""
        return self._names

    # -- Encode / Decode -----------------------------------------------------

    def encode(self, joint_dict: dict[str, float]) -> np.ndarray:
        """Encode a joint dictionary to a sorted float32 vector.

        All joint names in the encoder must be present in *joint_dict*.
        Extra keys in *joint_dict* are ignored.
        """
        try:
            return np.array([joint_dict[n] for n in self._names], dtype=np.float32)
        except KeyError as e:
            raise KeyError(f"Missing joint {e} in input dict") from None

    def decode(self, vec: np.ndarray) -> dict[str, float]:
        """Decode a vector back to a named joint dictionary."""
        if len(vec) != self.dof:
            raise ValueError(f"Expected vector of length {self.dof}, got {len(vec)}")
        return {n: float(vec[i]) for i, n in enumerate(self._names)}

    def encode_batch(self, dicts: list[dict[str, float]]) -> np.ndarray:
        """Encode a list of joint dicts to an (N, dof) float32 array."""
        return np.stack([self.encode(d) for d in dicts])

    # -- Serialization -------------------------------------------------------

    def to_json(self) -> str:
        """Serialize encoder metadata to JSON string."""
        return json.dumps(
            {"version": _VERSION, "joint_names": list(self._names)},
            indent=2,
        )

    def save(self, path: str | Path) -> None:
        """Save encoder metadata to a JSON file."""
        Path(path).write_text(self.to_json())

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"JointEncoder(dof={self.dof}, names={self._names})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JointEncoder):
            return NotImplemented
        return self._names == other._names
