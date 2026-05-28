"""Observation and action space definitions compatible with Gymnasium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Box:
    """Continuous space with element-wise bounds.

    Matches ``gymnasium.spaces.Box`` semantics. Observations and actions
    are ``np.float32`` arrays of shape ``(dim,)``.

    Parameters
    ----------
    low : array-like
        Lower bounds per dimension.
    high : array-like
        Upper bounds per dimension.
    """

    low: NDArray[np.float32]
    high: NDArray[np.float32]

    def __init__(self, low: Any, high: Any) -> None:
        object.__setattr__(self, "low", np.asarray(low, dtype=np.float32))
        object.__setattr__(self, "high", np.asarray(high, dtype=np.float32))
        if self.low.shape != self.high.shape:
            msg = f"low shape {self.low.shape} != high shape {self.high.shape}"
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape

    @property
    def dim(self) -> int:
        return int(np.prod(self.low.shape))

    def contains(self, x: NDArray[np.float32]) -> bool:
        arr = np.asarray(x, dtype=np.float32)
        return bool(
            arr.shape == self.shape and np.all(arr >= self.low) and np.all(arr <= self.high)
        )

    def sample(self, rng: np.random.Generator | None = None) -> NDArray[np.float32]:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(np.float32)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Box:
        return cls(low=data["low"], high=data["high"])


@dataclass(frozen=True)
class Discrete:
    """Discrete space with n choices ``{0, 1, ..., n-1}``.

    Matches ``gymnasium.spaces.Discrete`` semantics.

    Parameters
    ----------
    n : int
        Number of discrete choices.
    """

    n: int

    def __post_init__(self) -> None:
        if self.n < 1:
            msg = f"n must be >= 1, got {self.n}"
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[()]:
        return ()  # type: ignore[return-value]

    @property
    def dim(self) -> int:
        return 1

    def contains(self, x: int) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def sample(self, rng: np.random.Generator | None = None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.integers(0, self.n))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Discrete:
        return cls(n=data["n"])


@dataclass(frozen=True)
class MultiDiscrete:
    """Vector of discrete spaces with per-element cardinality.

    Matches ``gymnasium.spaces.MultiDiscrete`` and Rust
    ``ObservationSpace::MultiDiscrete { nvec }`` / ``ActionSpace::MultiDiscrete``.
    The observation/action is a 1-D integer array of length ``len(nvec)``
    where element ``i`` is in ``{0, 1, ..., nvec[i] - 1}``.

    Added under CODE_QUALITY_REVIEW P1.5 — the Python client previously
    only modelled Box / Discrete / Dict, so newer Rust envs exposing
    MultiDiscrete spaces could not be parsed by Python.

    Parameters
    ----------
    nvec : array-like of int
        Per-element cardinalities.
    """

    nvec: NDArray[np.int64]

    def __init__(self, nvec: Any) -> None:
        arr = np.asarray(nvec, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(f"nvec must be 1-D, got shape {arr.shape}")
        if np.any(arr < 1):
            raise ValueError(f"every entry of nvec must be >= 1, got {arr.tolist()}")
        object.__setattr__(self, "nvec", arr)

    @property
    def shape(self) -> tuple[int, ...]:
        return (int(self.nvec.shape[0]),)

    @property
    def dim(self) -> int:
        return int(self.nvec.shape[0])

    def contains(self, x: Any) -> bool:
        arr = np.asarray(x)
        return bool(
            arr.shape == self.shape
            and np.issubdtype(arr.dtype, np.integer)
            and np.all(arr >= 0)
            and np.all(arr < self.nvec)
        )

    def sample(self, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(0, self.nvec).astype(np.int64)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiDiscrete:
        return cls(nvec=data["nvec"])


@dataclass(frozen=True)
class MultiBinary:
    """Vector of ``n`` binary values, each in ``{0, 1}``.

    Matches ``gymnasium.spaces.MultiBinary`` and Rust
    ``ObservationSpace::MultiBinary { n }`` / ``ActionSpace::MultiBinary``.

    Parameters
    ----------
    n : int
        Number of binary elements.
    """

    n: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n,)

    @property
    def dim(self) -> int:
        return self.n

    def contains(self, x: Any) -> bool:
        arr = np.asarray(x)
        return bool(arr.shape == (self.n,) and np.all((arr == 0) | (arr == 1)))

    def sample(self, rng: np.random.Generator | None = None) -> NDArray[np.int8]:
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(0, 2, size=self.n).astype(np.int8)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiBinary:
        return cls(n=data["n"])


@dataclass(frozen=True)
class Image:
    """Raw image observation space.

    Matches Rust ``ObservationSpace::Image { height, width, channels }``.
    Observations are ``np.uint8`` arrays of shape ``(height, width, channels)``
    in HWC layout — same as the existing binary-frame decode path in
    ``GymClient`` (P1.5 centralises the model definition; the
    decoder in ``gymnasium_env.py`` continues to produce the same array
    shape and dtype).

    Parameters
    ----------
    height, width : int
        Image dimensions in pixels.
    channels : int
        Number of channels (typically 1 for grayscale, 3 for RGB, 4 for RGBA).
    """

    height: int
    width: int
    channels: int

    def __post_init__(self) -> None:
        dims = (
            ("height", self.height),
            ("width", self.width),
            ("channels", self.channels),
        )
        for name, val in dims:
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    @property
    def dim(self) -> int:
        return self.height * self.width * self.channels

    def contains(self, x: Any) -> bool:
        arr = np.asarray(x)
        return bool(arr.shape == self.shape and arr.dtype == np.uint8)

    def sample(self, rng: np.random.Generator | None = None) -> NDArray[np.uint8]:
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(0, 256, size=self.shape, dtype=np.uint8)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Image:
        return cls(
            height=data["height"],
            width=data["width"],
            channels=data["channels"],
        )


class Dict:
    """Dictionary observation space containing named sub-spaces.

    Compatible with ``gymnasium.spaces.Dict`` for goal-conditioned RL (HER).

    Parameters
    ----------
    spaces : dict[str, "Box | Discrete | MultiDiscrete | MultiBinary | Image | Dict"]
        Named sub-spaces.
    """

    def __init__(self, spaces: dict[str, Any]) -> None:
        self.spaces = dict(spaces)

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Sample from each sub-space."""
        if rng is None:
            rng = np.random.default_rng()
        return {k: v.sample(rng) for k, v in self.spaces.items()}

    def contains(self, x: dict) -> bool:
        """Check if *x* is in this space."""
        if not isinstance(x, dict):
            return False
        if set(x.keys()) != set(self.spaces.keys()):
            return False
        return all(self.spaces[k].contains(x[k]) for k in x)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dict:
        """Deserialize from ``{"spaces": {"key": <space_dict>, ...}}``."""
        sub = data["spaces"]
        return cls({k: space_from_dict(v) for k, v in sub.items()})

    def __repr__(self) -> str:
        inner = ", ".join(f"{k!r}: {v!r}" for k, v in self.spaces.items())
        return f"Dict({{{inner}}})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dict):
            return NotImplemented
        return self.spaces == other.spaces


def space_from_dict(data: dict[str, Any]) -> Any:
    """Deserialize a space from a protocol dict.

    Handles both flat format ``{"low": ..., "high": ...}`` and
    Rust serde enum format ``{"Box": {"low": ..., "high": ...}}``.

    Covers every Rust ``ObservationSpace`` and ``ActionSpace`` variant
    after P1.5: ``Box``, ``Discrete``, ``MultiDiscrete``, ``MultiBinary``,
    ``Image``, and ``Dict``.
    """
    # Rust serde enum format: {"Box": {...}}, {"Discrete": {...}}, etc.
    if "Box" in data:
        return Box.from_dict(data["Box"])
    if "Discrete" in data:
        return Discrete.from_dict(data["Discrete"])
    if "MultiDiscrete" in data:
        return MultiDiscrete.from_dict(data["MultiDiscrete"])
    if "MultiBinary" in data:
        return MultiBinary.from_dict(data["MultiBinary"])
    if "Image" in data:
        return Image.from_dict(data["Image"])
    if "Dict" in data:
        return Dict.from_dict(data["Dict"])
    # Flat format (no enum wrapper).
    if "low" in data and "high" in data:
        return Box.from_dict(data)
    if "nvec" in data:
        return MultiDiscrete.from_dict(data)
    if "height" in data and "width" in data and "channels" in data:
        return Image.from_dict(data)
    if "n" in data:
        return Discrete.from_dict(data)
    if "spaces" in data:
        return Dict.from_dict(data)
    msg = f"Cannot determine space type from keys: {list(data.keys())}"
    raise ValueError(msg)
