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


def space_from_dict(data: dict[str, Any]) -> Box | Discrete:
    """Deserialize a space from a protocol dict.

    Handles both flat format ``{"low": ..., "high": ...}`` and
    Rust serde enum format ``{"Box": {"low": ..., "high": ...}}``.
    """
    # Rust serde enum format: {"Box": {...}} or {"Discrete": {...}}
    if "Box" in data:
        return Box.from_dict(data["Box"])
    if "Discrete" in data:
        return Discrete.from_dict(data["Discrete"])
    # Flat format
    if "low" in data and "high" in data:
        return Box.from_dict(data)
    if "n" in data:
        return Discrete.from_dict(data)
    msg = f"Cannot determine space type from keys: {list(data.keys())}"
    raise ValueError(msg)
