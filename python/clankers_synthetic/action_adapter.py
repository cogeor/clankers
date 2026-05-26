"""Action-semantics adapters.

Bridge the synthetic compiler's absolute-joint-position output to the
four :class:`clankers_core::types::ActionSemantics` arms negotiated with
the env. Each adapter implements :meth:`ActionAdapter.to_env_action` and
:meth:`ActionAdapter.from_env_action`; the compiler dispatches through
the adapter selected by the constructor argument
``action_semantics: str``.

The four arms correspond verbatim to the W1
:class:`clankers_core::types::ActionSemantics` enum:

- ``NormalizedPosition`` — ``(j - centers) / half_ranges``, clamped to
  ``[-1, 1]``. The default, matching the W1 single-arm-pick env.
- ``AbsoluteJointPosition`` — identity (joint targets are the action).
- ``JointVelocity`` — finite-difference between successive targets,
  divided by ``control_dt``. Stateful.
- ``Torque`` — opaque; the synthetic compiler does not produce torques.
  The adapter exists so the handshake can negotiate the variant; calls
  to :meth:`to_env_action` raise :class:`NotImplementedError`.

Adding a new arm is additive: subclass :class:`ActionAdapter`, extend
:func:`select_adapter`, and ensure the new literal is appended (not
inserted) to the :data:`clankers_synthetic.specs.ActionSemantics` alias
so downstream consumers can pattern-match without breaking.
"""

from __future__ import annotations

import numpy as np


class ActionAdapter:
    """Abstract base for action-semantics adapters."""

    semantics: str = ""

    def to_env_action(self, joint_targets: np.ndarray) -> np.ndarray:
        """Convert absolute joint-position targets to the env's expected action."""
        raise NotImplementedError

    def from_env_action(self, action: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`to_env_action` (best-effort).

        Some semantics are lossy — ``Torque`` has no analytic inverse to
        joint positions, so the inverse returns the action verbatim.
        """
        raise NotImplementedError


class NormalizedPositionAdapter(ActionAdapter):
    """Normalised position: ``(j - centers) / half_ranges``, clamped to ``[-1, 1]``.

    Matches the W1 default for the current single-arm-pick env. Clamping
    preserves the pre-W8 behaviour for clients that previously called
    ``SkillCompiler.normalize_action`` directly.
    """

    semantics = "NormalizedPosition"

    def __init__(
        self,
        joint_centers: np.ndarray | list[float],
        joint_half_ranges: np.ndarray | list[float],
    ) -> None:
        self._centers = np.asarray(joint_centers, dtype=float)
        self._half_ranges = np.asarray(joint_half_ranges, dtype=float)
        # Guard against zero-range joints (would produce inf on divide).
        # SkillCompiler already enforces a 1e-6 floor; mirror that here so
        # the adapter can be used standalone.
        self._half_ranges = np.where(
            np.abs(self._half_ranges) < 1e-6,
            1e-6,
            self._half_ranges,
        )

    def to_env_action(self, joint_targets: np.ndarray) -> np.ndarray:
        targets = np.asarray(joint_targets, dtype=float)
        normalised = (targets - self._centers) / self._half_ranges
        return np.clip(normalised, -1.0, 1.0).astype(np.float32)

    def from_env_action(self, action: np.ndarray) -> np.ndarray:
        act = np.asarray(action, dtype=float)
        return (self._centers + act * self._half_ranges).astype(np.float32)


class AbsoluteJointPositionAdapter(ActionAdapter):
    """Identity adapter: the env consumes absolute joint positions directly."""

    semantics = "AbsoluteJointPosition"

    def to_env_action(self, joint_targets: np.ndarray) -> np.ndarray:
        return np.asarray(joint_targets, dtype=np.float32).copy()

    def from_env_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32).copy()


class JointVelocityAdapter(ActionAdapter):
    """Finite-difference velocity adapter.

    Stateful: tracks the previous joint targets so successive
    :meth:`to_env_action` calls produce ``(targets - last) / dt``. The
    first call returns zeros (no prior frame).
    """

    semantics = "JointVelocity"

    def __init__(self, control_dt: float = 0.02) -> None:
        if control_dt <= 0.0:
            raise ValueError(f"control_dt must be positive, got {control_dt}")
        self._dt = float(control_dt)
        self._last_positions: np.ndarray | None = None

    def to_env_action(self, joint_targets: np.ndarray) -> np.ndarray:
        targets = np.asarray(joint_targets, dtype=float)
        if self._last_positions is None:
            self._last_positions = targets.copy()
            return np.zeros_like(targets, dtype=np.float32)
        velocity = (targets - self._last_positions) / self._dt
        self._last_positions = targets.copy()
        return velocity.astype(np.float32)

    def from_env_action(self, action: np.ndarray) -> np.ndarray:
        """Integrate the velocity by one ``control_dt`` step.

        Updates the internal ``_last_positions`` accumulator so successive
        calls produce coherent absolute positions. If the integrator has
        not been primed (first call), starts from zero.
        """
        act = np.asarray(action, dtype=float)
        base = self._last_positions if self._last_positions is not None else np.zeros_like(act)
        nxt = base + act * self._dt
        self._last_positions = nxt.copy()
        return nxt.astype(np.float32)


class TorqueAdapter(ActionAdapter):
    """Opaque torque adapter.

    The synthetic compiler produces IK-derived position targets, not
    torques. :meth:`to_env_action` raises :class:`NotImplementedError`
    so torque-only envs surface the mismatch loudly during compilation.
    The adapter exists so the gym handshake can still negotiate the
    semantics; the compiler is expected to reject plans bound to a
    torque env before reaching the step loop.
    """

    semantics = "Torque"

    def to_env_action(self, joint_targets: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "TorqueAdapter cannot convert joint-position targets to torques. "
            "The synthetic compiler emits position targets; a torque-only env "
            "must be driven by a torque-producing controller (e.g. an MPC)."
        )

    def from_env_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32).copy()


_KNOWN_SEMANTICS: tuple[str, ...] = (
    "NormalizedPosition",
    "AbsoluteJointPosition",
    "JointVelocity",
    "Torque",
)


def select_adapter(
    semantics: str,
    joint_centers: np.ndarray | list[float] | None = None,
    joint_half_ranges: np.ndarray | list[float] | None = None,
    control_dt: float = 0.02,
) -> ActionAdapter:
    """Dispatch on ``semantics`` and return the matching adapter.

    Raises :class:`ValueError` on an unknown literal so the compiler
    surfaces handshake-vs-config drift loudly.
    """
    if semantics == "NormalizedPosition":
        if joint_centers is None or joint_half_ranges is None:
            raise ValueError(
                "NormalizedPosition adapter requires joint_centers and joint_half_ranges; got None."
            )
        return NormalizedPositionAdapter(joint_centers, joint_half_ranges)
    if semantics == "AbsoluteJointPosition":
        return AbsoluteJointPositionAdapter()
    if semantics == "JointVelocity":
        return JointVelocityAdapter(control_dt=control_dt)
    if semantics == "Torque":
        return TorqueAdapter()
    raise ValueError(f"unknown action semantics: {semantics!r}; expected one of {_KNOWN_SEMANTICS}")
