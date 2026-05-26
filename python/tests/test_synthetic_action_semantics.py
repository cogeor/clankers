"""Action-semantics + move_relative rejection tests (W8 PR3).

These tests pin the WS8-plan § 6 contracts:

1. ``ExecutionTrace`` / ``TraceStep`` REJECT instances without
   ``action_semantics`` (pydantic ValidationError).
2. ``SkillCompiler.execute`` REJECTS ``move_relative`` plans when no FK
   source is available (typed ``MoveRelativeWithoutFkError``).
3. The legacy opt-in ``SkillCompiler(legacy_move_relative=True)`` keeps
   the pre-W8 path with a ``DeprecationWarning``.
4. The :class:`NormalizedPositionAdapter` clamps to ``[-1, 1]``.
5. :func:`select_adapter` dispatches on the literal semantics string.
"""

from __future__ import annotations

import numpy as np
import pydantic
import pytest

from clankers_synthetic.action_adapter import (
    AbsoluteJointPositionAdapter,
    JointVelocityAdapter,
    NormalizedPositionAdapter,
    TorqueAdapter,
    select_adapter,
)
from clankers_synthetic.compiler import SkillCompiler
from clankers_synthetic.errors import MoveRelativeWithoutFkError
from clankers_synthetic.specs import (
    CanonicalPlan,
    ExecutionTrace,
    ResolvedSkill,
    TraceStep,
)

pytestmark = pytest.mark.synthetic

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_trace_without_action_semantics_is_rejected() -> None:
    """ExecutionTrace without action_semantics raises ValidationError."""
    with pytest.raises(pydantic.ValidationError) as exc:
        ExecutionTrace.model_validate(
            {
                "plan_id": "x",
                "steps": [],
                "total_reward": 0.0,
                "terminated": False,
                "truncated": False,
            }
        )
    assert "action_semantics" in str(exc.value)


def test_trace_step_without_action_semantics_is_rejected() -> None:
    """TraceStep without action_semantics raises ValidationError."""
    with pytest.raises(pydantic.ValidationError) as exc:
        TraceStep.model_validate(
            {
                "obs": [0.0],
                "action": [0.0],
                "next_obs": [0.0],
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
            }
        )
    assert "action_semantics" in str(exc.value)


def test_trace_step_invalid_semantics_literal_is_rejected() -> None:
    """TraceStep with a bogus semantics literal raises ValidationError."""
    with pytest.raises(pydantic.ValidationError):
        TraceStep.model_validate(
            {
                "obs": [0.0],
                "action": [0.0],
                "next_obs": [0.0],
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "action_semantics": "Nonsense",
            }
        )


# ---------------------------------------------------------------------------
# move_relative rejection tests
# ---------------------------------------------------------------------------


class _NoFkEnv:
    """Minimal env with NO forward_kinematics attr and empty info dicts."""

    def __init__(self, n_obs: int = 12) -> None:
        self.n_obs = n_obs

    def reset(self) -> tuple[np.ndarray, dict]:
        return np.zeros(self.n_obs, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        return np.zeros(self.n_obs, dtype=np.float32), 0.0, False, False, {}


_JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
_JOINT_LIMITS = {n: [-3.14, 3.14] for n in _JOINT_NAMES}


def _make_move_relative_plan() -> CanonicalPlan:
    return CanonicalPlan(
        plan_id="mr_test",
        skills=[
            ResolvedSkill(
                name="move_relative",
                params={"delta": [0.05, 0.0, 0.0]},
            )
        ],
    )


def test_move_relative_without_fk_is_rejected() -> None:
    """move_relative on an env without FK raises MoveRelativeWithoutFkError."""
    env = _NoFkEnv()
    compiler = SkillCompiler(
        ik_solver=None,
        joint_names=_JOINT_NAMES,
        joint_limits=_JOINT_LIMITS,
        action_semantics="NormalizedPosition",
    )
    plan = _make_move_relative_plan()
    with pytest.raises(MoveRelativeWithoutFkError):
        compiler.execute(plan, env)


def test_move_relative_legacy_opt_in_emits_deprecation_warning() -> None:
    """legacy_move_relative=True keeps pre-W8 path + emits DeprecationWarning."""
    env = _NoFkEnv()
    compiler = SkillCompiler(
        ik_solver=None,
        joint_names=_JOINT_NAMES,
        joint_limits=_JOINT_LIMITS,
        action_semantics="NormalizedPosition",
        legacy_move_relative=True,
    )
    plan = _make_move_relative_plan()
    with pytest.warns(DeprecationWarning):
        trace = compiler.execute(plan, env)
    # Plan executed (no exception); trace shape is well-formed.
    assert trace.plan_id == "mr_test"
    assert trace.action_semantics == "NormalizedPosition"


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


def test_normalized_position_adapter_clamps_to_range() -> None:
    """NormalizedPositionAdapter clamps over-range inputs to [-1, 1]."""
    adapter = NormalizedPositionAdapter([0.0] * 6, [1.0] * 6)
    out = adapter.to_env_action(np.array([2.0] * 6))
    assert np.allclose(out, np.ones(6))
    assert adapter.semantics == "NormalizedPosition"


def test_absolute_joint_position_adapter_is_identity() -> None:
    """AbsoluteJointPositionAdapter returns a float32 copy of the input."""
    x = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    adapter = AbsoluteJointPositionAdapter()
    out = adapter.to_env_action(x)
    assert out.dtype == np.float32
    assert np.allclose(out, x)


def test_joint_velocity_adapter_first_call_is_zero() -> None:
    """JointVelocityAdapter first call returns zeros (no prior frame)."""
    adapter = JointVelocityAdapter(0.02)
    assert adapter._last_positions is None
    out = adapter.to_env_action(np.array([0.1, 0.2, 0.3]))
    assert np.allclose(out, np.zeros(3))
    # Second call returns the finite-difference velocity.
    out2 = adapter.to_env_action(np.array([0.12, 0.22, 0.33]))
    assert np.allclose(out2, np.array([1.0, 1.0, 1.5]))


def test_torque_adapter_rejects_to_env_action() -> None:
    """TorqueAdapter.to_env_action raises NotImplementedError loudly."""
    adapter = TorqueAdapter()
    with pytest.raises(NotImplementedError):
        adapter.to_env_action(np.zeros(6))


def test_select_adapter_dispatches_on_semantics() -> None:
    """select_adapter returns the right subclass for each literal."""
    adapter = select_adapter(
        "NormalizedPosition",
        joint_centers=np.zeros(6),
        joint_half_ranges=np.ones(6),
    )
    assert isinstance(adapter, NormalizedPositionAdapter)
    assert adapter.semantics == "NormalizedPosition"

    assert isinstance(select_adapter("AbsoluteJointPosition"), AbsoluteJointPositionAdapter)
    assert isinstance(select_adapter("JointVelocity"), JointVelocityAdapter)
    assert isinstance(select_adapter("Torque"), TorqueAdapter)


def test_select_adapter_unknown_raises_value_error() -> None:
    """select_adapter raises ValueError on an unknown literal."""
    with pytest.raises(ValueError):
        select_adapter("Nonsense")
