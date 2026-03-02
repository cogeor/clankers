"""Tests for clankers_synthetic.parser — strict plan parser and canonicalization."""

from __future__ import annotations

import math

import numpy as np
import pytest

from clankers_synthetic.parser import PlanParser
from clankers_synthetic.specs import (
    CanonicalPlan,
    ConstraintSpec,
    ObjectSpec,
    ObservationSpec,
    PlanRejection,
    RobotSpec,
    SceneSpec,
    SimulationSpec,
)

# ---------------------------------------------------------------------------
# Shared fixture: minimal SceneSpec
# ---------------------------------------------------------------------------


@pytest.fixture()
def scene() -> SceneSpec:
    """Minimal scene with workspace [-0.5,-0.5,0.0] to [0.5,0.5,1.2],
    robot with ee_link_name='end_effector', and one object 'cube'."""
    return SceneSpec(
        scene_id="test_scene",
        simulation=SimulationSpec(),
        robot=RobotSpec(
            name="panda",
            urdf_path="/robots/panda.urdf",
            base_position=[0.0, 0.0, 0.0],
            base_orientation=[0.0, 0.0, 0.0, 1.0],
            ee_link_name="end_effector",
        ),
        objects=[
            ObjectSpec(
                name="cube",
                shape="box",
                shape_params={"half_extents": [0.025, 0.025, 0.025]},
                position=[0.2, 0.0, 0.8],
            ),
        ],
        constraints=ConstraintSpec(
            workspace_bounds_min=[-0.5, -0.5, 0.0],
            workspace_bounds_max=[0.5, 0.5, 1.2],
        ),
        observations=ObservationSpec(),
    )


@pytest.fixture()
def parser() -> PlanParser:
    return PlanParser(max_gripper_width=0.08)


# ---------------------------------------------------------------------------
# 1. test_valid_plan_parses
# ---------------------------------------------------------------------------


def test_valid_plan_parses(parser: PlanParser, scene: SceneSpec) -> None:
    """Full valid plan with move_to, move_linear, set_gripper, wait
    should produce a CanonicalPlan."""
    raw = {
        "plan_id": "plan_001",
        "plan_type": "pick_and_place",
        "rationale": "Approach, grasp, lift",
        "assumptions": ["cube is reachable"],
        "skills": [
            {
                "name": "move_to",
                "params": {
                    "target": {
                        "frame": "world",
                        "position": [0.2, 0.0, 0.8],
                    },
                    "speed_fraction": 0.3,
                },
            },
            {
                "name": "move_linear",
                "params": {
                    "direction": [0.0, 0.0, -1.0],
                    "distance": 0.05,
                },
            },
            {
                "name": "set_gripper",
                "params": {"width": 0.04},
            },
            {
                "name": "wait",
                "params": {"steps": 10},
            },
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, CanonicalPlan)
    assert result.plan_id == "plan_001"
    assert len(result.skills) == 4
    assert result.skills[0].name == "move_to"
    assert result.skills[1].name == "move_linear"
    assert result.skills[2].name == "set_gripper"
    assert result.skills[3].name == "wait"
    assert result.assumptions == ["cube is reachable"]
    assert result.metadata["plan_type"] == "pick_and_place"
    assert result.metadata["rationale"] == "Approach, grasp, lift"


# ---------------------------------------------------------------------------
# 2. test_unknown_skill_rejected
# ---------------------------------------------------------------------------


def test_unknown_skill_rejected(parser: PlanParser, scene: SceneSpec) -> None:
    """Skill 'fly_to' is not in vocabulary and should be rejected."""
    raw = {
        "plan_id": "bad_plan",
        "skills": [
            {
                "name": "fly_to",
                "params": {"target": [0, 0, 1]},
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, PlanRejection)
    assert "UNKNOWN_SKILL" in result.error_codes
    assert any("fly_to" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# 3. test_unknown_object_ref_rejected
# ---------------------------------------------------------------------------


def test_unknown_object_ref_rejected(parser: PlanParser, scene: SceneSpec) -> None:
    """Target references object 'mug' which is not in scene."""
    raw = {
        "plan_id": "bad_ref",
        "skills": [
            {
                "name": "move_to",
                "params": {
                    "target": {
                        "frame": "mug",
                        "position": [0.0, 0.0, 0.0],
                    },
                },
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, PlanRejection)
    assert any("unknown frame" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# 4. test_oob_target_rejected
# ---------------------------------------------------------------------------


def test_oob_target_rejected(parser: PlanParser, scene: SceneSpec) -> None:
    """Target position outside workspace bounds should be rejected."""
    raw = {
        "plan_id": "oob",
        "skills": [
            {
                "name": "move_to",
                "params": {
                    "target": {
                        "frame": "world",
                        "position": [0.0, 0.0, 2.0],  # Z above 1.2 max
                    },
                },
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, PlanRejection)
    assert any("outside workspace" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# 5. test_missing_required_params
# ---------------------------------------------------------------------------


def test_missing_required_params(parser: PlanParser, scene: SceneSpec) -> None:
    """move_to without 'target' param should be rejected."""
    raw = {
        "plan_id": "missing",
        "skills": [
            {
                "name": "move_to",
                "params": {"speed_fraction": 0.5},
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, PlanRejection)
    assert "MISSING_PARAMS" in result.error_codes


# ---------------------------------------------------------------------------
# 6. test_quaternion_normalization
# ---------------------------------------------------------------------------


def test_quaternion_normalization(parser: PlanParser, scene: SceneSpec) -> None:
    """Slightly non-unit quaternion (norm ~1.05) should be normalized."""
    # quaternion with norm slightly above 1.0
    quat = [0.0, 0.0, 0.0, 1.05]
    raw = {
        "plan_id": "quat_norm",
        "skills": [
            {
                "name": "move_to",
                "params": {
                    "target": {
                        "frame": "world",
                        "position": [0.1, 0.0, 0.5],
                    },
                    "orientation": {"quaternion": quat},
                },
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, CanonicalPlan)
    orient = result.skills[0].target_orientation
    assert orient is not None
    # Check that the quaternion is now unit norm
    norm = math.sqrt(sum(x * x for x in orient))
    assert abs(norm - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 7. test_quaternion_far_from_unit_rejected
# ---------------------------------------------------------------------------


def test_quaternion_far_from_unit_rejected(parser: PlanParser, scene: SceneSpec) -> None:
    """Quaternion with norm 0.5 should be rejected (too far from 1.0)."""
    raw = {
        "plan_id": "bad_quat",
        "skills": [
            {
                "name": "move_to",
                "params": {
                    "target": {
                        "frame": "world",
                        "position": [0.1, 0.0, 0.5],
                    },
                    "orientation": {
                        "quaternion": [0.0, 0.0, 0.0, 0.5],
                    },
                },
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, PlanRejection)
    assert any("quaternion norm" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# 8. test_speed_fraction_out_of_range
# ---------------------------------------------------------------------------


def test_speed_fraction_out_of_range(parser: PlanParser, scene: SceneSpec) -> None:
    """speed_fraction=1.5 should be rejected."""
    raw = {
        "plan_id": "fast",
        "skills": [
            {
                "name": "move_to",
                "params": {
                    "target": {
                        "frame": "world",
                        "position": [0.1, 0.0, 0.5],
                    },
                    "speed_fraction": 1.5,
                },
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, PlanRejection)
    assert "OUT_OF_RANGE" in result.error_codes
    assert any("speed_fraction" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# 9. test_gripper_width_out_of_range
# ---------------------------------------------------------------------------


def test_gripper_width_out_of_range(parser: PlanParser, scene: SceneSpec) -> None:
    """Gripper width=-0.1 should be rejected."""
    raw = {
        "plan_id": "neg_grip",
        "skills": [
            {
                "name": "set_gripper",
                "params": {"width": -0.1},
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, PlanRejection)
    assert "OUT_OF_RANGE" in result.error_codes
    assert any("gripper width" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# 10. test_default_speed_fraction_applied
# ---------------------------------------------------------------------------


def test_default_speed_fraction_applied(parser: PlanParser, scene: SceneSpec) -> None:
    """When speed_fraction is not specified, default 0.5 should be applied."""
    raw = {
        "plan_id": "defaults",
        "skills": [
            {
                "name": "move_to",
                "params": {
                    "target": {
                        "frame": "world",
                        "position": [0.1, 0.0, 0.5],
                    },
                    # no speed_fraction
                },
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, CanonicalPlan)
    assert result.skills[0].params["speed_fraction"] == 0.5


# ---------------------------------------------------------------------------
# 11. test_direction_vector_normalized
# ---------------------------------------------------------------------------


def test_direction_vector_normalized(parser: PlanParser, scene: SceneSpec) -> None:
    """direction [0,0,-2] should be normalized to [0,0,-1]."""
    raw = {
        "plan_id": "dir_norm",
        "skills": [
            {
                "name": "move_linear",
                "params": {
                    "direction": [0.0, 0.0, -2.0],
                    "distance": 0.1,
                },
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, CanonicalPlan)
    direction = result.skills[0].params["direction"]
    np.testing.assert_allclose(direction, [0.0, 0.0, -1.0], atol=1e-8)


# ---------------------------------------------------------------------------
# 12. test_zero_direction_rejected
# ---------------------------------------------------------------------------


def test_zero_direction_rejected(parser: PlanParser, scene: SceneSpec) -> None:
    """direction [0,0,0] should be rejected (zero vector)."""
    raw = {
        "plan_id": "zero_dir",
        "skills": [
            {
                "name": "move_linear",
                "params": {
                    "direction": [0.0, 0.0, 0.0],
                    "distance": 0.1,
                },
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, PlanRejection)
    assert any("direction vector is zero" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# 13. test_guard_condition_parsed
# ---------------------------------------------------------------------------


def test_guard_condition_parsed(parser: PlanParser, scene: SceneSpec) -> None:
    """Contact guard with valid body reference should be parsed."""
    raw = {
        "plan_id": "guard_ok",
        "skills": [
            {
                "name": "move_linear",
                "params": {
                    "direction": [0.0, 0.0, -1.0],
                    "distance": 0.2,
                    "guard": {
                        "type": "contact",
                        "body": "cube",
                        "min_force": 5.0,
                    },
                },
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, CanonicalPlan)
    guard = result.skills[0].guard
    assert guard is not None
    assert guard.type == "contact"
    assert guard.body == "cube"
    assert guard.min_force == 5.0


# ---------------------------------------------------------------------------
# 14. test_guard_unknown_body_rejected
# ---------------------------------------------------------------------------


def test_guard_unknown_body_rejected(parser: PlanParser, scene: SceneSpec) -> None:
    """Guard referencing non-existent body should be rejected."""
    raw = {
        "plan_id": "guard_bad",
        "skills": [
            {
                "name": "move_linear",
                "params": {
                    "direction": [0.0, 0.0, -1.0],
                    "distance": 0.2,
                    "guard": {
                        "type": "contact",
                        "body": "nonexistent_body",
                    },
                },
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, PlanRejection)
    assert "UNKNOWN_OBJECT" in result.error_codes
    assert any("nonexistent_body" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# 15. test_wait_negative_steps_rejected
# ---------------------------------------------------------------------------


def test_wait_negative_steps_rejected(parser: PlanParser, scene: SceneSpec) -> None:
    """wait with steps=-1 should be rejected."""
    raw = {
        "plan_id": "neg_wait",
        "skills": [
            {
                "name": "wait",
                "params": {"steps": -1},
            }
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, PlanRejection)
    assert any("steps must be positive integer" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# 16. test_multiple_errors_collected
# ---------------------------------------------------------------------------


def test_multiple_errors_collected(parser: PlanParser, scene: SceneSpec) -> None:
    """Plan with 3 bad skills should produce 3 errors in rejection."""
    raw = {
        "plan_id": "multi_err",
        "skills": [
            {"name": "fly_to", "params": {}},
            {"name": "teleport", "params": {}},
            {"name": "wait", "params": {"steps": -5}},
        ],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, PlanRejection)
    assert len(result.reasons) == 3


# ---------------------------------------------------------------------------
# 17. test_empty_skills_list
# ---------------------------------------------------------------------------


def test_empty_skills_list(parser: PlanParser, scene: SceneSpec) -> None:
    """Empty skills list should produce a valid CanonicalPlan (edge case)."""
    raw = {
        "plan_id": "empty",
        "skills": [],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, CanonicalPlan)
    assert len(result.skills) == 0


# ---------------------------------------------------------------------------
# 18. test_plan_metadata_preserved
# ---------------------------------------------------------------------------


def test_plan_metadata_preserved(parser: PlanParser, scene: SceneSpec) -> None:
    """plan_type, rationale, assumptions should be carried through."""
    raw = {
        "plan_id": "meta_plan",
        "plan_type": "reach",
        "rationale": "Simple approach movement",
        "assumptions": ["table is clear", "no obstacles"],
        "skills": [],
    }

    result = parser.parse(raw, scene)
    assert isinstance(result, CanonicalPlan)
    assert result.plan_id == "meta_plan"
    assert result.metadata["plan_type"] == "reach"
    assert result.metadata["rationale"] == "Simple approach movement"
    assert result.assumptions == ["table is clear", "no obstacles"]
