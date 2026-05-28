"""Strict plan parser and canonicalization for LLM-proposed skill plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from clankers_synthetic.specs import (
    CanonicalPlan,
    GuardCondition,
    PlanRejection,
    ResolvedSkill,
    SceneSpec,
)

# Valid skill names (exhaustive vocabulary from spec section 5.2)
VALID_SKILLS = frozenset(
    {
        "move_to",
        "move_linear",
        "move_relative",
        "set_gripper",
        "wait",
        "move_joints",
    }
)

# Required params per skill
REQUIRED_PARAMS: dict[str, set[str]] = {
    "move_to": {"target"},
    "move_linear": {"direction", "distance"},
    "move_relative": {"frame", "delta"},
    "set_gripper": {"width"},
    "wait": {"steps"},
    "move_joints": {"targets"},
}


# ---------------------------------------------------------------------------
# SkillValidator registry
# ---------------------------------------------------------------------------
#
# CODE_QUALITY_REVIEW Python Synthetic Pipeline finding / P1.13. The
# pre-P1.13 `PlanParser.parse` held a single long if/elif chain
# branching on `skill.name` for skill-specific validation. Adding a new
# skill required editing the branch *and* the test matrix in the same
# place.
#
# Each validator returns ``SkillValidatorResult``:
#
# - errors: per-skill error strings (added to PlanRejection.reasons).
# - error_codes: per-skill error codes (added to PlanRejection.error_codes).
# - resolved_params: the canonicalised ``params`` dict for ResolvedSkill;
#   ``None`` indicates the skill itself failed validation (skip).
# - target_world_position / target_orientation: only ``move_to`` populates
#   these today; future skills can extend without changing the registry
#   shape.


@dataclass
class SkillValidatorResult:
    """Outcome of running a SkillValidator on one raw skill dict."""

    errors: list[str]
    error_codes: list[str]
    resolved_params: dict[str, Any] | None = None
    target_world_position: list[float] | None = None
    target_orientation: list[float] | None = None


class PlanParser:
    """Validate and canonicalize LLM-proposed skill plans.

    Checks:
    - JSON structure matches expected schema
    - All skill names are in the vocabulary
    - All object references exist in SceneSpec.objects
    - All target positions are within workspace bounds
    - Quaternions are unit-norm (normalize if close, reject if far)
    - speed_fraction in [0.0, 1.0]
    - gripper width in [0.0, max_gripper_width]
    - Required params per skill are present

    Outputs CanonicalPlan with resolved world-frame targets, or PlanRejection.
    """

    def __init__(self, max_gripper_width: float = 0.08) -> None:
        self.max_gripper_width = max_gripper_width
        # Per-skill validator registry; populated at construction so
        # adding a new skill is a single registry entry plus a
        # _validate_<skill> method, rather than a new branch in parse().
        self._validators: dict[
            str,
            Any,  # bound method (params, scene, object_names, idx) -> SkillValidatorResult
        ] = {
            "move_to": self._validate_move_to,
            "move_linear": self._validate_move_linear,
            "move_relative": self._validate_move_relative,
            "set_gripper": self._validate_set_gripper,
            "wait": self._validate_wait,
            "move_joints": self._validate_move_joints,
        }

    def parse(self, raw: dict, scene: SceneSpec) -> CanonicalPlan | PlanRejection:
        """Parse and validate a raw LLM plan dict against the scene.

        Args:
            raw: Raw JSON dict from LLM (should match LLMProposedPlan structure)
            scene: SceneSpec for object reference resolution and bounds checking

        Returns:
            CanonicalPlan if valid, PlanRejection if not.
        """
        errors: list[str] = []
        error_codes: list[str] = []

        # 1. Validate top-level structure
        plan_id = raw.get("plan_id", "unknown")
        skills_raw = raw.get("skills", [])
        if not isinstance(skills_raw, list):
            return PlanRejection(
                reasons=["'skills' field must be a list"],
                raw_plan=raw,
                error_codes=["INVALID_STRUCTURE"],
            )

        # Build object name lookup
        object_names = {obj.name for obj in scene.objects}
        # Add robot-related body names
        object_names.add(scene.robot.ee_link_name)
        object_names.add(scene.robot.name)

        # 2. Validate each skill
        resolved_skills: list[ResolvedSkill] = []
        for i, skill_raw in enumerate(skills_raw):
            if not isinstance(skill_raw, dict):
                errors.append(f"Skill {i}: not a dict")
                error_codes.append("INVALID_SKILL_STRUCTURE")
                continue

            name = skill_raw.get("name", "")
            params = skill_raw.get("params", {})

            # Check skill name
            if name not in VALID_SKILLS:
                errors.append(f"Skill {i}: unknown skill '{name}'")
                error_codes.append("UNKNOWN_SKILL")
                continue

            # Check required params
            missing = REQUIRED_PARAMS.get(name, set()) - set(params.keys())
            if missing:
                errors.append(f"Skill {i} ({name}): missing required params: {missing}")
                error_codes.append("MISSING_PARAMS")
                continue

            # Per-skill validation via the registry (P1.13).
            target_pos = None
            target_orient = None
            guard = None
            result = self._validators[name](params, scene, object_names, i)
            if result.errors:
                errors.extend(result.errors)
                error_codes.extend(result.error_codes)
                continue
            assert result.resolved_params is not None, (
                f"validator for {name!r} returned no errors and no resolved_params"
            )
            target_pos = result.target_world_position
            target_orient = result.target_orientation
            resolved_params: dict = result.resolved_params

            # Validate speed_fraction if present
            sf = resolved_params.get("speed_fraction", params.get("speed_fraction"))
            if sf is not None:
                if not isinstance(sf, (int, float)) or sf < 0.0 or sf > 1.0:
                    errors.append(f"Skill {i}: speed_fraction must be in [0.0, 1.0], got {sf}")
                    error_codes.append("OUT_OF_RANGE")
                    continue
                resolved_params["speed_fraction"] = float(sf)
            elif name in (
                "move_to",
                "move_linear",
                "move_relative",
                "move_joints",
            ):
                # Apply default speed_fraction
                resolved_params["speed_fraction"] = 0.5

            # Parse guard condition
            guard_raw = params.get("guard")
            if guard_raw and isinstance(guard_raw, dict):
                prev_error_count = len(errors)
                guard = self._parse_guard(guard_raw, object_names, i, errors, error_codes)
                if len(errors) > prev_error_count:
                    continue

            resolved_skills.append(
                ResolvedSkill(
                    name=name,
                    target_world_position=target_pos,
                    target_orientation=target_orient,
                    params=resolved_params,
                    guard=guard,
                )
            )

        if errors:
            return PlanRejection(
                reasons=errors,
                raw_plan=raw,
                error_codes=list(set(error_codes)),
            )

        return CanonicalPlan(
            plan_id=plan_id,
            skills=resolved_skills,
            assumptions=raw.get("assumptions", []),
            metadata={
                "plan_type": raw.get("plan_type", ""),
                "rationale": raw.get("rationale", ""),
            },
        )

    # ------------------------------------------------------------------
    # SkillValidator methods (P1.13)
    # ------------------------------------------------------------------

    def _validate_move_to(
        self,
        params: dict,
        scene: SceneSpec,
        object_names: set[str],
        skill_idx: int,
    ) -> SkillValidatorResult:
        target_pos, orient, errs = self._resolve_move_to(params, scene, object_names, skill_idx)
        if errs:
            return SkillValidatorResult(
                errors=errs,
                error_codes=["INVALID_TARGET"] * len(errs),
            )
        resolved = {k: v for k, v in params.items() if k not in ("target", "orientation")}
        return SkillValidatorResult(
            errors=[],
            error_codes=[],
            resolved_params=resolved,
            target_world_position=target_pos,
            target_orientation=orient,
        )

    def _validate_move_linear(
        self,
        params: dict,
        scene: SceneSpec,
        object_names: set[str],
        skill_idx: int,
    ) -> SkillValidatorResult:
        del scene, object_names
        direction = params.get("direction")
        distance = params.get("distance")
        if not isinstance(direction, list) or len(direction) != 3:
            return SkillValidatorResult(
                errors=[f"Skill {skill_idx}: direction must be [x,y,z]"],
                error_codes=["INVALID_PARAMS"],
            )
        if not isinstance(distance, (int, float)) or distance <= 0:
            return SkillValidatorResult(
                errors=[f"Skill {skill_idx}: distance must be positive number"],
                error_codes=["INVALID_PARAMS"],
            )
        d = np.array(direction, dtype=float)
        norm = np.linalg.norm(d)
        if norm < 1e-8:
            return SkillValidatorResult(
                errors=[f"Skill {skill_idx}: direction vector is zero"],
                error_codes=["INVALID_PARAMS"],
            )
        resolved: dict[str, Any] = {
            "direction": (d / norm).tolist(),
            "distance": float(distance),
        }
        for k in ("speed_fraction", "tolerance"):
            if k in params:
                resolved[k] = params[k]
        return SkillValidatorResult(errors=[], error_codes=[], resolved_params=resolved)

    def _validate_move_relative(
        self,
        params: dict,
        scene: SceneSpec,
        object_names: set[str],
        skill_idx: int,
    ) -> SkillValidatorResult:
        del scene
        delta = params.get("delta")
        frame = params.get("frame")
        if not isinstance(delta, list) or len(delta) != 3:
            return SkillValidatorResult(
                errors=[f"Skill {skill_idx}: delta must be [dx,dy,dz]"],
                error_codes=["INVALID_PARAMS"],
            )
        if frame not in ("world", "ee") and frame not in object_names:
            return SkillValidatorResult(
                errors=[f"Skill {skill_idx}: unknown frame '{frame}'"],
                error_codes=["UNKNOWN_FRAME"],
            )
        return SkillValidatorResult(
            errors=[],
            error_codes=[],
            resolved_params={"frame": frame, "delta": delta},
        )

    def _validate_set_gripper(
        self,
        params: dict,
        scene: SceneSpec,
        object_names: set[str],
        skill_idx: int,
    ) -> SkillValidatorResult:
        del scene, object_names
        width = params.get("width")
        if not isinstance(width, (int, float)):
            return SkillValidatorResult(
                errors=[f"Skill {skill_idx}: width must be a number"],
                error_codes=["INVALID_PARAMS"],
            )
        if width < 0 or width > self.max_gripper_width:
            return SkillValidatorResult(
                errors=[
                    f"Skill {skill_idx}: gripper width {width} out of "
                    f"range [0, {self.max_gripper_width}]"
                ],
                error_codes=["OUT_OF_RANGE"],
            )
        resolved: dict[str, Any] = {"width": float(width)}
        for k in ("force", "wait_settle_steps"):
            if k in params:
                resolved[k] = params[k]
        return SkillValidatorResult(errors=[], error_codes=[], resolved_params=resolved)

    def _validate_wait(
        self,
        params: dict,
        scene: SceneSpec,
        object_names: set[str],
        skill_idx: int,
    ) -> SkillValidatorResult:
        del scene, object_names
        steps = params.get("steps")
        if not isinstance(steps, int) or steps <= 0:
            return SkillValidatorResult(
                errors=[f"Skill {skill_idx}: steps must be positive integer"],
                error_codes=["INVALID_PARAMS"],
            )
        return SkillValidatorResult(errors=[], error_codes=[], resolved_params={"steps": steps})

    def _validate_move_joints(
        self,
        params: dict,
        scene: SceneSpec,
        object_names: set[str],
        skill_idx: int,
    ) -> SkillValidatorResult:
        del scene, object_names
        targets = params.get("targets")
        if not isinstance(targets, dict):
            return SkillValidatorResult(
                errors=[f"Skill {skill_idx}: targets must be dict"],
                error_codes=["INVALID_PARAMS"],
            )
        return SkillValidatorResult(errors=[], error_codes=[], resolved_params={"targets": targets})

    def _resolve_move_to(
        self,
        params: dict,
        scene: SceneSpec,
        object_names: set[str],
        skill_idx: int,
    ) -> tuple:
        """Resolve move_to target to world frame position.

        Returns:
            (world_position, orientation, errors) tuple.
        """
        errors: list[str] = []
        target = params.get("target", {})

        if not isinstance(target, dict):
            return None, None, [f"Skill {skill_idx}: target must be a dict"]

        frame = target.get("frame", "world")
        position = target.get("position")

        if not isinstance(position, list) or len(position) != 3:
            return (
                None,
                None,
                [f"Skill {skill_idx}: target.position must be [x,y,z]"],
            )

        world_pos = [float(p) for p in position]

        # Resolve frame references
        if frame == "world":
            pass  # already world frame
        elif frame in object_names:
            # For object frame targets, we note it but don't resolve at
            # parse time (resolved at execution time from body_poses)
            pass
        elif frame == "ee":
            pass  # resolved at execution time
        else:
            errors.append(f"Skill {skill_idx}: unknown frame '{frame}'")
            return None, None, errors

        # Bounds check (only for world frame)
        if frame == "world":
            bounds_min = scene.constraints.workspace_bounds_min
            bounds_max = scene.constraints.workspace_bounds_max
            for dim, (val, lo, hi) in enumerate(
                zip(world_pos, bounds_min, bounds_max, strict=False)
            ):
                if val < lo or val > hi:
                    axis = "XYZ"[dim]
                    errors.append(
                        f"Skill {skill_idx}: target {axis}={val} outside workspace [{lo}, {hi}]"
                    )

        # Parse orientation if present
        orient = None
        orient_raw = params.get("orientation")
        if orient_raw and isinstance(orient_raw, dict):
            quat = orient_raw.get("quaternion")
            if isinstance(quat, list) and len(quat) == 4:
                # Normalize quaternion
                q = np.array(quat, dtype=float)
                norm = np.linalg.norm(q)
                if norm < 0.9 or norm > 1.1:
                    errors.append(f"Skill {skill_idx}: quaternion norm {norm:.3f} too far from 1.0")
                else:
                    orient = (q / norm).tolist()

        return world_pos if not errors else None, orient, errors

    def _parse_guard(
        self,
        guard_raw: dict,
        object_names: set[str],
        skill_idx: int,
        errors: list[str],
        error_codes: list[str],
    ) -> GuardCondition | None:
        """Parse a guard condition dict."""
        guard_type = guard_raw.get("type")
        if guard_type not in ("contact", "distance", "timeout"):
            errors.append(f"Skill {skill_idx}: unknown guard type '{guard_type}'")
            error_codes.append("INVALID_GUARD")
            return None

        if guard_type == "contact":
            body = guard_raw.get("body")
            if body and body not in object_names:
                errors.append(f"Skill {skill_idx}: guard body '{body}' not in scene")
                error_codes.append("UNKNOWN_OBJECT")
                return None
            return GuardCondition(
                type="contact",
                body=body,
                min_force=guard_raw.get("min_force"),
            )
        elif guard_type == "distance":
            from_body = guard_raw.get("from")
            if from_body and from_body not in object_names:
                errors.append(f"Skill {skill_idx}: guard from '{from_body}' not in scene")
                error_codes.append("UNKNOWN_OBJECT")
                return None
            return GuardCondition(
                type="distance",
                from_body=from_body,
                threshold=guard_raw.get("threshold"),
            )
        else:  # timeout
            return GuardCondition(
                type="timeout",
                steps=guard_raw.get("steps"),
            )
