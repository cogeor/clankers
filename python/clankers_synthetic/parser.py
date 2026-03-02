"""Strict plan parser and canonicalization for LLM-proposed skill plans."""

from __future__ import annotations

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

            # Resolve skill-specific validation
            target_pos = None
            target_orient = None
            guard = None
            resolved_params: dict = {}

            if name == "move_to":
                target_pos, orient, errs = self._resolve_move_to(params, scene, object_names, i)
                errors.extend(errs)
                if errs:
                    error_codes.extend(["INVALID_TARGET"] * len(errs))
                    continue
                target_orient = orient
                resolved_params = {
                    k: v for k, v in params.items() if k not in ("target", "orientation")
                }

            elif name == "move_linear":
                direction = params.get("direction")
                distance = params.get("distance")
                if not isinstance(direction, list) or len(direction) != 3:
                    errors.append(f"Skill {i}: direction must be [x,y,z]")
                    error_codes.append("INVALID_PARAMS")
                    continue
                if not isinstance(distance, (int, float)) or distance <= 0:
                    errors.append(f"Skill {i}: distance must be positive number")
                    error_codes.append("INVALID_PARAMS")
                    continue
                # Normalize direction vector
                d = np.array(direction, dtype=float)
                norm = np.linalg.norm(d)
                if norm < 1e-8:
                    errors.append(f"Skill {i}: direction vector is zero")
                    error_codes.append("INVALID_PARAMS")
                    continue
                resolved_params = {
                    "direction": (d / norm).tolist(),
                    "distance": float(distance),
                }
                # Copy optional params
                for k in ("speed_fraction", "tolerance"):
                    if k in params:
                        resolved_params[k] = params[k]

            elif name == "move_relative":
                delta = params.get("delta")
                frame = params.get("frame")
                if not isinstance(delta, list) or len(delta) != 3:
                    errors.append(f"Skill {i}: delta must be [dx,dy,dz]")
                    error_codes.append("INVALID_PARAMS")
                    continue
                if frame not in ("world", "ee") and frame not in object_names:
                    errors.append(f"Skill {i}: unknown frame '{frame}'")
                    error_codes.append("UNKNOWN_FRAME")
                    continue
                resolved_params = {"frame": frame, "delta": delta}

            elif name == "set_gripper":
                width = params.get("width")
                if not isinstance(width, (int, float)):
                    errors.append(f"Skill {i}: width must be a number")
                    error_codes.append("INVALID_PARAMS")
                    continue
                if width < 0 or width > self.max_gripper_width:
                    errors.append(
                        f"Skill {i}: gripper width {width} out of "
                        f"range [0, {self.max_gripper_width}]"
                    )
                    error_codes.append("OUT_OF_RANGE")
                    continue
                resolved_params = {"width": float(width)}
                for k in ("force", "wait_settle_steps"):
                    if k in params:
                        resolved_params[k] = params[k]

            elif name == "wait":
                steps = params.get("steps")
                if not isinstance(steps, int) or steps <= 0:
                    errors.append(f"Skill {i}: steps must be positive integer")
                    error_codes.append("INVALID_PARAMS")
                    continue
                resolved_params = {"steps": steps}

            elif name == "move_joints":
                targets = params.get("targets")
                if not isinstance(targets, dict):
                    errors.append(f"Skill {i}: targets must be dict")
                    error_codes.append("INVALID_PARAMS")
                    continue
                resolved_params = {"targets": targets}

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
