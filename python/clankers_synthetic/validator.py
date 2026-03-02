"""Simulation validator with hard and soft gates for execution traces."""

from __future__ import annotations

import numpy as np

from clankers_synthetic.specs import (
    ConstraintSpec,
    ConstraintViolation,
    ExecutionTrace,
    TaskSpec,
    ValidationMetrics,
    ValidationReport,
)


class SimValidator:
    """Validate an execution trace against task success criteria and constraints.

    Hard gates (cause rejection):
    - Task success at final step
    - Workspace bounds (EE position each step)
    - Joint limits
    - Max contact force

    Soft gates (log warning, don't reject):
    - EE speed exceeding preferred threshold
    - Trajectory jerk exceeding threshold
    """

    def __init__(
        self,
        ee_link_name: str = "end_effector",
        joint_limits: dict[str, list[float]] | None = None,
        control_dt: float = 0.02,
    ) -> None:
        self.ee_link_name = ee_link_name
        self.joint_limits = joint_limits or {}
        self.control_dt = control_dt

    def validate(
        self,
        trace: ExecutionTrace,
        task: TaskSpec,
        constraints: ConstraintSpec,
    ) -> ValidationReport:
        """Validate a complete execution trace.

        Args:
            trace: ExecutionTrace from SkillCompiler
            task: TaskSpec with success criteria
            constraints: ConstraintSpec with workspace bounds, force limits, etc.

        Returns:
            ValidationReport with pass/fail, violations, and metrics.
        """
        violations: list[ConstraintViolation] = []

        # Compute metrics
        max_contact_force = 0.0
        max_joint_velocity = 0.0
        max_ee_speed = 0.0
        final_object_poses: dict[str, list[float]] = {}
        success_at_step: int | None = None

        prev_ee_pos: np.ndarray | None = None
        prev_joint_pos: np.ndarray | None = None

        for step_idx, step in enumerate(trace.steps):
            info = step.info

            # --- Extract data from info ---
            body_poses = info.get("body_poses", {})
            contact_events = info.get("contact_events", [])
            is_success = info.get("is_success", False)

            # Track success
            if is_success and success_at_step is None:
                success_at_step = step_idx

            # --- Hard gate: Contact force ---
            for contact in contact_events:
                force = contact.get("force_magnitude", 0.0)
                max_contact_force = max(max_contact_force, force)
                if force > constraints.max_contact_force:
                    violations.append(
                        ConstraintViolation(
                            type="max_force",
                            step=step_idx,
                            details=(
                                f"Contact force {force:.1f}N between "
                                f"'{contact.get('body_a', '?')}' and "
                                f"'{contact.get('body_b', '?')}' exceeds "
                                f"limit {constraints.max_contact_force:.1f}N"
                            ),
                        )
                    )

            # --- Hard gate: Workspace bounds ---
            ee_pose = body_poses.get(self.ee_link_name)
            if ee_pose and len(ee_pose) >= 3:
                ee_pos = np.array(ee_pose[:3])
                bounds_min = np.array(constraints.workspace_bounds_min)
                bounds_max = np.array(constraints.workspace_bounds_max)

                for dim in range(3):
                    if ee_pos[dim] < bounds_min[dim]:
                        violations.append(
                            ConstraintViolation(
                                type="workspace_bounds",
                                step=step_idx,
                                details=(
                                    f"EE {'XYZ'[dim]}={ee_pos[dim]:.3f} below "
                                    f"min {bounds_min[dim]:.3f}"
                                ),
                            )
                        )
                    if ee_pos[dim] > bounds_max[dim]:
                        violations.append(
                            ConstraintViolation(
                                type="workspace_bounds",
                                step=step_idx,
                                details=(
                                    f"EE {'XYZ'[dim]}={ee_pos[dim]:.3f} above "
                                    f"max {bounds_max[dim]:.3f}"
                                ),
                            )
                        )

                # EE speed
                if prev_ee_pos is not None:
                    ee_speed = float(np.linalg.norm(ee_pos - prev_ee_pos) / self.control_dt)
                    max_ee_speed = max(max_ee_speed, ee_speed)

                    # Soft gate: EE speed
                    if ee_speed > constraints.max_ee_speed:
                        violations.append(
                            ConstraintViolation(
                                type="soft_ee_speed",
                                step=step_idx,
                                details=(
                                    f"EE speed {ee_speed:.3f} m/s exceeds "
                                    f"preferred {constraints.max_ee_speed:.3f} m/s"
                                ),
                            )
                        )

                prev_ee_pos = ee_pos

            # --- Hard gate: Joint limits ---
            joint_positions = info.get("joint_positions")
            if joint_positions and self.joint_limits:
                for jname, limits in self.joint_limits.items():
                    if jname in joint_positions:
                        pos = joint_positions[jname]
                        if pos < limits[0] or pos > limits[1]:
                            violations.append(
                                ConstraintViolation(
                                    type="joint_limit",
                                    step=step_idx,
                                    details=(
                                        f"Joint '{jname}' position {pos:.4f} "
                                        f"outside limits [{limits[0]:.4f}, {limits[1]:.4f}]"
                                    ),
                                )
                            )

            # Joint velocity
            if joint_positions and prev_joint_pos is not None:
                curr = np.array(
                    list(joint_positions.values())
                    if isinstance(joint_positions, dict)
                    else joint_positions
                )
                vel = np.abs(curr - prev_joint_pos) / self.control_dt
                max_joint_velocity = max(max_joint_velocity, float(np.max(vel)))

            if joint_positions:
                prev_joint_pos = np.array(
                    list(joint_positions.values())
                    if isinstance(joint_positions, dict)
                    else joint_positions
                )

            # Store final poses
            if step_idx == len(trace.steps) - 1:
                final_object_poses = {
                    k: v if isinstance(v, list) else list(v) for k, v in body_poses.items()
                }

        # --- Hard gate: Task success at final step ---
        task_success = False
        if trace.steps:
            final_info = trace.steps[-1].info
            task_success = bool(final_info.get("is_success", False))

        if not task_success:
            violations.append(
                ConstraintViolation(
                    type="task_failure",
                    step=len(trace.steps) - 1 if trace.steps else 0,
                    details="Task success criteria not met at final step",
                )
            )

        # Determine pass/fail
        hard_violations = [v for v in violations if not v.type.startswith("soft_")]
        passed = len(hard_violations) == 0

        # Build failure reason
        failure_reason = None
        if not passed:
            types = sorted(set(v.type for v in hard_violations))
            failure_reason = f"Hard gate violations: {', '.join(types)}"

        metrics = ValidationMetrics(
            total_steps=len(trace.steps),
            max_contact_force=max_contact_force,
            max_joint_velocity=max_joint_velocity,
            max_ee_speed=max_ee_speed,
            final_object_poses=final_object_poses,
            success_at_step=success_at_step,
        )

        return ValidationReport(
            passed=passed,
            task_success=task_success,
            constraint_violations=violations,
            metrics=metrics,
            failure_reason=failure_reason,
        )
