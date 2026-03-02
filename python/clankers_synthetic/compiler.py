"""Skill compiler: execute CanonicalPlan skill-by-skill through a gymnasium env."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from clankers_synthetic.ik_solver import DlsSolver
from clankers_synthetic.specs import (
    CanonicalPlan,
    ExecutionTrace,
    GuardCondition,
    ResolvedSkill,
    TraceStep,
)


class StepEnv(Protocol):
    """Minimal env interface needed by the compiler."""

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]: ...
    def reset(self) -> tuple[Any, dict]: ...


class SkillCompiler:
    """Execute a CanonicalPlan through a live environment step-by-step.

    For each skill:
    1. Compute target joint positions (IK for Cartesian skills, direct for joint-space)
    2. Interpolate from current to target over N steps
    3. env.step(action) for each interpolated waypoint
    4. Check guard conditions via info dict
    5. Record (obs, action, next_obs, info) into trace

    Args:
        ik_solver: DlsSolver for Cartesian-to-joint conversion.
        joint_names: Ordered list of joint names.
        joint_limits: Dict of joint_name -> [lower, upper].
        control_dt: Control timestep in seconds (default 0.02).
        max_joint_velocity: Max joint velocity in rad/s (default 2.0).
    """

    def __init__(
        self,
        ik_solver: DlsSolver | None = None,
        joint_names: list[str] | None = None,
        joint_limits: dict[str, list[float]] | None = None,
        control_dt: float = 0.02,
        max_joint_velocity: float = 2.0,
    ) -> None:
        self.ik_solver = ik_solver
        self.joint_names = joint_names or []
        self.joint_limits = joint_limits or {}
        self.control_dt = control_dt
        self.max_joint_velocity = max_joint_velocity
        self._n_joints = len(self.joint_names)

        # Precompute joint centers and half-ranges for action normalization
        self._joint_centers = np.zeros(self._n_joints)
        self._joint_half_ranges = np.ones(self._n_joints)
        for i, name in enumerate(self.joint_names):
            if name in self.joint_limits:
                lo, hi = self.joint_limits[name]
                self._joint_centers[i] = (lo + hi) / 2.0
                self._joint_half_ranges[i] = max((hi - lo) / 2.0, 1e-6)

    def normalize_action(self, joint_targets: np.ndarray) -> np.ndarray:
        """Convert absolute joint positions to normalized [-1, 1] actions."""
        return (joint_targets - self._joint_centers) / self._joint_half_ranges

    def execute(self, plan: CanonicalPlan, env: StepEnv) -> ExecutionTrace:
        """Execute a plan through the environment.

        Args:
            plan: Validated CanonicalPlan with resolved skills.
            env: Environment implementing step() protocol.

        Returns:
            ExecutionTrace with all recorded transitions.
        """
        steps: list[TraceStep] = []
        total_reward = 0.0
        terminated = False
        truncated = False

        # Reset env
        obs, info = env.reset()
        current_obs = np.array(obs, dtype=np.float32) if not isinstance(obs, np.ndarray) else obs

        # Current joint positions (from obs or zeros)
        current_joints = self._extract_joint_positions(current_obs, info)

        for skill in plan.skills:
            if terminated or truncated:
                break

            skill_steps, current_obs, current_joints, terminated, truncated, info = (
                self._execute_skill(skill, env, current_obs, current_joints, steps)
            )
            total_reward += sum(s.reward for s in skill_steps)

        final_info = info if steps else {}

        return ExecutionTrace(
            plan_id=plan.plan_id,
            steps=steps,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            final_info=final_info if isinstance(final_info, dict) else {},
        )

    def _execute_skill(
        self,
        skill: ResolvedSkill,
        env: StepEnv,
        current_obs: np.ndarray,
        current_joints: np.ndarray,
        all_steps: list[TraceStep],
    ) -> tuple[list[TraceStep], np.ndarray, np.ndarray, bool, bool, dict]:
        """Execute a single skill.

        Returns:
            Tuple of (new_steps, obs, joints, terminated, truncated, info).
        """
        if skill.name == "move_to":
            return self._exec_move_to(skill, env, current_obs, current_joints, all_steps)
        elif skill.name == "move_linear":
            return self._exec_move_linear(skill, env, current_obs, current_joints, all_steps)
        elif skill.name == "set_gripper":
            return self._exec_set_gripper(skill, env, current_obs, current_joints, all_steps)
        elif skill.name == "move_relative":
            return self._exec_move_relative(skill, env, current_obs, current_joints, all_steps)
        elif skill.name == "move_joints":
            return self._exec_move_joints(skill, env, current_obs, current_joints, all_steps)
        elif skill.name == "wait":
            return self._exec_wait(skill, env, current_obs, current_joints, all_steps)
        else:
            # Unsupported skill -- treat as wait(1)
            return self._exec_wait(
                ResolvedSkill(name="wait", params={"steps": 1}),
                env,
                current_obs,
                current_joints,
                all_steps,
            )

    def _exec_move_to(self, skill, env, obs, joints, all_steps):
        """Execute move_to: IK solve target, interpolate, step."""
        target_pos = skill.target_world_position
        if target_pos is None:
            # Fallback: no target, just hold position
            return self._exec_wait(
                ResolvedSkill(name="wait", params={"steps": 1}),
                env,
                obs,
                joints,
                all_steps,
            )

        target = np.array(target_pos)
        speed_frac = skill.params.get("speed_fraction", 0.5)

        # IK solve
        if self.ik_solver is not None:
            ik_result = self.ik_solver.solve(target, joints)
            target_joints = ik_result.joint_angles
        else:
            # No IK solver -- just hold position
            target_joints = joints.copy()

        # Interpolate
        return self._interpolate_to_target(
            target_joints,
            speed_frac,
            skill.guard,
            env,
            obs,
            joints,
            all_steps,
        )

    def _exec_move_linear(self, skill, env, obs, joints, all_steps):
        """Execute move_linear: step-wise Cartesian straight line."""
        direction = np.array(skill.params.get("direction", [0, 0, 0]))
        distance = skill.params.get("distance", 0.0)
        speed_frac = skill.params.get("speed_fraction", 0.5)

        # Compute number of steps based on speed
        max_vel = speed_frac * self.max_joint_velocity
        n_steps = max(1, int(distance / max(max_vel * self.control_dt, 1e-6)))
        n_steps = min(n_steps, 500)  # cap

        new_steps: list[TraceStep] = []
        terminated = False
        truncated = False
        info: dict = {}
        current_joints = joints.copy()
        current_obs = obs

        for _step_i in range(n_steps):
            if terminated or truncated:
                break

            # For simplicity in MVP, compute intermediate IK waypoints
            # along the Cartesian direction. If IK is available, solve for
            # the incremental waypoint; otherwise just hold current joints.
            if self.ik_solver is not None:
                # Use body_poses from info to get current EE position
                body_poses = info.get("body_poses", {}) if info else {}
                ee_pose = body_poses.get("end_effector", None)
                current_ee = np.array(ee_pose[:3]) if ee_pose and len(ee_pose) >= 3 else np.zeros(3)
                waypoint = current_ee + direction * distance * (1.0 / n_steps)
                ik_result = self.ik_solver.solve(waypoint, current_joints)
                target_joints = ik_result.joint_angles
            else:
                target_joints = current_joints

            action = np.asarray(target_joints, dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs_arr = (
                np.array(next_obs, dtype=np.float32)
                if not isinstance(next_obs, np.ndarray)
                else next_obs
            )

            trace_step = TraceStep(
                obs=current_obs.tolist() if hasattr(current_obs, "tolist") else list(current_obs),
                action=action.tolist() if hasattr(action, "tolist") else list(action),
                next_obs=next_obs_arr.tolist()
                if hasattr(next_obs_arr, "tolist")
                else list(next_obs_arr),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            new_steps.append(trace_step)
            all_steps.append(trace_step)
            current_obs = next_obs_arr
            current_joints = self._extract_joint_positions(current_obs, info)

            # Check guard
            if skill.guard and self._guard_triggered(skill.guard, info):
                break

        return new_steps, current_obs, current_joints, terminated, truncated, info

    def _exec_move_relative(self, skill, env, obs, joints, all_steps):
        """Execute move_relative: move EE by delta in world frame."""
        delta = np.array(skill.params.get("delta", [0, 0, 0]), dtype=float)
        speed_frac = skill.params.get("speed_fraction", 0.5)

        # Apply delta to current EE position via IK.
        # frame param is accepted by the parser but we treat delta as
        # world-frame offset (frame resolution requires FK).
        if self.ik_solver is not None:
            target = np.zeros(3) + delta
            ik_result = self.ik_solver.solve(target, joints)
            target_joints = ik_result.joint_angles
        else:
            target_joints = joints.copy()

        return self._interpolate_to_target(
            target_joints, speed_frac, skill.guard, env, obs, joints, all_steps
        )

    def _exec_move_joints(self, skill, env, obs, joints, all_steps):
        """Execute move_joints: direct joint-space interpolation to targets."""
        targets = skill.params.get("targets", {})
        speed_frac = skill.params.get("speed_fraction", 0.5)

        # Build target joint array from current + overrides
        target_joints = joints.copy()
        for name, value in targets.items():
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                target_joints[idx] = value

        return self._interpolate_to_target(
            target_joints, speed_frac, skill.guard, env, obs, joints, all_steps
        )

    def _exec_set_gripper(self, skill, env, obs, joints, all_steps):
        """Execute set_gripper: set gripper width and wait for settle."""
        wait_steps = skill.params.get("wait_settle_steps", 5)

        # Set gripper width on the bridge env if supported
        width = skill.params.get("width", 0.0)
        if hasattr(env, "gripper_width"):
            env.gripper_width = width

        new_steps: list[TraceStep] = []
        terminated = False
        truncated = False
        info: dict = {}
        current_obs = obs
        current_joints = joints.copy()

        for _ in range(max(1, wait_steps)):
            if terminated or truncated:
                break
            action = np.asarray(current_joints, dtype=np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs_arr = (
                np.array(next_obs, dtype=np.float32)
                if not isinstance(next_obs, np.ndarray)
                else next_obs
            )

            trace_step = TraceStep(
                obs=current_obs.tolist() if hasattr(current_obs, "tolist") else list(current_obs),
                action=action.tolist() if hasattr(action, "tolist") else list(action),
                next_obs=next_obs_arr.tolist()
                if hasattr(next_obs_arr, "tolist")
                else list(next_obs_arr),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            new_steps.append(trace_step)
            all_steps.append(trace_step)
            current_obs = next_obs_arr

        # Extract final joint positions for handoff to next skill
        current_joints = self._extract_joint_positions(current_obs, info)
        return new_steps, current_obs, current_joints, terminated, truncated, info

    def _exec_wait(self, skill, env, obs, joints, all_steps):
        """Execute wait: hold current positions for N steps."""
        n_steps = skill.params.get("steps", 1)

        new_steps: list[TraceStep] = []
        terminated = False
        truncated = False
        info: dict = {}
        current_obs = obs
        current_joints = joints.copy()

        for _ in range(n_steps):
            if terminated or truncated:
                break
            action = np.asarray(current_joints, dtype=np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs_arr = (
                np.array(next_obs, dtype=np.float32)
                if not isinstance(next_obs, np.ndarray)
                else next_obs
            )

            trace_step = TraceStep(
                obs=current_obs.tolist() if hasattr(current_obs, "tolist") else list(current_obs),
                action=action.tolist() if hasattr(action, "tolist") else list(action),
                next_obs=next_obs_arr.tolist()
                if hasattr(next_obs_arr, "tolist")
                else list(next_obs_arr),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            new_steps.append(trace_step)
            all_steps.append(trace_step)
            current_obs = next_obs_arr

        return new_steps, current_obs, current_joints, terminated, truncated, info

    def _interpolate_to_target(
        self,
        target_joints: np.ndarray,
        speed_fraction: float,
        guard: GuardCondition | None,
        env: StepEnv,
        obs: np.ndarray,
        current_joints: np.ndarray,
        all_steps: list[TraceStep],
    ):
        """Interpolate from current joints to target and step env."""
        delta = target_joints - current_joints
        max_delta = np.max(np.abs(delta))
        max_vel = max(speed_fraction * self.max_joint_velocity, 1e-6)
        n_steps = max(1, int(max_delta / (max_vel * self.control_dt)))
        n_steps = min(n_steps, 500)  # cap

        new_steps: list[TraceStep] = []
        terminated = False
        truncated = False
        info: dict = {}
        current_obs = obs

        for step_i in range(n_steps):
            if terminated or truncated:
                break
            frac = (step_i + 1) / n_steps
            waypoint = current_joints + delta * frac
            action = np.asarray(waypoint, dtype=np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs_arr = (
                np.array(next_obs, dtype=np.float32)
                if not isinstance(next_obs, np.ndarray)
                else next_obs
            )

            trace_step = TraceStep(
                obs=current_obs.tolist() if hasattr(current_obs, "tolist") else list(current_obs),
                action=action.tolist() if hasattr(action, "tolist") else list(action),
                next_obs=next_obs_arr.tolist()
                if hasattr(next_obs_arr, "tolist")
                else list(next_obs_arr),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            new_steps.append(trace_step)
            all_steps.append(trace_step)
            current_obs = next_obs_arr

            # Check guard
            if guard and self._guard_triggered(guard, info):
                break

        final_joints = self._extract_joint_positions(current_obs, info)
        return new_steps, current_obs, final_joints, terminated, truncated, info

    def _guard_triggered(self, guard: GuardCondition, info: dict) -> bool:
        """Check if a guard condition is satisfied."""
        if guard.type == "contact":
            contacts = info.get("contact_events", [])
            for contact in contacts:
                if (
                    guard.body
                    and contact.get("body_a") != guard.body
                    and contact.get("body_b") != guard.body
                ):
                    continue
                force = contact.get("force_magnitude", 0.0)
                if guard.min_force is not None and force >= guard.min_force:
                    return True
        elif guard.type == "distance":
            body_poses = info.get("body_poses", {})
            ee_pose = body_poses.get("end_effector", [])
            target_pose = body_poses.get(guard.from_body, []) if guard.from_body else []
            if len(ee_pose) >= 3 and len(target_pose) >= 3:
                dist = np.linalg.norm(np.array(ee_pose[:3]) - np.array(target_pose[:3]))
                if guard.threshold is not None and dist <= guard.threshold:
                    return True
        elif guard.type == "timeout":
            pass  # Handled by step counting in caller
        return False

    def _extract_joint_positions(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Extract current joint positions from observation or info."""
        # Try info first
        if isinstance(info, dict):
            jp = info.get("joint_positions")
            if jp is not None:
                if isinstance(jp, dict):
                    return np.array([jp.get(name, 0.0) for name in self.joint_names])
                return np.array(jp, dtype=float)
        # JointStateSensor produces interleaved [pos0, vel0, pos1, vel1, ...]
        # so obs length = 2 * n_joints. If obs is shorter (e.g. mock env),
        # fall back to taking the first n_joints elements directly.
        if len(obs) >= 2 * self._n_joints:
            return np.array(obs[0 : 2 * self._n_joints : 2], dtype=float)
        n = min(self._n_joints, len(obs))
        return np.array(obs[:n], dtype=float)
