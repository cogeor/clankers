"""Physics-Validated Candidate Bank (PVCB) refiner.

Applies deterministic rewrite rules to failing plans, then falls back to
LLM re-proposal if deterministic fixes don't resolve the issue.
"""
from __future__ import annotations

import hashlib
import json
from typing import Union

from clankers_synthetic.parser import PlanParser
from clankers_synthetic.planner import LLMPlanner
from clankers_synthetic.specs import (
    CanonicalPlan,
    PlanRejection,
    ResolvedSkill,
    SceneSpec,
    TaskSpec,
    ValidationReport,
)


class PVCBRefiner:
    """Candidate-level refinement for failed plan executions.

    Strategy:
    1. Deterministic fixes first (speed reduction, clearance increase,
       timeout extension).
    2. Optional LLM re-proposal with structured failure context.
    3. Loop detection via plan hash to prevent oscillation.

    Args:
        planner: LLMPlanner for re-proposal fallback.
        parser: PlanParser for re-validating LLM-refined plans.
        max_iterations: Max refinement attempts per candidate.
    """

    def __init__(
        self,
        planner: LLMPlanner | None = None,
        parser: PlanParser | None = None,
        max_iterations: int = 3,
    ) -> None:
        self.planner = planner
        self.parser = parser or PlanParser()
        self.max_iterations = max_iterations

    def refine(
        self,
        plan: CanonicalPlan,
        report: ValidationReport,
        scene: SceneSpec,
        task: TaskSpec,
    ) -> CanonicalPlan | None:
        """Attempt to refine a failing plan.

        Tries deterministic fixes first, then falls back to LLM re-proposal.
        Uses plan hashing for loop detection.

        Returns refined CanonicalPlan or None if candidate should be rejected.
        """
        seen_hashes: set[str] = set()
        seen_hashes.add(self._plan_hash(plan))

        current_plan = plan
        current_report = report
        attempt_history: list[dict] = []

        for iteration in range(self.max_iterations):
            # Try deterministic fixes first
            fixed = self._deterministic_fix(current_plan, current_report)

            if fixed is not None:
                h = self._plan_hash(fixed)
                if h in seen_hashes:
                    # Loop detected -- deterministic fix produced a plan
                    # we already tried. Fall through to LLM fallback.
                    fixed = None
                else:
                    seen_hashes.add(h)
                    # Deterministic fix produces a valid CanonicalPlan
                    # directly, so no need to re-parse through the parser.
                    return fixed

            if fixed is None and self.planner is not None:
                # LLM re-proposal fallback
                attempt_history.append({
                    "iteration": iteration,
                    "plan_hash": self._plan_hash(current_plan),
                    "failure_reason": current_report.failure_reason,
                    "violations": [
                        v.dict() for v in current_report.constraint_violations
                    ],
                })

                try:
                    refined = self.planner.refine_candidate(
                        plan=current_plan.dict(),
                        failure_report=current_report.dict(),
                        scene=scene,
                        task=task,
                        attempt_history=attempt_history,
                    )
                    raw_plan = refined["plan"]
                    result = self.parser.parse(raw_plan, scene)
                    if isinstance(result, PlanRejection):
                        continue  # Parser rejected the LLM's refined plan
                    # result is a CanonicalPlan
                    h = self._plan_hash(result)
                    if h in seen_hashes:
                        continue  # Loop detected
                    seen_hashes.add(h)
                    return result  # type: ignore
                except Exception:
                    continue  # LLM call failed, try next iteration

            # If no fix and no planner, reject
            if self.planner is None:
                return None

        return None  # Max iterations exceeded

    def _deterministic_fix(
        self,
        plan: CanonicalPlan,
        report: ValidationReport,
    ) -> CanonicalPlan | None:
        """Apply deterministic rewrite rules based on violation types.

        Rules:
        - max_force violation: reduce speed_fraction by 30% on motion skills
        - soft_ee_speed violation: reduce speed_fraction by 20%
        - max_force + set_gripper: increase wait_settle_steps
        - workspace_bounds: cannot fix deterministically (return None)
        - joint_limit: cannot fix deterministically (return None)
        - task_failure: cannot fix deterministically (return None)
        """
        violation_types = {v.type for v in report.constraint_violations}

        # Can't fix these deterministically
        unfixable = {"workspace_bounds", "joint_limit", "task_failure"}
        if violation_types & unfixable:
            return None

        # Only proceed if there's a fixable violation
        fixable = {"max_force", "soft_ee_speed"}
        if not (violation_types & fixable):
            return None

        # Determine speed reduction factor
        reduction = 0.7 if "max_force" in violation_types else 0.8

        motion_skills = frozenset({
            "move_to", "move_linear", "move_relative", "move_joints",
        })

        new_skills = []
        for skill in plan.skills:
            new_params = dict(skill.params)

            if skill.name in motion_skills:
                sf = new_params.get("speed_fraction", 0.5)
                new_params["speed_fraction"] = max(sf * reduction, 0.05)

            # Increase settle time for gripper if force is an issue
            if skill.name == "set_gripper" and "max_force" in violation_types:
                wait = new_params.get("wait_settle_steps", 5)
                new_params["wait_settle_steps"] = min(wait + 5, 50)

            new_skills.append(ResolvedSkill(
                name=skill.name,
                target_world_position=skill.target_world_position,
                target_orientation=skill.target_orientation,
                params=new_params,
                guard=skill.guard,
            ))

        return CanonicalPlan(
            plan_id=plan.plan_id + "_refined",
            skills=new_skills,
            assumptions=plan.assumptions,
            metadata={**plan.metadata, "refined": True, "reduction": reduction},
        )

    def _plan_hash(self, plan: CanonicalPlan) -> str:
        """Compute a deterministic hash of the plan for loop detection.

        Hashes skill names, params, and target positions. Ignores metadata
        (which changes on each refinement) to detect functional duplicates.
        """
        skills_data = [
            {
                "name": s.name,
                "params": s.params,
                "target": s.target_world_position,
            }
            for s in plan.skills
        ]
        return hashlib.sha256(
            json.dumps(skills_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
