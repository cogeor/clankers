"""Prompt assembler and LLM planner for robotic manipulation planning.

PromptAssembler converts SceneSpec + TaskSpec into structured LLM prompts.
LLMPlanner wraps OpenAIClient to propose plans and refine failed candidates.
"""

from __future__ import annotations

import json

from clankers_synthetic.openai_client import OpenAIClient
from clankers_synthetic.specs import LLMRequest, SceneSpec, TaskSpec


class PromptAssembler:
    """Convert SceneSpec + TaskSpec into structured LLM prompts.

    Args:
        few_shot_examples: Optional list of (SceneSpec, TaskSpec, plan_dict) exemplars.
        max_context_tokens: Approximate token budget for scene description.
    """

    TEMPLATE_VERSION = "1.0.0"

    SKILL_VOCABULARY_TABLE = """
| Skill | Params | Description |
|-------|--------|-------------|
| move_to | target: {frame, position}, orientation?, speed_fraction, tolerance? | IK solve target, interpolate joints |
| move_linear | direction: [x,y,z], distance, speed_fraction, guard? | Cartesian straight-line with step-wise IK |
| move_relative | frame, delta: [dx,dy,dz], speed_fraction | Move EE by delta relative to frame |
| set_gripper | width, force?, wait_settle_steps? | Set gripper joint target, wait for settle |
| wait | steps | Hold current positions for N steps |
| move_joints | targets: {name: rad}, speed_fraction | Direct joint-space interpolation |
"""

    def __init__(
        self,
        few_shot_examples: list | None = None,
        max_context_tokens: int = 8000,
    ) -> None:
        self.few_shot_examples = few_shot_examples or []
        self.max_context_tokens = max_context_tokens

    def assemble(
        self,
        scene: SceneSpec,
        task: TaskSpec,
        model: str = "gpt-5",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMRequest:
        """Build a complete LLM request.

        System message contains:
        - Role instruction (robotic manipulation planner)
        - Skill vocabulary table
        - Output JSON schema requirements
        - Rules (units, frames, constraints)

        User message contains:
        - Scene spec (JSON)
        - Task spec (JSON)
        - Few-shot examples (if provided)
        - "Plan the trajectory."
        """
        system_msg = self._build_system_message()
        user_msg = self._build_user_message(scene, task)
        return LLMRequest(
            system_message=system_msg,
            user_message=user_msg,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_system_message(self) -> str:
        return (
            "You are a robotic manipulation planner. Given a scene description "
            "and task, output a JSON skill plan that a 6-DOF robot arm can execute.\n"
            "\n"
            "## Available Skills\n"
            f"{self.SKILL_VOCABULARY_TABLE}\n"
            "\n"
            "## Output Schema\n"
            "{\n"
            '  "plan_id": "<unique_id>",\n'
            '  "plan_type": "skill_plan",\n'
            '  "rationale": "<brief strategy description>",\n'
            '  "assumptions": ["<list of preconditions>"],\n'
            '  "uncertainty_flags": ["<list of uncertainties>"],\n'
            '  "skills": [\n'
            "    {\n"
            '      "name": "<skill_name>",\n'
            '      "params": { ... },\n'
            '      "comment": "<optional explanation>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "\n"
            "## Rules\n"
            "- All positions are in meters, Z-up coordinate frame.\n"
            "- All orientations are quaternions [qx, qy, qz, qw].\n"
            "- Gripper width in meters (0.0 = closed, 0.08 = fully open).\n"
            "- speed_fraction is [0.0, 1.0] relative to joint velocity limits.\n"
            "- You MUST reference only objects listed in the scene.\n"
            "- All target positions MUST be within the workspace bounds.\n"
            "- Output ONLY valid JSON. No explanation outside the JSON.\n"
            "- Keep plans concise: prefer 4-10 skills per plan.\n"
            "\n"
            f"Prompt template version: {self.TEMPLATE_VERSION}"
        )

    def _build_user_message(self, scene: SceneSpec, task: TaskSpec) -> str:
        scene_json = json.dumps(scene.model_dump(), indent=2, default=str)
        task_json = json.dumps(task.model_dump(), indent=2, default=str)

        parts = [f"## Scene\n{scene_json}", f"## Task\n{task_json}"]

        if self.few_shot_examples:
            for i, (ex_scene, ex_task, ex_plan) in enumerate(self.few_shot_examples):
                parts.append(f"## Example {i + 1}")
                parts.append(f"Scene: {json.dumps(ex_scene.model_dump(), default=str)}")
                parts.append(f"Task: {json.dumps(ex_task.model_dump(), default=str)}")
                parts.append(f"Plan: {json.dumps(ex_plan, indent=2)}")

        parts.append("## Plan the trajectory.")
        return "\n\n".join(parts)


class LLMPlanner:
    """LLM-based plan proposer using OpenAI API.

    Args:
        assembler: PromptAssembler for building prompts.
        client: OpenAIClient for API calls.
        model: Default model name.
        temperature: Default sampling temperature.
        n_candidates: Default number of candidates per proposal.
        max_tokens: Max output tokens.
    """

    def __init__(
        self,
        assembler: PromptAssembler | None = None,
        client: OpenAIClient | None = None,
        model: str = "gpt-5",
        temperature: float = 0.3,
        n_candidates: int = 3,
        max_tokens: int = 4096,
    ) -> None:
        self.assembler = assembler or PromptAssembler()
        self.client = client or OpenAIClient()
        self.model = model
        self.temperature = temperature
        self.n_candidates = n_candidates
        self.max_tokens = max_tokens

    def propose(
        self,
        scene: SceneSpec,
        task: TaskSpec,
        n_candidates: int | None = None,
        seed: int | None = None,
    ) -> list:
        """Generate candidate plans from the LLM.

        Returns list of dicts, each with:
        - "plan": raw plan dict (LLMProposedPlan structure)
        - "provenance": metadata dict from OpenAIClient
        """
        n = n_candidates or self.n_candidates
        request = self.assembler.assemble(
            scene,
            task,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        results = self.client.request_json(request, n=n, seed=seed)
        return [
            {"plan": r["content"], "provenance": r["provenance"]}
            for r in results
        ]

    def refine_candidate(
        self,
        plan: dict,
        failure_report: dict,
        scene: SceneSpec,
        task: TaskSpec,
        attempt_history: list | None = None,
    ) -> dict:
        """Refine a failed plan using LLM with structured failure context.

        Returns dict with "plan" and "provenance" keys.
        """
        history_text = ""
        if attempt_history:
            history_text = "\n## Attempt History\n" + json.dumps(
                attempt_history, indent=2
            )

        refinement_context = (
            "\n## Previous Plan Failed\n"
            f"{json.dumps(plan, indent=2)}\n"
            "\n## Validation Report\n"
            f"{json.dumps(failure_report, indent=2)}\n"
            "\n## Specific Failure\n"
            f"{failure_report.get('failure_reason', 'Unknown')}\n"
            "\n## Constraints Violated\n"
            f"{json.dumps(failure_report.get('constraint_violations', []), indent=2)}\n"
            f"{history_text}\n"
            "\nRevise the plan to avoid these failures. Output the complete revised plan."
        )

        # Build base request then append refinement context
        base_request = self.assembler.assemble(
            scene,
            task,
            model=self.model,
            temperature=min(self.temperature + 0.1, 1.0),
            max_tokens=self.max_tokens,
        )
        refined_request = LLMRequest(
            system_message=base_request.system_message,
            user_message=base_request.user_message + "\n\n" + refinement_context,
            model=base_request.model,
            temperature=base_request.temperature,
            max_tokens=base_request.max_tokens,
        )
        results = self.client.request_json(refined_request, n=1)
        return {"plan": results[0]["content"], "provenance": results[0]["provenance"]}
