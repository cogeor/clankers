"""End-to-end synthetic dataset generation pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from clankers_synthetic.compiler import SkillCompiler
from clankers_synthetic.openai_client import OpenAIClient
from clankers_synthetic.packager import DatasetPackager
from clankers_synthetic.parser import PlanParser
from clankers_synthetic.planner import LLMPlanner, PromptAssembler
from clankers_synthetic.pvcb_refiner import PVCBRefiner
from clankers_synthetic.specs import (
    CanonicalPlan,
    DatasetManifest,
    PlanRejection,
    SceneSpec,
    TaskSpec,
)
from clankers_synthetic.validator import SimValidator

logger = logging.getLogger(__name__)


def generate_dataset(
    scene: SceneSpec,
    task: TaskSpec,
    output_dir: str,
    *,
    env_factory: Any = None,  # Callable that returns a StepEnv
    n_plans: int = 10,
    model: str = "gpt-5",
    temperature: float = 0.3,
    max_refine_iters: int = 3,
    n_augmentations: int = 0,
    seed: int = 42,
    openai_api_key: str | None = None,
    ik_solver: Any | None = None,
) -> DatasetManifest:
    """Generate a synthetic trajectory dataset.

    Pipeline:
    1. For each of n_plans:
       a. LLM proposes a skill plan
       b. Parser validates + canonicalizes
       c. Compiler executes through env
       d. Validator checks hard/soft gates
       e. If failed: PVCBRefiner attempts fix
    2. Package passing traces to dataset

    Args:
        scene: SceneSpec describing the scene.
        task: TaskSpec describing the task goal.
        output_dir: Directory to write dataset to.
        env_factory: Callable that creates a new env instance. Called per plan.
        n_plans: Number of LLM plan proposals.
        model: LLM model name.
        temperature: LLM sampling temperature.
        max_refine_iters: Max PVCB refinement iterations per plan.
        n_augmentations: Number of augmentation variants (0 = none, future use).
        seed: Random seed.
        openai_api_key: Optional API key override.
        ik_solver: Optional DlsSolver for Cartesian skill compilation.

    Returns:
        DatasetManifest with dataset metadata.
    """
    # Initialize pipeline components
    client = OpenAIClient(api_key=openai_api_key)
    assembler = PromptAssembler()
    planner = LLMPlanner(
        assembler=assembler,
        client=client,
        model=model,
        temperature=temperature,
        n_candidates=1,
    )
    parser = PlanParser()
    validator = SimValidator(
        ee_link_name=scene.robot.ee_link_name,
        joint_limits=scene.robot.joint_limits,
    )
    refiner = PVCBRefiner(
        planner=planner,
        parser=parser,
        max_iterations=max_refine_iters,
    )
    compiler = SkillCompiler(
        ik_solver=ik_solver,
        joint_names=scene.robot.joint_names,
        joint_limits=scene.robot.joint_limits,
        control_dt=scene.simulation.control_dt,
    )
    packager = DatasetPackager()

    # Compute hashes for provenance
    scene_hash = hashlib.sha256(
        json.dumps(scene.dict(), sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    task_hash = hashlib.sha256(
        json.dumps(task.dict(), sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    # Generate plans
    passing_traces = []
    all_plans = []
    all_prompts = []

    n_proposed = 0
    n_parsed = 0
    n_passed = 0
    n_refined = 0

    for plan_idx in range(n_plans):
        logger.info(f"Generating plan {plan_idx + 1}/{n_plans}")

        # 1. Propose plan via LLM
        try:
            candidates = planner.propose(scene, task, n_candidates=1, seed=seed + plan_idx)
        except Exception as e:
            logger.warning(f"Plan {plan_idx}: LLM proposal failed: {e}")
            continue

        n_proposed += 1

        if not candidates:
            continue

        candidate = candidates[0]
        raw_plan = candidate["plan"]
        provenance = candidate.get("provenance", {})

        # 2. Parse and validate
        result = parser.parse(raw_plan, scene)
        if isinstance(result, PlanRejection):
            logger.info(f"Plan {plan_idx}: rejected by parser: {result.reasons}")
            continue

        n_parsed += 1
        canonical_plan: CanonicalPlan = result  # type: ignore

        # 3. Execute through env
        if env_factory is None:
            logger.warning("No env_factory provided -- skipping execution")
            continue

        env = env_factory()
        try:
            trace = compiler.execute(canonical_plan, env)
        except Exception as e:
            logger.warning(f"Plan {plan_idx}: execution failed: {e}")
            continue

        # 4. Validate
        report = validator.validate(trace, task, scene.constraints)

        if report.passed:
            n_passed += 1
            passing_traces.append((trace, report))
            all_plans.append(canonical_plan.dict())
            all_prompts.append(provenance)
        else:
            # 5. PVCB refinement
            logger.info(f"Plan {plan_idx}: failed validation, attempting refinement")
            refined = refiner.refine(canonical_plan, report, scene, task)

            if refined is not None:
                # Re-execute refined plan
                env2 = env_factory()
                try:
                    trace2 = compiler.execute(refined, env2)
                    report2 = validator.validate(trace2, task, scene.constraints)
                    if report2.passed:
                        n_refined += 1
                        n_passed += 1
                        passing_traces.append((trace2, report2))
                        all_plans.append(refined.dict())
                        all_prompts.append(provenance)
                except Exception as e:
                    logger.warning(f"Plan {plan_idx}: refined execution failed: {e}")

    logger.info(
        f"Pipeline complete: {n_proposed} proposed, {n_parsed} parsed, "
        f"{n_passed} passed ({n_refined} via refinement)"
    )

    if not passing_traces:
        logger.warning("No passing traces -- writing empty dataset")
        os.makedirs(output_dir, exist_ok=True)
        return DatasetManifest(
            output_dir=output_dir,
            n_episodes=0,
            n_original=0,
            n_augmented=0,
            schema_version=DatasetPackager.SCHEMA_VERSION,
            scene_spec_hash=scene_hash,
            task_spec_hash=task_hash,
            prompt_template_version=PromptAssembler.TEMPLATE_VERSION,
            llm_model=model,
            stats={"validation_pass_rate": 0.0},
        )

    # 6. Package
    manifest = packager.package(
        traces=passing_traces,
        output_dir=output_dir,
        scene_spec_hash=scene_hash,
        task_spec_hash=task_hash,
        prompt_template_version=PromptAssembler.TEMPLATE_VERSION,
        llm_model=model,
        seed=seed,
        plans=all_plans,
        prompts=all_prompts,
    )

    return manifest
