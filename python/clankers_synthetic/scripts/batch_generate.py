#!/usr/bin/env python3
"""Batch synthetic dataset generation for arm pick-and-place.

Runs the hardcoded dry-run pick plan N times against the arm_pick_gym
server, leveraging Rapier's non-determinism for trajectory variation.
Packages passing traces into a dataset directory and reports success rate.

Usage:
    1. Start gym server:
       cargo run -j 24 -p clankers-examples --bin arm_pick_gym

    2. Run batch generation:
       python python/clankers_synthetic/scripts/batch_generate.py --n-episodes 20

    Options:
       --n-episodes N    Number of episodes to generate (default: 20)
       --port PORT       Gym server port (default: 9880)
       --output DIR      Output directory (default: output/arm_pick_batch)
       --verbose         Enable debug logging
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Batch arm pick-and-place dataset generation")
    parser.add_argument("--host", default="127.0.0.1", help="Gym server host")
    parser.add_argument("--port", type=int, default=9880, help="Gym server port")
    parser.add_argument("--n-episodes", type=int, default=20, help="Number of episodes to generate")
    parser.add_argument("--output", default="output/arm_pick_batch", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Lazy imports so --help is fast
    from clankers_synthetic.compiler import SkillCompiler
    from clankers_synthetic.packager import DatasetPackager
    from clankers_synthetic.scripts.run_arm_pick import (
        build_ik_solver,
        load_scene_and_task,
        make_env_factory,
    )
    from clankers_synthetic.specs import (
        CanonicalPlan,
        ResolvedSkill,
        ValidationMetrics,
        ValidationReport,
    )

    print("=" * 60)
    print("  Batch Arm Pick-and-Place Dataset Generation")
    print("=" * 60)

    # 1. Load scene and task
    print("\n[1/4] Loading scene and task specs...")
    scene, task = load_scene_and_task()
    print(f"  Scene: {scene.scene_id}")
    print(f"  Task:  {task.task_id}")

    # 2. Build IK solver
    print("\n[2/4] Building IK solver...")
    ik_solver = build_ik_solver(scene)

    # 3. Setup compiler
    arm_joint_names = [
        name
        for name, jtype in zip(
            scene.robot.joint_names, scene.robot.joint_types.values(), strict=False
        )
        if jtype == "revolute"
    ]
    arm_joint_limits = {
        name: limits for name, limits in scene.robot.joint_limits.items() if name in arm_joint_names
    }

    compiler = SkillCompiler(
        ik_solver=ik_solver,
        joint_names=arm_joint_names,
        joint_limits=arm_joint_limits,
        control_dt=scene.simulation.control_dt,
    )

    env_factory = make_env_factory(args.host, args.port, scene)

    # Hardcoded pick plan (same as run_arm_pick.py dry-run)
    plan = CanonicalPlan(
        plan_id="batch_pick",
        skills=[
            ResolvedSkill(
                name="move_to",
                target_world_position=[0.28, 0.0, 0.55],
                params={"speed_fraction": 0.3},
            ),
            ResolvedSkill(name="wait", params={"steps": 20}),
            ResolvedSkill(
                name="move_to",
                target_world_position=[0.28, 0.0, 0.50],
                params={"speed_fraction": 0.1},
            ),
            ResolvedSkill(
                name="set_gripper",
                params={"width": 0.0, "wait_settle_steps": 30},
            ),
            ResolvedSkill(name="wait", params={"steps": 20}),
            ResolvedSkill(
                name="move_to",
                target_world_position=[0.28, 0.0, 0.65],
                params={"speed_fraction": 0.15},
            ),
            ResolvedSkill(name="wait", params={"steps": 30}),
        ],
    )

    # 4. Run episodes
    print(f"\n[3/4] Generating {args.n_episodes} episodes...")
    print(f"  Server: {args.host}:{args.port}")

    raw_dir = os.path.join(args.output, "raw_traces")
    os.makedirs(raw_dir, exist_ok=True)

    results = []  # (trace, success, cube_z, length)
    passing_traces = []  # (trace, report) for DatasetPackager

    success_threshold = 0.525  # cube z must be >= this

    t_start = time.time()
    for ep in range(args.n_episodes):
        plan_copy = CanonicalPlan(
            plan_id=f"batch_pick_{ep:04d}",
            skills=plan.skills,
        )

        env = None
        try:
            env = env_factory()
            trace = compiler.execute(plan_copy, env)
        except Exception as e:
            logger.warning(f"  Episode {ep}: execution failed: {e}")
            results.append(
                {"episode": ep, "success": False, "cube_z": 0.0, "length": 0, "error": str(e)}
            )
            continue
        finally:
            if env is not None:
                env.close()

        # Check success from final step
        cube_z = 0.0
        success = False
        body_poses: dict = {}
        if trace.steps:
            last_info = trace.steps[-1].info
            body_poses = last_info.get("body_poses", {})
            if "red_cube" in body_poses:
                cube_z = body_poses["red_cube"][2]
                success = cube_z >= success_threshold

        length = len(trace.steps)
        results.append({"episode": ep, "success": success, "cube_z": cube_z, "length": length})

        # Save raw trace
        trace_path = os.path.join(raw_dir, f"trace_{ep:04d}.json")
        with open(trace_path, "w") as f:
            json.dump(trace.model_dump(), f, default=str)

        # Create a minimal validation report for passing traces
        if success:
            final_poses = body_poses if trace.steps else {}
            report = ValidationReport(
                passed=True,
                task_success=True,
                constraint_violations=[],
                metrics=ValidationMetrics(
                    total_steps=length,
                    max_contact_force=0.0,
                    max_joint_velocity=0.0,
                    max_ee_speed=0.0,
                    final_object_poses=final_poses,
                    success_at_step=length - 1,
                ),
            )
            passing_traces.append((trace, report))

        status = "OK" if success else "FAIL"
        print(f"  Episode {ep:3d}/{args.n_episodes}: {status}  cube_z={cube_z:.4f}  steps={length}")

    elapsed = time.time() - t_start

    # 5. Package dataset
    print("\n[4/4] Packaging dataset...")
    n_success = sum(1 for r in results if r["success"])
    n_total = len(results)
    success_rate = n_success / max(n_total, 1)

    if passing_traces:
        packager = DatasetPackager()
        manifest = packager.package(
            traces=passing_traces,
            output_dir=args.output,
            scene_spec_hash="batch",
            task_spec_hash="batch",
            llm_model="deterministic_dry_run",
        )
        print(f"  Dataset written to: {manifest.output_dir}")
        print(f"  Episodes packaged: {manifest.n_episodes}")
    else:
        print("  WARNING: No successful episodes to package!")

    # Summary report
    cube_heights = [r["cube_z"] for r in results if "cube_z" in r]
    step_counts = [r["length"] for r in results if "length" in r]

    print("\n" + "=" * 60)
    print("  BATCH GENERATION REPORT")
    print("=" * 60)
    print(f"  Total episodes:     {n_total}")
    print(f"  Successful:         {n_success}")
    print(f"  Failed:             {n_total - n_success}")
    print(f"  Success rate:       {success_rate:.1%}")
    print(f"  Time elapsed:       {elapsed:.1f}s")
    print(f"  Time per episode:   {elapsed / max(n_total, 1):.2f}s")
    if cube_heights:
        print(f"  Cube z (mean):      {sum(cube_heights) / len(cube_heights):.4f}")
        print(f"  Cube z (min):       {min(cube_heights):.4f}")
        print(f"  Cube z (max):       {max(cube_heights):.4f}")
    if step_counts:
        print(f"  Steps (mean):       {sum(step_counts) / len(step_counts):.0f}")
        print(f"  Steps (min):        {min(step_counts)}")
        print(f"  Steps (max):        {max(step_counts)}")
    print("=" * 60)

    # Save report JSON
    report_path = os.path.join(args.output, "batch_report.json")
    report_data = {
        "n_total": n_total,
        "n_success": n_success,
        "success_rate": success_rate,
        "elapsed_seconds": elapsed,
        "episodes": results,
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
