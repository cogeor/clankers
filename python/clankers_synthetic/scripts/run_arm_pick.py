#!/usr/bin/env python3
"""End-to-end arm pick-and-place synthetic data generation.

Usage:
    1. Start gym server:
       cargo run -p clankers-examples --bin arm_pick_gym

    2. Run this script:
       python python/clankers_synthetic/scripts/run_arm_pick.py

    Options:
       --port PORT        Gym server port (default: 9880)
       --n-plans N        Number of LLM plans to generate (default: 3)
       --model MODEL      LLM model name (default: gpt-4o)
       --output DIR       Output directory (default: output/arm_pick_dataset)
       --dry-run          Skip LLM calls, use a hardcoded test plan
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path


def _project_root() -> Path:
    """Find the project root (directory containing Cargo.toml)."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "Cargo.toml").exists():
            return parent
    return Path.cwd()


def load_scene_and_task(
    scene_path: str | None = None,
    task_path: str | None = None,
):
    """Load SceneSpec and TaskSpec from JSON files."""
    from clankers_synthetic.specs import SceneSpec, TaskSpec

    root = _project_root()
    scenes_dir = root / "python" / "clankers_synthetic" / "scenes"

    if scene_path is None:
        scene_path = str(scenes_dir / "arm_pick_cube.json")
    if task_path is None:
        task_path = str(scenes_dir / "arm_pick_cube_task.json")

    with open(scene_path) as f:
        scene = SceneSpec(**json.load(f))
    with open(task_path) as f:
        task = TaskSpec(**json.load(f))

    return scene, task


def make_env_factory(host: str, port: int, scene):
    """Create an env factory that returns ClankersBridgeEnv instances."""
    from clankers_synthetic.clankers_bridge import ClankersBridgeEnv

    n_arm = sum(1 for jt in scene.robot.joint_types.values() if jt == "revolute")
    n_gripper = sum(1 for jt in scene.robot.joint_types.values() if jt == "prismatic")

    def factory():
        env = ClankersBridgeEnv(
            host=host,
            port=port,
            n_arm_joints=n_arm,
            n_gripper_joints=n_gripper,
        )
        return env

    return factory


def build_ik_solver(scene):
    """Build a Python IK solver from the scene spec."""
    from clankers_synthetic.ik_solver import DlsSolver, KinematicChain

    root = _project_root()
    urdf_path = root / scene.robot.urdf_path

    if not urdf_path.exists():
        print(f"  WARNING: URDF not found at {urdf_path}, IK solver disabled")
        return None

    # base_link is the URDF root; for this arm it's "base"
    base_link = "base"
    chain = KinematicChain.from_urdf(
        str(urdf_path),
        base_link=base_link,
        ee_link=scene.robot.ee_link_name,
    )
    solver = DlsSolver(
        chain=chain,
        max_iterations=100,
        tolerance=1e-4,
        damping=0.01,
    )
    print(f"  IK solver: {len(chain.joints)} DOF chain to '{scene.robot.ee_link_name}'")
    return solver


def run_dry_run(scene, task, env_factory, ik_solver, output_dir: str):
    """Run a simple hardcoded plan without LLM calls."""
    from clankers_synthetic.compiler import SkillCompiler
    from clankers_synthetic.specs import CanonicalPlan, ResolvedSkill

    print("\n--- DRY RUN: using hardcoded pick plan ---\n")

    # Hardcoded pick plan — tuned for this arm + cube geometry:
    #   Cube center: [0.3, 0, 0.425], 2.5cm side (collision box)
    #   Table top:   z=0.4125
    #   At reaching config, fingers are ~0.030 below EE
    #   Finger collision: 1cmx1cmx4cm box, inner edges at y=+-0.010 when closed
    #   Cube width 0.025 > finger gap 0.020 -> gripper CAN squeeze
    #   IK + physics settling causes ~0.054 undershoot in z
    #   Target EE z=0.51 → settled ~0.456 → fingers ~0.426 ≈ cube center
    plan = CanonicalPlan(
        plan_id="dry_run_001",
        skills=[
            # 1. Approach above cube (x=0.28 centers finger collision on cube)
            ResolvedSkill(
                name="move_to",
                target_world_position=[0.28, 0.0, 0.55],
                params={"speed_fraction": 0.3},
            ),
            # 2. Settle above cube
            ResolvedSkill(
                name="wait",
                params={"steps": 20},
            ),
            # 3. Lower to grasp height
            #    Finger body settles ~0.03 below EE, collider tip 0.02 below body
            #    Need finger tip above table (z=0.4125) and at cube center (z=0.425)
            #    Target EE z=0.50 → settled ~0.47 → finger body ~0.44 → tip ~0.42
            ResolvedSkill(
                name="move_to",
                target_world_position=[0.28, 0.0, 0.50],
                params={"speed_fraction": 0.1},
            ),
            # 4. Close gripper immediately (no wait — avoid gravity droop)
            ResolvedSkill(
                name="set_gripper",
                params={"width": 0.0, "wait_settle_steps": 30},
            ),
            # 5. Settle with cube gripped
            ResolvedSkill(
                name="wait",
                params={"steps": 20},
            ),
            # 6. Lift
            ResolvedSkill(
                name="move_to",
                target_world_position=[0.28, 0.0, 0.65],
                params={"speed_fraction": 0.15},
            ),
            # 7. Hold at top
            ResolvedSkill(
                name="wait",
                params={"steps": 30},
            ),
        ],
    )

    # Arm-only joint names (exclude gripper for IK)
    arm_joint_names = [
        name
        for name, jtype in zip(
            scene.robot.joint_names,
            scene.robot.joint_types.values(),
            strict=False,
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

    env = env_factory()
    try:
        trace = compiler.execute(plan, env)
        print(f"  Execution complete: {len(trace.steps)} steps")
        print(f"  Terminated: {trace.terminated}, Truncated: {trace.truncated}")
        if trace.steps:
            last_info = trace.steps[-1].info
            body_poses = last_info.get("body_poses", {})
            if "red_cube" in body_poses:
                cube_z = body_poses["red_cube"][2]
                print(f"  Final red_cube z={cube_z:.4f}")
            if "end_effector" in body_poses:
                ee = body_poses["end_effector"][:3]
                print(f"  Final EE pos=[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")

        # Save trace as JSON
        os.makedirs(output_dir, exist_ok=True)
        trace_path = os.path.join(output_dir, "dry_run_trace.json")
        with open(trace_path, "w") as f:
            json.dump(trace.model_dump(), f, indent=2, default=str)
        print(f"  Trace saved to: {trace_path}")
    finally:
        env.close()


def run_full_pipeline(
    scene,
    task,
    env_factory,
    ik_solver,
    output_dir: str,
    n_plans: int,
    model: str,
):
    """Run the full LLM-driven pipeline."""
    from clankers_synthetic.pipeline import generate_dataset

    print(f"\n--- FULL PIPELINE: {n_plans} plans, model={model} ---\n")

    manifest = generate_dataset(
        scene=scene,
        task=task,
        output_dir=output_dir,
        env_factory=env_factory,
        n_plans=n_plans,
        model=model,
        ik_solver=ik_solver,
    )

    print("\nPipeline complete!")
    print(f"  Output: {manifest.output_dir}")
    print(f"  Episodes: {manifest.n_episodes}")
    print(f"  Original: {manifest.n_original}")
    print(f"  Stats: {manifest.stats}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Arm pick-and-place synthetic data generation")
    parser.add_argument("--host", default="127.0.0.1", help="Gym server host")
    parser.add_argument("--port", type=int, default=9880, help="Gym server port")
    parser.add_argument("--n-plans", type=int, default=3, help="Number of LLM plans")
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    parser.add_argument("--output", default="output/arm_pick_dataset", help="Output directory")
    parser.add_argument("--scene", default=None, help="Scene JSON path")
    parser.add_argument("--task", default=None, help="Task JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Use hardcoded plan, no LLM")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("  Arm Pick-and-Place Synthetic Data Generation")
    print("=" * 60)

    # 1. Load scene and task
    print("\n[1/4] Loading scene and task specs...")
    scene, task = load_scene_and_task(args.scene, args.task)
    print(f"  Scene: {scene.scene_id}")
    print(f"  Task:  {task.task_id}")
    print(f"  Robot: {scene.robot.name} ({len(scene.robot.joint_names)} joints)")
    print(f"  Objects: {[o.name for o in scene.objects]}")

    # 2. Build IK solver
    print("\n[2/4] Building IK solver...")
    ik_solver = build_ik_solver(scene)

    # 3. Create env factory
    print(f"\n[3/4] Creating env factory (connecting to {args.host}:{args.port})...")
    env_factory = make_env_factory(args.host, args.port, scene)

    # 4. Run pipeline
    print("\n[4/4] Running pipeline...")
    if args.dry_run:
        run_dry_run(scene, task, env_factory, ik_solver, args.output)
    else:
        run_full_pipeline(
            scene,
            task,
            env_factory,
            ik_solver,
            args.output,
            args.n_plans,
            args.model,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
