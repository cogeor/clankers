# CityBuilder

Rust-based robotics simulator with LLM-driven synthetic data generation and a Gymnasium-compatible training interface. Built on Bevy ECS and Rapier 3D physics, with Python training via TCP protocol.

## What It Does

- **LLM-driven synthetic motion data** — GPT-4 plans manipulation tasks, the simulator executes and validates them, producing labeled trajectory datasets at scale
- **Behavioral cloning pipeline** — train models from synthetic trajectories with automatic joint encoding, velocity/position prediction modes, and ONNX export
- **Gymnasium-compatible training** — PPO, SAC, or any RL algorithm via Stable Baselines3 over a TCP gym protocol
- **Rust simulation engine** — Bevy ECS plugin architecture with Rapier 3D rigid-body physics, URDF loading, domain randomization, and motor models
- **Synthetic image generation** — headless GPU rendering for camera and depth observations, with generative AI augmentation pipeline
- **ONNX policy deployment** — train in Python, export to ONNX, run natively in the simulator at full speed

## Architecture

```
Python (training)                    Rust (simulation)
-----------------                    ------------------
SB3 / PyTorch                        Bevy ECS
    |                                    |
Gymnasium env <---- TCP/JSON ----> citybuilder-gym server
    |                                    |
rewards.py                         +----+----+
terminations.py                    | Rapier  |
joint_encoder.py                   | physics |
trajectory_dataset.py              +----+----+
                                        |
                                   URDF loading
                                   Motor models
                                   Domain randomization
                                   ONNX inference
                                   3D visualization
```

Training happens in Python. Simulation happens in Rust. They communicate over TCP with JSON messages. Trained policies export to ONNX and run natively in the simulator at full speed.

## Project Structure

```
crates/
+-- citybuilder-core          System ordering, SimTime, Sensor/ActionApplicator traits
+-- citybuilder-noise         Gaussian, uniform, bias, drift noise models
+-- citybuilder-actuator-core Motor model math (transmission, friction) -- no Bevy
+-- citybuilder-actuator      Bevy plugin: PD control, joint components, dynamics
+-- citybuilder-env           Episode lifecycle, sensors (joint, IMU, contact, camera)
+-- citybuilder-urdf          URDF parsing and Bevy entity spawning
+-- citybuilder-physics       Rapier 3D backend with configurable solver
+-- citybuilder-policy        ONNX inference runner for trained policies
+-- citybuilder-domain-rand   Per-episode physics randomization for sim-to-real
+-- citybuilder-gym           TCP server, Gymnasium protocol, VecEnv support
+-- citybuilder-render        Headless GPU rendering for image observations
+-- citybuilder-viz           Interactive 3D visualization with egui
+-- citybuilder-teleop        Manual control interfaces for debugging
+-- citybuilder-ik            Inverse kinematics solver
+-- citybuilder-mpc           Centroidal convex MPC + whole-body controller
+-- citybuilder-sim           Top-level plugin, SceneBuilder
+-- citybuilder-record        MCAP episode recording with provenance
+-- citybuilder-test-utils    Shared test fixtures

python/
+-- citybuilder/              Training client library
|   +-- client.py          TCP client
|   +-- gymnasium_env.py   Full Gymnasium interface
|   +-- sb3_vec_env.py     Stable Baselines3 vectorized wrapper
|   +-- joint_encoder.py   Robot-agnostic joint position encoding
|   +-- trajectory_dataset.py  PyTorch Dataset for offline training
|   +-- rewards.py         Reward function templates
|   +-- terminations.py    Episode termination conditions
|   +-- augmentation/      Synthetic image augmentation with diffusion models
|   ...
+-- citybuilder_synthetic/    LLM-driven synthetic trajectory generation
|   +-- pipeline.py        End-to-end: plan -> compile -> validate -> package
|   +-- compiler.py        Skill execution through simulation
|   +-- planner.py         LLM plan generation (GPT-4)
|   ...
+-- examples/              Training scripts
    +-- train_ppo.py       PPO training with SB3
    +-- train_joint_bc.py  Behavioral cloning from trajectories
    +-- replay_policy.py   Policy vs ground-truth comparison
    ...

examples/
+-- src/bin/               20 Rust binaries (pendulum, cartpole, arm, quadruped)
+-- urdf/                  Robot models (pendulum, cartpole, 6-DOF arm, quadruped)
```

## Quick Start

Build and test:

```sh
cargo test --workspace -j 24
cargo build --release -j 24
```

Run a CartPole gym server, then train PPO from Python:

```sh
# Terminal 1: start the simulator
cargo run -j 24 --release -p citybuilder-examples --bin cartpole_gym

# Terminal 2: train
pip install -e "python[sb3]"
python python/examples/cartpole_train_ppo.py
```

Run the quadruped with MPC:

```sh
cargo run -j 24 --release -p citybuilder-examples --bin quadruped_mpc_viz
```

## Robots

| Robot | DOF | URDF | Examples |
|-------|-----|------|----------|
| Pendulum | 1 | `examples/urdf/pendulum.urdf` | `pendulum_headless`, `pendulum_viz` |
| CartPole | 2 | `examples/urdf/cartpole.urdf` | `cartpole_gym`, `cartpole_vec_gym`, `cartpole_policy_viz` |
| 6-DOF Arm | 8 | `examples/urdf/six_dof_arm.urdf` | `arm_gym`, `arm_ik_viz`, `arm_pick_gym`, `arm_pick_replay` |
| Quadruped | 12 | `examples/urdf/quadruped.urdf` | `quadruped_mpc`, `quadruped_mpc_viz`, `quadruped_mpc_bench` |

## Training Pipeline

**Online RL** — simulator runs a gym server, Python agent connects and trains:

```sh
cargo run -j 24 -p citybuilder-examples --bin arm_gym
python python/examples/train_ppo.py --port 9879
python python/examples/export_sb3_to_onnx.py  # export to ONNX
cargo run -j 24 -p citybuilder-examples --bin arm_with_policy  # run in Rust
```

**Offline BC** — train from recorded trajectory data:

```sh
python python/examples/train_joint_bc.py \
    --trace output/arm_pick_dataset/dry_run_trace.json \
    --scene python/citybuilder_synthetic/scenes/arm_pick_cube.json \
    --mode velocity
python python/examples/replay_policy.py \
    --model joint_bc.onnx \
    --trace output/arm_pick_dataset/dry_run_trace.json --plot
```

**Synthetic data generation** — LLM plans a manipulation task, simulator executes and validates:

```sh
python -m citybuilder_synthetic \
    --scene python/citybuilder_synthetic/scenes/arm_pick_cube.json \
    --task python/citybuilder_synthetic/scenes/arm_pick_cube_task.json
```

## Joint Encoder

Any robot's joints are encoded in alphabetical order for deterministic vector layout:

```python
from citybuilder.joint_encoder import JointEncoder

encoder = JointEncoder(["wrist", "elbow", "shoulder"])
# Sorted: elbow, shoulder, wrist -> indices 0, 1, 2

vec = encoder.encode({"wrist": -0.3, "shoulder": 0.5, "elbow": 1.2})
# array([1.2, 0.5, -0.3])  -- always alphabetical

restored = encoder.decode(vec)
# {"elbow": 1.2, "shoulder": 0.5, "wrist": -0.3}
```

Models trained with the encoder embed joint metadata in ONNX, so any model with matching input/output dimensions works.

## Key Design Decisions

- **Bevy ECS** for modularity — each feature is a plugin, compose what you need
- **Rapier 3D physics** — rigid-body simulation with configurable solver iterations
- **TCP protocol** keeps Python and Rust cleanly separated — no FFI, no shared memory
- **ONNX export** for policies — train in Python, deploy in Rust at native speed
- **Alphabetic joint encoding** for robot-agnostic DL — same model code works across robots
- **MCAP recording** for data provenance — replay any episode offline

## License

MIT
