# Clanker: Roadmap to v1.0

Based on the deep-dive architectural analysis and cross-comparison with industry-standard simulators (Unity ML-Agents, Godot RL, Isaac Sim, MuJoCo, Copper-rs), the Clanker simulation engine is currently estimated at **90% to 95% complete** toward a viable "v1" release.

The foundational architecture—the Bevy/Rapier determinism, the Gym TCP bridging, the decoupled episodic lifecycles, the ONNX deployment pathway, the full sensor suite, and the Python training integration—is solidly in place and production-grade.

## Completed Milestones

### 1. Domain Randomization (`clankers-domain-rand`) ✅
*   **Status:** Fully implemented.
*   8 randomizer types: mass, friction, joint damping/stiffness, motor parameters, geometry scaling, external forces, center of mass, and restitution.
*   Configurable per-episode or per-interval randomization via ECS systems.

### 2. Sensor Noise Models (`clankers-noise`) ✅
*   **Status:** Fully implemented.
*   6 composable noise models: Gaussian, uniform, salt-and-pepper, quantization, drift, and bias.
*   MEMS presets for accelerometer and gyroscope noise profiles.
*   Trait-based `NoiseModel` architecture with deterministic seeded RNG.

### 3. Sensor Suite (`clankers-env`) ✅
*   **Status:** Fully implemented with comprehensive coverage.
*   **Joint sensors:** `JointStateSensor`, `JointCommandSensor`, `JointTorqueSensor` (+ robot-scoped variants).
*   **IMU sensor:** `ImuSensor` reading linear acceleration and angular velocity (+ `RobotImuSensor`).
*   **Contact sensor:** `ContactSensor` reading collision normal forces (+ `RobotContactSensor`).
*   **Raycast sensor:** `RaycastSensor` for LiDAR-like spatial awareness (+ `RobotRaycastSensor`).
*   **End-effector pose:** `EndEffectorPoseSensor` reading world-space position and orientation (+ `RobotEndEffectorPoseSensor`).
*   **Image sensor:** `ImageSensor` via `clankers-render` for RGB/RGBA observations.
*   **Noise wrapping:** `NoisySensor<S>` composable wrapper for any sensor.

### 4. Python Training Integration ✅
*   **Status:** Fully implemented.
*   `ClankerGymnasiumEnv` — standard `gymnasium.Env` subclass for Stable-Baselines3.
*   Pluggable reward functions: `DistanceReward`, `SparseReward`, `ActionPenaltyReward`, `CompositeReward`.
*   Pluggable termination conditions: `SuccessTermination`, `TimeoutTermination`, `FailureTermination`, `CompositeTermination`.
*   Training example script with PPO via `examples/train_ppo.py`.

### 5. Reward/Termination Architecture ✅
*   **Status:** Cleanly separated.
*   Rewards and termination conditions are computed entirely in Python during training.
*   Rust simulation provides observations and episode lifecycle; Python computes task-level logic.
*   Follows the principle: simulation physics in Rust, task logic in Python.

## Remaining Priorities (The Last 5%)

### 1. URDF Hardening / Constraints
*   **Status:** Basic parsing of standard arms is functional (`clankers-urdf`).
*   **Remaining:**
    *   Stress-test against complex community-standard URDFs (edge-cases in joint limits, inertial properties).
    *   Evaluate MJCF format support for tendon-driven robots (may push to v1.1).

### 2. Deployment Bridge
*   **Status:** Not started.
*   **Remaining:**
    *   Binary trajectory logging format for deterministic replay.
    *   Minimal ROS2 bridge for porting ONNX policies to physical robots.

### 3. Documentation & Examples
*   **Status:** Core docs complete, examples minimal.
*   **Remaining:**
    *   End-to-end tutorial: URDF → training → deployment.
    *   Additional example environments (locomotion, manipulation).

## Estimate

With the core sensor suite, domain randomization, noise models, and Python integration all complete, the engine is ready for early adopters. Hardening for a `v1.0.0` release (URDF edge-cases, additional examples, deployment bridge) is achievable within **1 to 2 months**.
