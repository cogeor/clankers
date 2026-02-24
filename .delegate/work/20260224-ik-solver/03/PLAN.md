# Loop 03: IK Arm Example

## Goal
Create a headless 6-DOF arm IK example binary demonstrating the clankers-ik solver
integrated with the full simulation pipeline (physics, actuators, sensors).

## Tasks

1. Create `examples/src/bin/arm_ik.rs` — 6-DOF arm IK example
   - Parse SIX_DOF_ARM_URDF, build scene with SceneBuilder
   - Switch all actuators to Position mode (PID: kp=100, kd=10)
   - Add Rapier physics backend
   - Build KinematicChain to "end_effector" link
   - Define 6 reachable target positions
   - IK control system: solve DLS each step, write JointCommand
   - Run 300 simulation steps, print FK verification

2. Add dependencies to `examples/Cargo.toml`
   - `clankers-ik = { workspace = true, features = ["bevy"] }`
   - `nalgebra.workspace = true`
