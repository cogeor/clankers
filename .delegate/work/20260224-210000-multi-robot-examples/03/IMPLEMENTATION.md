# Implementation Log

## Task 1: Create multi_robot_viz.rs

Completed: 2026-02-24T21:45:00Z

### Changes

- `examples/src/bin/multi_robot_viz.rs`: Created new example binary demonstrating two robots (cartpole + two-link arm) in a single Bevy window with Rapier physics and the robot selector GUI.

  **Visual markers (8 components):**
  - Cartpole: `RailVisual`, `CartVisual`, `PivotVisual`, `PoleVisual` (same pattern as pendulum_viz)
  - Two-link arm: `ArmBaseVisual`, `UpperArmVisual`, `ForearmVisual`, `EEVisual`

  **Joint reference resources (2 resources):**
  - `CartPoleJoints { cart, pole }` for cartpole joint entities
  - `ArmJoints { shoulder, elbow }` for arm joint entities

  **Geometry constants:**
  - `ARM_X_OFFSET = 3.0` (spatial separation from cartpole)
  - `ARM_BASE_HEIGHT = 0.05`, `UPPER_ARM_LEN = 0.3`, `FOREARM_LEN = 0.25` (match URDF)

  **Startup systems (2):**
  - `spawn_cartpole_meshes`: Rail (gray), cart (blue), pivot (yellow), pole with children (red) -- identical to pendulum_viz
  - `spawn_arm_meshes`: Base cylinder (dark green), upper arm parent with cylinder child (green), forearm parent with cylinder child (yellow), end-effector sphere (orange)

  **Visual sync systems (5):**
  - `sync_cart_visual`: Translates cart mesh by cart_slide JointState position
  - `sync_pivot_visual`: Translates pivot sphere to match cart X
  - `sync_pole_visual`: Translates + rotates pole by cart position and pole_hinge angle (Z-rotation)
  - `sync_upper_arm_visual`: Rotates upper arm parent by shoulder angle (Z-rotation)
  - `sync_forearm_visual`: Computes upper arm tip via trigonometry, positions forearm at tip and rotates by shoulder+elbow, positions end-effector at forearm tip

  **Main function pipeline:**
  1. Parse both URDFs (cartpole + two_link_arm)
  2. Build scene with `SceneBuilder::new().with_robot(...).with_robot(...).build()`
  3. Extract joint entities from both SpawnedRobots
  4. Add `ClankersPhysicsPlugin` with `RapierBackend`
  5. Register both robots with `register_robot` (both fixed base)
  6. Register `JointStateSensor::new(4)` for all 4 joints
  7. Insert joint reference resources
  8. Add `DefaultPlugins` with window title "Clankers -- Multi-Robot Viz"
  9. Add `ClankersTeleopPlugin` + `ClankersVizPlugin` (no manual TeleopConfig/KeyboardTeleopMap -- `sync_teleop_to_robot` from loop 02 handles automatic rebinding)
  10. Add startup systems for both robot meshes
  11. Add visual sync systems after `ClankersSet::Simulate`
  12. Reset episode and run

### Verification

- [x] `cargo build -p clankers-examples --bin multi_robot_viz --release`: compiles with no errors or warnings
- [x] `cargo test -p clankers-examples`: all tests pass (0 failures)
- [x] `cargo test -p clankers-viz`: all 21 tests pass (no regressions from loops 01/02)
- [x] Two robots registered in scene via SceneBuilder (cartpole + two_link_arm)
- [x] Visual meshes spatially offset (cartpole at x=0, arm at x=3)
- [x] No manual TeleopConfig/KeyboardTeleopMap -- relies on `sync_teleop_to_robot`
- [x] No sensor noise or ONNX policy (teleop-only example)

### Notes

The two-link arm visual sync uses forward kinematics trigonometry to compute joint positions:
- Upper arm tip: `base + upper_arm_len * rot(shoulder_angle)` applied to up vector
- Forearm tip: `upper_tip + forearm_len * rot(shoulder_angle + elbow_angle)` applied to up vector

The `sync_forearm_visual` system uses query filters `(With<ForearmVisual>, Without<EEVisual>)` and `(With<EEVisual>, Without<ForearmVisual>)` to avoid query conflicts on the Transform component.

---
