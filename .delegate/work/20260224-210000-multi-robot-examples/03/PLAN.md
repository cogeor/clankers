# Loop 03: Create multi_robot_viz.rs example with per-robot meshes and visual sync

## Overview

Create `examples/src/bin/multi_robot_viz.rs` -- a windowed Bevy example that visualizes
two robots (cartpole and two-link arm) side by side with Rapier physics, individual visual
meshes, per-robot visual sync systems, and the new robot selector GUI from loops 01 & 02.

This example demonstrates the full multi-robot viz pipeline: URDF parsing, SceneBuilder
with multiple robots, Rapier registration, visual mesh spawning, JointState-driven sync,
and dynamic teleop rebinding via the `sync_teleop_to_robot` system.

## Tasks

### Task 1: Create multi_robot_viz.rs

**Goal:** Write the complete example binary following the pendulum_viz.rs pattern but
with two robots and simpler visuals (no sensor noise, no policy).

**Files:**
| Action | Path |
|--------|------|
| CREATE | `examples/src/bin/multi_robot_viz.rs` |

**Steps:**

1. Define visual marker components for both robots:
   - Cartpole: `RailVisual`, `CartVisual`, `PivotVisual`, `PoleVisual` (same as pendulum_viz)
   - Two-link arm: `ArmBaseVisual`, `UpperArmVisual`, `ForearmVisual`, `EEVisual`

2. Define joint reference resources:
   - `CartPoleJoints { cart: Entity, pole: Entity }`
   - `ArmJoints { shoulder: Entity, elbow: Entity }`

3. Write `spawn_cartpole_meshes` system:
   - Rail at x=0.0 (gray), cart (blue), pivot sphere (yellow), pole with children (red)
   - Same geometry as pendulum_viz.rs

4. Write `spawn_arm_meshes` system:
   - Base cylinder at x=3.0 (dark green, fixed)
   - Upper arm parent at x=3.0 with cylinder child (green, rotates by shoulder angle)
   - Forearm parent at x=3.0 with cylinder child (yellow, rotates by shoulder+elbow)
   - End-effector sphere (orange)

5. Write cartpole visual sync systems:
   - `sync_cart_visual`: translate cart mesh by cart_slide position
   - `sync_pivot_visual`: translate pivot to match cart
   - `sync_pole_visual`: translate + rotate pole by cart position and pole_hinge angle

6. Write arm visual sync systems:
   - `sync_upper_arm_visual`: rotate upper arm parent by shoulder angle (Z-axis)
   - `sync_forearm_visual`: position at upper arm end, rotate by shoulder+elbow angles

7. Write `main()`:
   - Parse both URDFs (cartpole + two_link_arm)
   - Build scene with `SceneBuilder::new().with_robot(...).with_robot(...)`
   - Add `ClankersPhysicsPlugin` with `RapierBackend`
   - Register both robots with `register_robot` (both fixed base)
   - Register `JointStateSensor` for observations
   - Insert joint reference resources
   - Add `DefaultPlugins` with window title "Clankers -- Multi-Robot Viz"
   - Add `ClankersTeleopPlugin` + `ClankersVizPlugin`
   - Add startup systems for both robot meshes
   - Add visual sync systems after `ClankersSet::Simulate`
   - Reset episode and run

**Verify:** `cargo build -p clankers-examples --bin multi_robot_viz --release`

## Acceptance Criteria

- [ ] `multi_robot_viz` binary compiles with `--release`
- [ ] Two robots are registered in the scene (cartpole + two_link_arm)
- [ ] Visual meshes are spatially offset (cartpole at x=0, arm at x=3)
- [ ] Visual sync systems drive mesh transforms from JointState
- [ ] No manual TeleopConfig/KeyboardTeleopMap -- `sync_teleop_to_robot` handles it
- [ ] ClankersVizPlugin provides robot selector GUI and teleop rebinding
- [ ] No sensor noise or ONNX policy (teleop-only example)
