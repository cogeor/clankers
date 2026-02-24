# Loop 03 — Implementation

## Changes

### `examples/Cargo.toml`
- Added `clankers-physics.workspace = true` dependency

### `examples/src/bin/pendulum_viz.rs`

**Imports:**
- Added `clankers_physics::rapier::{bridge::register_robot, RapierBackend, RapierContext}`
- Added `clankers_physics::ClankersPhysicsPlugin`
- Removed `JointCommand` import (no longer used directly)

**Main function:**
- Parse URDF explicitly with `clankers_urdf::parse_string()` and keep model for physics registration
- Use `SceneBuilder::with_robot(model.clone(), ...)` instead of `with_robot_urdf()`
- Add `ClankersPhysicsPlugin::new(RapierBackend)` after scene build (so SimConfig exists)
- Register robot with rapier context: remove `RapierContext` resource, call `register_robot()` with `fixed_base: true`, re-insert resource
- Disjoint field borrowing: `scene.robots["cartpole"]` (immutable) and `scene.app.world_mut()` (mutable) borrow different fields of `SpawnedScene`

**Visual sync systems:**
- `sync_cart_visual`: reads `JointState.position` instead of `JointCommand.value`
- `sync_pivot_visual`: reads `JointState.position` instead of `JointCommand.value`
- `sync_pole_visual`: reads both `JointState.position` values instead of `JointCommand.value`

**Comments/docs:**
- Updated module-level docs to describe physics-driven visuals
- Renumbered steps in main() to account for new physics steps (3-4)
- Updated print messages to mention Rapier physics

## Architecture decision
Physics integration lives in the example, NOT in `SceneBuilder` or `clankers-sim`, to avoid
`rapier3d` as a transitive dependency for all crates. Users opt in by adding `clankers-physics`
to their own crate and calling `register_robot()` after scene build.
