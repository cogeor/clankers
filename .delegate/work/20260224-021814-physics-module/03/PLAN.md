# Loop 03 — Wire physics into pendulum_viz example

## Goal
Integrate `clankers-physics` (Rapier backend) into the `pendulum_viz` example so
the cart-pole is driven by real physics simulation instead of teleop command values.

## Tasks

### 1. Add `clankers-physics` dependency to examples
- Add `clankers-physics.workspace = true` to `examples/Cargo.toml`

### 2. Update `pendulum_viz.rs`
- Parse URDF explicitly via `clankers_urdf::parse_string` (keep model for physics registration)
- Use `SceneBuilder::with_robot(model.clone(), ...)` instead of `with_robot_urdf`
- Add `ClankersPhysicsPlugin::new(RapierBackend)` after scene build (SimConfig must exist first)
- Register robot with rapier context via `register_robot(&mut ctx, &model, spawned, world, true)`
- Switch visual sync systems from reading `JointCommand.value` to `JointState.position`
- Update doc comments to reflect physics-driven visuals

### 3. Verify
- `cargo check -p clankers-examples`
- `cargo test --workspace`

## Architecture notes
- Physics integration stays in the example, NOT in `SceneBuilder`, to avoid `rapier3d` as transitive dep
- The `fixed_base: true` flag anchors the rail link to the world
- Visual mesh axes don't match URDF axes exactly (hand-crafted visual), but angle magnitudes are correct
