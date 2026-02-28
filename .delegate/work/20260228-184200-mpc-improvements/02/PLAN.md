# Loop 2: Decouple Viz Framerate from Simulation

## Goal
Physics/MPC runs at fixed 50Hz regardless of render frame rate. OK to drop viz frames.

## Approach
Use Bevy FixedUpdate schedule for simulation pipeline in viz mode.

## Changes

### `crates/clankers-core/src/lib.rs`
- Configure ClankersSet chain on FixedUpdate in addition to Update

### `crates/clankers-physics/src/rapier/backend.rs`
- Add `RapierBackendFixed` that registers rapier_step_system on FixedUpdate

### `crates/clankers-physics/src/rapier/mod.rs`
- Export `RapierBackendFixed`

### `crates/clankers-viz/src/plugin.rs`
- Change `ClankersVizPlugin` from unit struct to struct with `fixed_update: bool`
- When fixed_update=true, configure run conditions on FixedUpdate, move mode gate systems to FixedUpdate

### `examples/src/quadruped_setup.rs`
- Add `use_fixed_update: bool` to QuadrupedSetupConfig
- Use RapierBackendFixed when true

### `examples/src/bin/quadruped_mpc_viz.rs`
- Set FixedUpdate timestep to 0.02s
- Use `use_fixed_update: true` in setup
- Use `ClankersVizPlugin { fixed_update: true }`
- Register mpc_control_system on FixedUpdate
- Keep visual sync on Update

## Verification
- `cargo build -j 24 -p clankers-examples`
- `cargo test -j 24 -p clankers-mpc --lib`
- `cargo run --release -p clankers-examples --bin quadruped_mpc_bench -- --velocity 0.5 --gait trot` → ≥4.2m
