# Loop 2 Implementation: Decouple Viz Framerate from Simulation

## Changes

### `crates/clankers-core/src/lib.rs`
- Added `ClankersSet` chain configuration on `FixedUpdate` in addition to `Update`

### `crates/clankers-physics/src/rapier/backend.rs`
- Extracted shared `insert_rapier_context()` helper
- Added `RapierBackendFixed` that registers `rapier_step_system` on `FixedUpdate`
- Original `RapierBackend` unchanged (backward compatible)

### `crates/clankers-physics/src/rapier/mod.rs`
- Export `RapierBackendFixed`

### `crates/clankers-viz/src/plugin.rs`
- Changed `ClankersVizPlugin` from unit struct to `{ fixed_update: bool }` with `Default`
- When `fixed_update=true`: mode gate, teleop sync, and ClankersSet run conditions on `FixedUpdate`
- When `fixed_update=false`: original `Update` behavior preserved

### `crates/clankers-viz/src/lib.rs`
- Updated doctest to use `ClankersVizPlugin::default()`

### `examples/src/quadruped_setup.rs`
- Added `use_fixed_update: bool` to `QuadrupedSetupConfig`
- Uses `RapierBackendFixed` when `use_fixed_update=true`

### `examples/src/bin/quadruped_mpc_viz.rs`
- Set `use_fixed_update: true` in setup config
- Set `Time::<Fixed>::from_seconds(mpc_dt)` for 50Hz FixedUpdate
- `mpc_control_system` registered on `FixedUpdate` instead of `Update`
- Visual sync systems remain on `Update` (render-rate)

### Other viz binaries (backward compat)
- `cartpole_policy_viz.rs`, `multi_robot_viz.rs`, `pendulum_viz.rs`: changed to `ClankersVizPlugin::default()`
