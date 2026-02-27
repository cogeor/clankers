# Loop 01 Implementation

## Changes

### plugin.rs
1. **Force sign fix**: Negate MPC forces before J^T mapping (`-solution.forces[leg_idx]`)
2. **Velocity support**: Added `body_linear_velocity` and `body_angular_velocity` fields to `MpcPipelineState`
3. **body_state_from_transform**: Changed signature to accept velocity params instead of hardcoding zeros
4. **System**: Passes `state.body_linear_velocity` and `state.body_angular_velocity` to the function

### mpc_walk.rs
1. **Force sign fix**: Same negation as plugin
2. **Unused mut**: Removed unnecessary `mut` on solver variable

## Test results
- `cargo test -p clankers-mpc`: 41/41 passed
- `cargo build -p clankers-examples`: compiles clean
- `cargo test --test mpc_walk standing_maintains_height`: passed
