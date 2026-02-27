# Loop 01: Fix plugin force sign + zero velocity bugs

## Changes

### 1. Negate MPC forces in plugin.rs stance control (line 289)
- Change `jacobian_transpose_torques(&jacobian, force)` to `jacobian_transpose_torques(&jacobian, &(-force))`
- MPC produces ground reaction forces (ground pushes up on foot)
- Body must apply -force through J^T (Newton's 3rd law)
- Both standalone examples already do this correctly

### 2. Negate MPC forces in mpc_walk.rs test (line 339)
- Same fix as plugin.rs

### 3. Add velocity fields to MpcPipelineState
- Add `body_linear_velocity: Vector3<f64>` and `body_angular_velocity: Vector3<f64>`
- These get read by the MPC system instead of hardcoding zeros
- User updates them each frame (keeps clankers-mpc independent of physics backend)

### 4. Modify body_state_from_transform to accept velocities
- Change signature to accept linear_velocity and angular_velocity params
- System passes state.body_linear_velocity and state.body_angular_velocity

## Files modified
- `crates/clankers-mpc/src/plugin.rs`
- `examples/tests/mpc_walk.rs`
