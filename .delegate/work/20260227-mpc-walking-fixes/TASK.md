# TASK: Fix MPC Quadruped Walking Quality

## Problem

The quadruped MPC walks but poorly: stumbling during walk/trot, low swing height,
velocity under-tracking (0.08 m/s vs 0.30 m/s command).

## Root Causes (from diagnostic)

### Fundamental Bugs (plugin.rs)
1. **Force sign convention**: Plugin does NOT negate MPC forces before J^T mapping.
   Both standalone examples correctly use `-force`. MPC produces ground reaction
   forces (up); body must apply `-force` through foot.
2. **Zero body velocities**: `body_state_from_transform` zeros angular_velocity
   and linear_velocity. MPC state vector includes velocities — solver thinks robot
   is stationary every frame.

### Performance
3. **MPC solve time 55-67ms** vs 20ms budget (dt=0.02). Root cause: per-solve
   Clarabel overhead — dense→CSC conversion, solver construction, settings rebuild.
   The actual IPM on 120-var QP should take <5ms.

### Tuning
4. **Swing max_force**: 30 Nm may clip trajectory tracking for 10cm step height.
5. **Gain blending**: Hip_ab kp drops 96% (500→20) in 10% of swing — too aggressive.
6. **Velocity tracking weights**: vx=5, vy=5 vs pz=50 — height dominates velocity.

## Files

- `crates/clankers-mpc/src/plugin.rs` — force sign fix, velocity from Bevy
- `crates/clankers-mpc/src/solver.rs` — Clarabel overhead reduction
- `crates/clankers-mpc/src/types.rs` — weight tuning
- `examples/src/bin/quadruped_mpc_viz.rs` — gain blending, max_force tuning
- `examples/src/bin/quadruped_mpc.rs` — same tuning as viz
- `examples/tests/mpc_walk.rs` — keep in sync
