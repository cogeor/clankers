# Loop 02: Smoothness — Liftoff Timing + Gain Blending

## Changes

1. **Liftoff detection**: Track `prev_contacts` per leg. Set `swing_starts` only
   at the exact stance→swing transition (not every stance frame or early swing).

2. **Gain blending**: Over the first 10% of swing phase, linearly ramp motor gains
   from stance values to swing values:
   - hip_ab: kp 500→20, kd 20→2, max 200→30
   - pitch/knee: kp 0→20, kd 5→2, max 50→30

3. **Plugin (direct torque)**: Fade out stance damping term over first 10% of swing.

## Files modified

- `crates/clankers-mpc/src/plugin.rs` — prev_contacts field, liftoff detection, damping blend
- `examples/src/bin/quadruped_mpc_viz.rs` — prev_contacts field, liftoff detection, gain blend
- `examples/src/bin/quadruped_mpc.rs` — prev_contacts, liftoff detection, gain blend
- `examples/tests/mpc_walk.rs` — prev_contacts, liftoff detection, damping blend
