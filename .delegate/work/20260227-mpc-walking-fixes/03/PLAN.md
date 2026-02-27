# Loop 03: Tune MPC weights, swing gains, motor limits

## Changes

### 1. Increase velocity tracking weights (types.rs)
- vx: 5.0 → 20.0
- vy: 5.0 → 20.0
- vz: 1.0 → 5.0
Rationale: With pz=50, velocity was 10x under-weighted. Robot holds height
well but barely tracks commanded velocity (0.08 vs 0.30 m/s).

### 2. Raise swing motor max_force (both examples)
- Swing floor: 30 Nm → 60 Nm
- Stance hip_ab: 200 Nm (unchanged — already high)
- Stance pitch/knee: 50 Nm (unchanged)
Rationale: 30 Nm clips trajectory tracking for 10cm step height, especially
for knee joint with longer moment arm.

### 3. Soften gain blending (both examples)
- Increase blend window from 10% → 20% of swing phase
- Raise hip_ab swing kp floor from 20 → 80 (less lateral splay)
Rationale: 96% kp drop (500→20) in 10% of swing causes lateral splay at liftoff.

### 4. Sync mpc_walk.rs test
- Same blend window change (10% → 20%)

## Files modified
- `crates/clankers-mpc/src/types.rs` — q_weights
- `examples/src/bin/quadruped_mpc_viz.rs` — max_force, blend
- `examples/src/bin/quadruped_mpc.rs` — max_force, blend
- `examples/tests/mpc_walk.rs` — blend window
- `crates/clankers-mpc/src/plugin.rs` — blend window
