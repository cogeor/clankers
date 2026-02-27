# Loop 03 Implementation

## Changes

### types.rs — velocity tracking weights
- vx: 5 → 20, vy: 5 → 20, vz: 1 → 5
- Height still dominant (pz=50) but velocity now 4x more influential

### Both examples — swing motor limits + blend
- Swing max_force: 30 → 60 Nm (allows full trajectory tracking for 10cm step)
- Hip_ab swing kp: 20 → 80 (reduces lateral splay at liftoff)
- Blend window: 10% → 20% of swing phase (gentler transition)

### plugin.rs + mpc_walk.rs
- Same blend window: 10% → 20%

## Test results
- `cargo test -p clankers-mpc`: 41/41 passed
- `cargo build -p clankers-examples`: clean
- `standing_maintains_height` integration test: passed
