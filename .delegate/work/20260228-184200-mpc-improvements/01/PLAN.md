# Loop 1: Bezier Swing Trajectories

## Goal
Replace min-jerk quintic in `swing_foot_position`/`swing_foot_velocity` with 12-point Bezier curves (MIT Cheetah style). Boundary conditions: zero velocity and acceleration at liftoff/touchdown.

## File
`crates/clankers-mpc/src/swing.rs`

## Tasks

### 1. Add Bezier evaluation helpers
- `bezier_eval(points: &[f64; 12], t: f64) -> f64` using De Casteljau's algorithm
- `bezier_derivative(points: &[f64; 12], t: f64) -> f64` using hodograph (degree-10 Bezier of differences)

### 2. Define control point arrays
- Horizontal S-curve: `[0,0,0,0, 0.5,0.5, 0.5,0.5, 1,1,1,1]` — zero vel/accel at endpoints
- Height profile: `[0,0,0, h*0.9, h*0.9, h, h, h*0.9, h*0.9, 0,0,0]` — smooth bell with zero derivatives at endpoints

### 3. Replace `swing_foot_position` internals
- Use bezier_eval for horizontal interpolation parameter `s` (replaces `10t³-15t⁴+6t⁵`)
- Use bezier_eval for height profile (replaces `64*t³*(1-t)³`)

### 4. Replace `swing_foot_velocity` internals
- Use bezier_derivative for ds/dt
- Use bezier_derivative for dh/dt

### 5. Update tests
- Endpoint tests stay the same (boundary conditions preserved)
- Peak height test: Bezier height profile peaks at ~step_height at t=0.5
- Velocity tests: zero at endpoints, nonzero at midpoint
- Symmetry tests stay the same

## Verification
- `cargo test -j 24 -p clankers-mpc --lib`
- `cargo run --release -p clankers-examples --bin quadruped_mpc_bench -- --velocity 0.5 --gait trot` → ≥4.3m
