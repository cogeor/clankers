# TASK: MPC Locomotion Improvements

## Context

The quadruped MPC controller has several issues identified through testing:
- Feet accelerate too quickly during swing (would break real actuators)
- Viz is unstable at 1 m/s due to frame-rate coupling with physics
- No torque rate limiting
- Raibert heuristic needs velocity-dependent tuning
- Fixed gait timing at all speeds
- No high-frequency inner loop

## Requirements

### 1. Bezier Swing Trajectories
Replace linear interpolation in `swing_foot_position`/`swing_foot_velocity` with 10-12 point Bezier curves (MIT Cheetah style). Boundary conditions: zero velocity and acceleration at liftoff/touchdown.

### 2. Decouple Viz Framerate from Simulation
The viz binary ties MPC to Bevy frame rate. Decouple so physics/MPC runs at fixed 50Hz regardless of render frame rate. OK to drop viz frames.

### 3. Torque Rate Clamping
Add `tau = clamp(tau_desired, tau_prev - delta_max, tau_prev + delta_max)` at actuator output. Not inside MPC QP.

### 4. Raibert Heuristic Tuning Plan
Create a plan/config for velocity-dependent cp_gain tuning. The current `raibert_foot_target` already implements the heuristic but gains may need adaptation.

### 5. Adaptive Gait Timing
Implement velocity-dependent gait parameter adaptation: decrease stance duration and increase step frequency at higher speeds. Should be generic, reusable.

### 6. High-Frequency Inner Loop
Add a higher-frequency PD loop between MPC updates. MPC runs at 50Hz, inner PD at 500-1000Hz within substeps.

## Verification

After EACH improvement, run:
```
cargo run -p clankers-examples --bin quadruped_mpc_bench -- --velocity 0.5 --gait trot
```
Robot must still walk at least as well as before (~4.3m final X, stable).

## Key Files
- `crates/clankers-mpc/src/swing.rs` — swing trajectory
- `crates/clankers-physics/src/rapier/systems.rs` — rapier step system
- `crates/clankers-viz/src/` — viz plugin/systems
- `examples/src/mpc_control.rs` — shared MPC control logic
- `examples/src/bin/quadruped_mpc_viz.rs` — viz binary
- `examples/src/bin/quadruped_mpc_bench.rs` — bench binary
