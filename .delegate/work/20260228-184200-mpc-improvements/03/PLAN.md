# Loop 3: Torque Rate Clamping

## Goal
Add motor command rate limiting at actuator output, NOT inside MPC QP.

## Changes
- `crates/clankers-physics/src/rapier/systems.rs`: Add `MotorRateLimits` resource, apply clamping in `rapier_step_system`
- `crates/clankers-physics/src/rapier/mod.rs`: Export `MotorRateLimits`
- `examples/src/quadruped_setup.rs`: Add `motor_rate_limit` config option
