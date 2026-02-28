# Loop 3 Implementation: Torque Rate Clamping

## `crates/clankers-physics/src/rapier/systems.rs`
- Added `MotorRateLimits` resource with `delta_max: f32` and `prev_targets: HashMap<Entity, f32>`
- Modified `rapier_step_system` to accept optional `ResMut<MotorRateLimits>`
- When present, clamps motor target_pos: `target = clamp(target, prev - delta_max, prev + delta_max)`
- Previous targets tracked per-entity for smooth limiting

## `examples/src/quadruped_setup.rs`
- Added `motor_rate_limit: Option<f32>` to `QuadrupedSetupConfig`
- When `Some`, inserts `MotorRateLimits::new(delta_max)` resource
- Default: `None` (no rate limiting, backward compatible)
