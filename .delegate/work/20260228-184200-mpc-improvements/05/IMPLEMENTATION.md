# Loop 5 Implementation: High-Frequency Inner PD Loop

## `crates/clankers-physics/src/rapier/systems.rs`
- Added `InnerPdState` resource: stores previous control step target positions
- Modified `rapier_step_system`:
  - When InnerPdState + MotorOverrides present, collects override entries
  - Linearly interpolates target_pos across substeps: `interp = prev + (target - prev) * (sub+1)/substeps`
  - Each substep updates motor targets before stepping physics
  - Effective PD rate: 20 substeps * 50Hz = 1000Hz
- Without InnerPdState, behavior unchanged (ZOH, all substeps with same target)

## `examples/src/quadruped_setup.rs`
- Added `inner_pd: bool` to config (default false)
- When true, inserts `InnerPdState::default()` resource
