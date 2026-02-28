# Loop 4 Implementation: Adaptive Gait Timing Wiring

## Changes

### `examples/src/bin/quadruped_mpc_bench.rs`
- Added `--adaptive-gait` CLI flag
- When enabled, sets `adaptive_gait: Some(AdaptiveGaitConfig::default())`
- Added `AdaptiveGaitConfig` import

### `examples/src/bin/quadruped_mpc_viz.rs`
- Added `--adaptive-gait` CLI flag
- When enabled, sets `adaptive_gait: Some(AdaptiveGaitConfig::default())`
- Added `AdaptiveGaitConfig` import

## Notes
- Default AdaptiveGaitConfig needs tuning for this robot — at 0.5 m/s it changes
  cycle_time from 0.35s to 0.50s which destabilizes. Users should tune l_max,
  t_cycle_max, and duty_speed_slope for their specific robot.
- The flag is opt-in so default behavior is preserved.
