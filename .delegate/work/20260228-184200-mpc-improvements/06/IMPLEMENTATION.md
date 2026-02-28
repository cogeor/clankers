# Loop 6 Implementation: Raibert Heuristic Tuning Config

## `crates/clankers-mpc/src/swing.rs`
- Added `CpGainProfile` struct with `breakpoints: Vec<(f64, f64)>` for velocity-dependent gain lookup
- `CpGainProfile::lookup(speed)`: linear interpolation between breakpoints, clamps outside range, fallback 0.5 for empty
- `CpGainProfile::default()`: constant 0.5 at all speeds (matches original behavior)
- Added `cp_gain_profile: Option<CpGainProfile>` to `SwingConfig`
- Added `SwingConfig::effective_cp_gain(speed)`: uses profile if set, else returns constant `cp_gain`
- Added tuning guidelines documentation on CpGainProfile
- Added 7 unit tests: default constant, interpolation, clamping, empty fallback, effective_cp_gain with/without profile

## `examples/src/mpc_control.rs`
- Changed `compute_mpc_step` to compute `body_speed = body_state.linear_velocity.xy().norm()`
- Replaced `state.swing_config.cp_gain` with `state.swing_config.effective_cp_gain(body_speed)` in raibert_foot_target call

## `apps/clankers-app/src/main.rs`
- Fixed `ClankersVizPlugin` → `ClankersVizPlugin::default()` (backward compat from loop 2 struct change)
