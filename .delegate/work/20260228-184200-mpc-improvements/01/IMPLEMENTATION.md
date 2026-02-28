# Loop 1 Implementation: Bezier Swing Trajectories

## Changes to `crates/clankers-mpc/src/swing.rs`

### Added
- `BEZIER_S` constant: 12-point control array for horizontal S-curve interpolation `[0,0,0, 0.5,0.5, 0.5,0.5, 0.5,0.5, 1,1,1]`
- `BEZIER_H` constant: 12-point control array for height profile `[0,0,0, 0.9,0.9, 1.0,1.0, 0.9,0.9, 0,0,0]`
- `BEZIER_H_PEAK` constant: pre-computed peak value (0.886230468750) for normalization
- `bezier_eval()`: De Casteljau's algorithm for degree-11 Bezier evaluation
- `bezier_derivative()`: hodograph-based derivative of degree-11 Bezier

### Changed
- `swing_foot_position`: now uses `bezier_eval(&BEZIER_S, t)` for horizontal interpolation and `bezier_eval(&BEZIER_H, t) * step_height / BEZIER_H_PEAK` for height profile
- `swing_foot_velocity`: now uses `bezier_derivative(&BEZIER_S/H, t)` for derivatives
- Updated test `swing_velocity_nonzero_at_midpoint` threshold from 0.5 to 0.1 (Bezier S-curve has broader acceleration profile vs min-jerk's sharp peak)

### Properties preserved
- Zero position error at endpoints (t=0, t=1)
- Zero velocity at endpoints
- Zero acceleration at endpoints (new: min-jerk only had zero vel)
- Peak height = step_height at t=0.5
- Symmetric height profile
- Zero vertical velocity at midpoint
