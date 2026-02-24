# Loop 01: Implementation

## Files Created
- `examples/urdf/quadruped.urdf` — 258 lines, quadruped robot URDF

## Files Modified
- `examples/src/lib.rs` — added `QUADRUPED_URDF` constant

## URDF Design
- Body: 0.4×0.2×0.1m box, 5kg, with correct box inertia tensor
- 4 legs (FL, FR, RL, RR) at body corners: (±0.15, ±0.08, -0.05)
- Per leg: hip_pitch (Y axis, ±1.0 rad) + knee_pitch (Y axis, -2.5..0 rad)
- Leg segments: 0.15m each (upper_leg 0.5kg, lower_leg 0.3kg, foot 0.1kg)
- Fixed foot joints for IK chain end-effectors
- Standing height: body center at ~0.35m above ground
- Total robot mass: 5.0 + 4×(0.5+0.3+0.1) = 8.6kg

## Build
- `cargo test -p clankers-examples --lib` — PASS (compiles with include_str!)
