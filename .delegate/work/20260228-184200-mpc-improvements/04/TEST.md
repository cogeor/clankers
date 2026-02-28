# Loop 4 Test: Adaptive Gait Timing Wiring

## Build
- `cargo build -j 24 -p clankers-examples` ✓

## Bench Verification (default, no adaptive gait)
- Trot: Final X=+4.285m ✓ (same as before — opt-in flag not active)

## Bench with adaptive gait
- Trot: Final X=+1.278m — default params too aggressive, needs tuning per-robot.
  This is expected and documented.

## Ready for Commit: yes
