# Loop 5: High-Frequency Inner PD Loop

## Goal
Interpolate motor target positions across physics substeps for 1000Hz effective PD rate.

## Changes
- `crates/clankers-physics/src/rapier/systems.rs`: Add InnerPdState, interpolate targets in substep loop
- `crates/clankers-physics/src/rapier/mod.rs`: Export InnerPdState
- `examples/src/quadruped_setup.rs`: Add `inner_pd: bool` config option
