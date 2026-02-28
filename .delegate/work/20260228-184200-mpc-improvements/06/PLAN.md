# Loop 6: Raibert Heuristic Tuning Config

## Goal
Add velocity-dependent cp_gain profile to SwingConfig so foot placement gain adapts with speed.

## Changes
- `crates/clankers-mpc/src/swing.rs`: Add CpGainProfile struct with lookup(), effective_cp_gain() on SwingConfig, unit tests
- `examples/src/mpc_control.rs`: Wire effective_cp_gain(body_speed) into compute_mpc_step
- `apps/clankers-app/src/main.rs`: Fix ClankersVizPlugin unit struct usage
