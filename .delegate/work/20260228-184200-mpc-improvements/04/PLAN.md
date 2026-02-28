# Loop 4: Adaptive Gait Timing Wiring

## Goal
Wire existing AdaptiveGaitConfig into bench and viz binaries. Opt-in via `--adaptive-gait` CLI flag.

## Changes
- `examples/src/bin/quadruped_mpc_bench.rs`: Add `--adaptive-gait` flag, wire to MpcLoopState
- `examples/src/bin/quadruped_mpc_viz.rs`: Add `--adaptive-gait` flag, wire to MpcLoopState
