# MPC Quadruped Gait Diagnosis Report

## Executive Conclusion
The current walk/trot stumbling and low foot clearance are most consistent with a **control tuning/integration issue**, not a **fundamental convex MPC solver failure**.

This conclusion is supported by passing solver and gait tests, plus runtime evidence showing repeated MPC convergence while locomotion quality remains poor.

## Evidence Summary

### 1. Solver and gait components are functioning at baseline
- `clankers-mpc` tests passed (`41 passed, 0 failed`).
- `examples/tests/mpc_walk.rs` locomotion tests passed (`7 passed, 0 failed`, 2 ignored).
- In headless runtime (`quadruped_mpc`), MPC repeatedly reports `OK` convergence with physically plausible force outputs.

This does not prove gait quality is good, but it does argue against a hard solver formulation bug.

### 2. Swing trajectory generation is not the same as the QP solver
- Foot arc generation (`swing_foot_position`, `step_height = 0.10`) comes from the swing planner in `clankers-mpc/src/swing.rs`.
- The QP solver computes stance/contact forces.

So statements like "the MPC solver computes the foot parabola" are technically inaccurate.

### 3. Low-level control allocation likely under-delivers commanded swing lift
- The runtime control in both `quadruped_mpc.rs` and `quadruped_mpc_viz.rs` uses mixed motor settings that can conflict:
  - A fixed joint target position (`q0`) is maintained.
  - Cartesian swing behavior is injected via velocity/force-like terms.
  - Gain and max-force schedules differ across joints and phases.
- This is a plausible source of feet dragging/stumbling even when swing trajectories request higher clearance.

### 4. Reported single-cause certainty was too strong
- The previous report claimed one dominant mechanism (joint spring overpowering swing) as definitive.
- That mechanism is plausible, but current evidence supports a broader diagnosis: **controller-interface and gain/timing coupling**, not a single isolated constant.

## Scope Clarification
- This issue is not isolated to `quadruped_mpc_viz.rs`; the same control pattern appears in `quadruped_mpc.rs`.
- No code changes were made in this analysis; this is diagnosis only.

## Final Assessment
**Agree with the high-level conclusion:** tune/integration issue, not fundamental solver issue.  
**Refined interpretation:** the root cause is likely multi-factor in the low-level control layer (motor command mapping, phase transitions, and gain/force scheduling), rather than a single parameter alone.
