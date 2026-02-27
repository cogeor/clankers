# TASK: Aggressive Velocity Tuning

## Goal
Get the quadruped significantly closer to the 0.3 m/s target velocity (currently ~0.06 m/s effective, target is 0.3 m/s).

## Current State
- Baseline: Final X=0.605m in 10s → ~0.06 m/s (20% of target)
- Best single experiment: --q-pz 20 --q-vx 40 → 0.721m → ~0.07 m/s
- StanceConfig and CLI experiment flags are implemented
- Swing IK fix was tested and rejected (destabilizes)

## Available CLI Flags
```
--q-vx <f64>         Override q_weights[9,10] (vx,vy), default 20
--q-pz <f64>         Override q_weights[5] (pz height), default 50
--r-weight <f64>     Override r_weight, default 1e-6
--horizon <usize>    Override horizon, default 10
--mu <f64>           Override friction_coeff, default 0.4
--f-max <f64>        Override f_max per foot, default 120
--stance-kp <f32>    Override stance pitch_knee_kp, default 5.0
--stance-max-f <f32> Override stance pitch_knee_max_f, default 50
--raibert-kv <f64>   Override raibert_kv, default 0.15
```

## Strategy
1. Systematic parameter sweeps using the benchmark CLI
2. Combine winners from each sweep
3. Apply best parameters as new defaults
4. Verify stability (z > 0.15, roll < 15 deg)

## Success Criteria
- Effective velocity >= 0.15 m/s (50% of target) — stretch goal 0.2 m/s
- Robot stays upright (z > 0.15m, max roll < 15 deg)
- Tests still pass
