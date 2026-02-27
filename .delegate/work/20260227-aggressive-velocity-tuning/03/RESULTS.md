# Loop 03: Extended Sweep + Apply Reliable Defaults

## Methodology
- Each config tested with **5 runs** (not 3) to catch non-deterministic instability
- Discovered that 3-run tests were misleading — configs that appeared 3/3 stable
  could fail 3/5 or 4/5 on extended testing
- Total configs tested in this loop: ~40 across G, H, I, K series

## Key Findings

### Non-determinism is pervasive
- Configs at the stability boundary (q_vx=200, q_pz=10, r=1e-8) show
  1-4 failures out of 5 runs
- Root cause: chaotic physics dynamics in Rapier (HashMap iteration order,
  floating-point non-determinism, contact detection sensitivity)

### Stability boundary map (5-run tested)
| Config | Stable? | Speed (m/s) | Max Roll |
|--------|---------|-------------|----------|
| Original (q_vx=20, q_pz=50, r=1e-6) | 5/5 | 0.050-0.051 | 3.3 deg |
| K1: q_vx=100, q_pz=20, r=1e-7 | 5/5 | 0.076-0.090 | 4.7 deg |
| K2: q_vx=120, q_pz=20, r=1e-7 | 5/5 | 0.087-0.097 | 5.0 deg |
| K4: q_vx=100, q_pz=15, r=1e-7 | 5/5 | 0.078-0.093 | 3.5 deg |
| K5: q_vx=100, q_pz=10, r=1e-7 | 5/5 | 0.082-0.086 | 3.5 deg |
| **K3: q_vx=150, q_pz=20, r=1e-7** | **5/5** | **0.101-0.115** | **7.1 deg** |
| K10: q_vx=150, q_pz=20, r=5e-8 | 5/5 | 0.097-0.122 | 6.6 deg |
| K11: q_vx=170, q_pz=20, r=1e-7, q_roll=40 | 5/5 | 0.092-0.110 | 7.3 deg |
| K13: q_vx=170, q_pz=20, r=5e-8, q_roll=40 | 5/5 | 0.104-0.113 | 10.5 deg |
| K6: q_vx=180, q_pz=20, r=1e-7 | 4/5 | - | TIPPED |
| K7: q_vx=200, q_pz=20, r=1e-7 | 4/5 | - | TIPPED |
| G8: q_vx=200, q_pz=10, r=1e-8 | 1/5 | - | TIPPED |
| G3: q_vx=150, q_pz=10, r=1e-7 | 2/5 | - | TIPPED |

### Other explored dimensions (all destabilize or don't help)
- **Gait cycle time**: 0.25s and 0.45s both worse than 0.35s default
- **Duty factor**: 0.6 destabilizes
- **Step height**: 0.05m doesn't help
- **MPC friction mu**: 0.6 marginal, 0.8 tips
- **f_max=200**: destabilizes with aggressive weights
- **Stance kp**: 0=tips, 2=tips, 5=optimal, 10=stable but 2x slower, 20=very slow
- **Raibert kv**: 0.15 optimal, 0.25+ destabilizes
- **q_omega (angular vel damping)**: no improvement
- **horizon=20**: stable but 20ms solve time, no speed improvement

### Critical insight: q_pz=20 is the key stabilizer
- q_pz=5 (previous default): allows tipping at q_vx>=150
- q_pz=10: marginal stability
- q_pz=20: robust stability for q_vx up to 150
- q_pz=25: slightly worse (takes too much force budget)

## Applied Config (K3)
| Param | Original | New | Change |
|-------|----------|-----|--------|
| q_weights[5] (pz) | 50 | 20 | 2.5x lower |
| q_weights[9,10] (vx,vy) | 20 | 150 | 7.5x higher |
| r_weight | 1e-6 | 1e-7 | 10x lower |

**Validated 5/5 runs: 0.093-0.115 m/s (avg 0.108), max roll 6.6 deg**

## Speed Achievement
- Baseline: 0.051 m/s (17% of 0.3 target)
- New default: 0.108 m/s avg (36% of target)
- **2.1x improvement** while maintaining 100% reliability
