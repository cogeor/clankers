# Loop 02: Combo Experiments

## Best Config (horizon=10, real-time compatible)
| Param | Old | New | Change |
|-------|-----|-----|--------|
| q_weights[5] (pz) | 50.0 | 5.0 | 10x lower |
| q_weights[9,10] (vx,vy) | 20.0 | 150.0 | 7.5x higher |
| r_weight | 1e-6 | 1e-7 | 10x lower |

**Result: 1.30m avg, 0.108 m/s (2.2x baseline), 4.5 deg roll, 3.7ms solve**

## Best Config (horizon=20, offline/benchmark only)
| Param | Value | Result |
|-------|-------|--------|
| q_pz=5, q_vx=200, h=20, r=1e-7 | K3 | 1.495m, 0.125 m/s |
| q_pz=5, q_vx=150, h=20, r=1e-8 | K2 | 1.444m, 0.120 m/s |
| q_pz=3, q_vx=300, h=20, r=1e-8 | I4/J6 | 1.35-1.68m, 0.11-0.14 m/s |

## Key Insights
- **r_weight is the hidden lever**: 1e-6→1e-7 gives ~20% speed boost
- **q_vx=150 with pz=5 is the sweet spot** for h=10
- **horizon=20 provides more stability headroom** for aggressive weights
- Run-to-run variance exists due to chaotic dynamics (~±10%)
