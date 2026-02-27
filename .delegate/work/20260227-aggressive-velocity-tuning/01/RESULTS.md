# Loop 01: Parameter Sweep Results

## Baseline
| Config | Final X | Avg Speed | Min Z | Max Roll |
|--------|---------|-----------|-------|----------|
| Default | 0.605m | 0.050 m/s | 0.213 | 3.3 deg |

## A: Q-Weight Sweeps
| Config | Final X | Avg Speed | Min Z | Max Roll | Status |
|--------|---------|-----------|-------|----------|--------|
| q-pz=30 | 0.625m | 0.052 | 0.214 | 3.3 | stable |
| q-pz=20 | 0.641m | 0.053 | 0.215 | 3.2 | stable |
| q-pz=10 | 0.661m | 0.055 | 0.216 | 3.1 | stable |
| q-vx=40 | 0.657m | 0.055 | 0.210 | 3.4 | stable |
| q-vx=60 | 0.736m | 0.061 | 0.207 | 3.4 | stable |
| q-vx=80 | 0.804m | 0.067 | 0.205 | 3.5 | stable |
| q-vx=100 | 0.839m | 0.070 | 0.204 | 3.5 | stable |
| **q-vx=150** | **1.139m** | **0.095** | 0.202 | 3.6 | **stable** |
| q-vx=200 | 1.134m | 0.095 | 0.104 | 91.0 | TIPPED |
| q-pz=10 q-vx=100 | 1.006m | 0.084 | 0.206 | 3.4 | stable |
| **q-pz=5 q-vx=150** | **1.186m** | **0.099** | 0.204 | 3.5 | **stable** |
| q-pz=5 q-vx=200 | 1.036m | 0.086 | 0.101 | 93.7 | TIPPED |

**Winner: q-pz=5, q-vx=150** (1.186m, 2x baseline)

## B: Horizon (with best Q)
| Config | Final X | Avg Speed | Min Z | Max Roll | Status |
|--------|---------|-----------|-------|----------|--------|
| horizon=15 | 0.377m | 0.031 | 0.100 | 90.4 | TIPPED |
| **horizon=20** | **1.472m** | **0.123** | 0.204 | 6.0 | **stable** |

**Winner: horizon=20** (1.472m, 2.4x baseline)

## C: Friction/Force (with best Q)
| Config | Final X | Avg Speed | Min Z | Max Roll | Status |
|--------|---------|-----------|-------|----------|--------|
| mu=0.6 | 1.166m | 0.097 | 0.203 | 5.5 | stable |
| mu=0.8 | 0.242m | 0.020 | 0.100 | 99.1 | TIPPED |
| f-max=200 | 1.284m | 0.107 | 0.204 | 6.5 | stable |
| mu=0.6 f-max=200 | 1.235m | 0.103 | 0.204 | 9.3 | stable |

**Winner: f-max=200** (marginal improvement, mu=0.6 slight regression)

## D: Stance Gains (with best Q)
| Config | Final X | Avg Speed | Min Z | Max Roll | Status |
|--------|---------|-----------|-------|----------|--------|
| stance-kp=1.0 | 0.392m | 0.033 | 0.100 | 103.0 | TIPPED |
| stance-kp=0.0 | 0.616m | 0.051 | 0.050 | 180.0 | TIPPED |
| stance-max-f=100 | 0.674m | 0.056 | 0.100 | 114.4 | TIPPED |
| stance-kp=1 max-f=100 | 0.259m | 0.022 | 0.100 | 90.4 | TIPPED |

**Winner: defaults (kp=5, max_f=50)** — all changes destabilize

## E: Raibert Gain (with best Q)
| Config | Final X | Avg Speed | Min Z | Max Roll | Status |
|--------|---------|-----------|-------|----------|--------|
| kv=0.25 | 1.137m | 0.095 | 0.204 | 4.8 | stable |
| kv=0.35 | 0.850m | 0.071 | 0.103 | 102.2 | TIPPED |
| kv=0.50 | 0.195m | 0.016 | 0.100 | 91.3 | TIPPED |

**Winner: default kv=0.15** — higher values destabilize with aggressive Q-weights

## Key Findings
1. **q_vx is the primary velocity lever** — monotonically improves until instability at 200
2. **Lowering q_pz frees force budget** for velocity tracking
3. **Horizon=20 is the biggest single improvement** when combined with q_pz=5/q_vx=150
4. **Stance gains and Raibert kv should stay at defaults** — the system is already near stability limits with aggressive Q-weights
