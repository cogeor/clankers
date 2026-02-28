# Quadruped MPC Walk Cycle — Complete Technical Report

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Robot Specifications](#2-robot-specifications)
3. [Gait Scheduler](#3-gait-scheduler)
4. [Reference Trajectory](#4-reference-trajectory)
5. [Centroidal Dynamics Model](#5-centroidal-dynamics-model)
6. [Condensed QP Formulation](#6-condensed-qp-formulation)
7. [Stance Leg Control](#7-stance-leg-control)
8. [Swing Leg Control](#8-swing-leg-control)
9. [Physics Engine Settings](#9-physics-engine-settings)
10. [Per-Tick Data Flow](#10-per-tick-data-flow)
11. [Performance Results](#11-performance-results)
12. [Tuning Insights](#12-tuning-insights)
13. [File Reference](#13-file-reference)

---

## 1. System Architecture

The pipeline follows the **MIT Cheetah convex MPC** approach (Di Carlo et al., IROS 2018):

```
  FK (foot positions)
        |
  Gait Scheduler ──> contact_sequence (horizon x 4)
        |
  Reference Trajectory (constant velocity)
        |
  Centroidal MPC (Clarabel QP)
        |
        +──── Stance Legs ──> J^T(-f_mpc) + IK ──> Motor Commands
        |
        +──── Swing Legs ──> Raibert + Min-Jerk + Cartesian PD ──> J^T(F) ──> Motor Commands
        |
  12 MotorCommands -> Actuators
```

**Key files:**

| Component | File |
|-----------|------|
| Shared control pipeline | `examples/src/mpc_control.rs` |
| MPC types & config | `crates/clankers-mpc/src/types.rs` |
| Gait scheduler | `crates/clankers-mpc/src/gait.rs` |
| Centroidal dynamics | `crates/clankers-mpc/src/centroidal.rs` |
| QP solver | `crates/clankers-mpc/src/solver.rs` |
| Swing planner | `crates/clankers-mpc/src/swing.rs` |
| Whole-body control (J^T) | `crates/clankers-mpc/src/wbc.rs` |
| IK solver (DLS) | `crates/clankers-ik/src/solver.rs` |
| Headless example | `examples/src/bin/quadruped_mpc.rs` |
| Viz example | `examples/src/bin/quadruped_mpc_viz.rs` |
| Benchmark example | `examples/src/bin/quadruped_mpc_bench.rs` |
| URDF model | `examples/urdf/quadruped.urdf` |

---

## 2. Robot Specifications

### 2.1 Mass Budget

| Component | Mass (kg) | Count | Subtotal (kg) |
|-----------|-----------|-------|----------------|
| Body | 5.0 | 1 | 5.0 |
| Hip link | 0.1 | 4 | 0.4 |
| Upper leg | 0.5 | 4 | 2.0 |
| Lower leg | 0.3 | 4 | 1.2 |
| Foot | 0.1 | 4 | 0.4 |
| **Total** | | | **9.0** |

### 2.2 Body

| Parameter | Value | Unit |
|-----------|-------|------|
| Dimensions (box) | 0.4 x 0.2 x 0.1 | m |
| Mass | 5.0 | kg |
| Ixx | 0.02083 | kg·m² |
| Iyy | 0.07083 | kg·m² |
| Izz | 0.08333 | kg·m² |
| Collider (cuboid half-extents) | 0.2 x 0.1 x 0.05 | m |
| Collider friction | 0.5 | — |

### 2.3 Hip Links (all 4 identical)

| Parameter | Value | Unit |
|-----------|-------|------|
| Dimensions (box) | 0.04 x 0.04 x 0.04 | m |
| Mass | 0.1 | kg |
| Ixx / Iyy / Izz | 0.00003 | kg·m² |
| Collider (cuboid half-extents) | 0.02 x 0.02 x 0.02 | m |
| Collider friction | 0.3 | — |

### 2.4 Upper Legs (all 4 identical)

| Parameter | Value | Unit |
|-----------|-------|------|
| Geometry | cylinder | — |
| Radius | 0.015 | m |
| Length | 0.15 | m |
| Mass | 0.5 | kg |
| CoM offset from parent | (0, 0, -0.075) | m |
| Ixx | 0.001 | kg·m² |
| Iyy | 0.001 | kg·m² |
| Izz | 0.0002 | kg·m² |
| Collider (capsule_z) | r=0.015, half-len=0.075 | m |
| Collider friction | 0.3 | — |
| Collider restitution | 0.0 | — |

### 2.5 Lower Legs (all 4 identical)

| Parameter | Value | Unit |
|-----------|-------|------|
| Geometry | cylinder | — |
| Radius | 0.012 | m |
| Length | 0.15 | m |
| Mass | 0.3 | kg |
| CoM offset from parent | (0, 0, -0.075) | m |
| Ixx | 0.0006 | kg·m² |
| Iyy | 0.0006 | kg·m² |
| Izz | 0.0001 | kg·m² |
| Collider (capsule_z) | r=0.012, half-len=0.075 | m |
| Collider friction | 0.3 | — |
| Collider restitution | 0.0 | — |

### 2.6 Feet (all 4 identical)

| Parameter | Value | Unit |
|-----------|-------|------|
| Geometry | sphere | — |
| Radius | 0.02 | m |
| Mass | 0.1 | kg |
| Ixx / Iyy / Izz | 0.00001 | kg·m² |
| Collider friction | 1.0 | — |
| Collider restitution | 0.0 | — |

### 2.7 Hip Offsets (body-frame, from body CoM)

| Leg | X (m) | Y (m) | Z (m) |
|-----|--------|--------|--------|
| Front Left (FL) | +0.15 | +0.08 | -0.05 |
| Front Right (FR) | +0.15 | -0.08 | -0.05 |
| Rear Left (RL) | -0.15 | +0.08 | -0.05 |
| Rear Right (RR) | -0.15 | -0.08 | -0.05 |

### 2.8 Joint Specifications

Each leg has 3 revolute joints:

**Hip Abduction (X-axis rotation):**

| Parameter | Value | Unit |
|-----------|-------|------|
| Axis | X (1, 0, 0) | — |
| Position limits | -0.5 to +0.5 | rad |
| Effort limit | 20 | Nm |
| Velocity limit | 10 | rad/s |
| Damping | 0.1 | Nm·s/rad |
| Friction | 0.05 | — |

**Hip Pitch (Y-axis rotation):**

| Parameter | Value | Unit |
|-----------|-------|------|
| Axis | Y (0, 1, 0) | — |
| Position limits | -1.0 to +1.0 | rad |
| Effort limit | 30 | Nm |
| Velocity limit | 10 | rad/s |
| Damping | 0.1 | Nm·s/rad |
| Friction | 0.05 | — |

**Knee Pitch (Y-axis rotation):**

| Parameter | Value | Unit |
|-----------|-------|------|
| Axis | Y (0, 1, 0) | — |
| Position limits | -2.5 to 0.0 | rad |
| Effort limit | 30 | Nm |
| Velocity limit | 10 | rad/s |
| Damping | 0.1 | Nm·s/rad |
| Friction | 0.05 | — |

### 2.9 Kinematic Chain Dimensions

| Segment | Length (m) |
|---------|-----------|
| Upper leg | 0.15 |
| Lower leg | 0.15 |
| Total leg reach | ~0.30 |

### 2.10 Standing Configuration (initial joint angles)

| Joint | Angle (rad) | Angle (deg) |
|-------|-------------|-------------|
| Hip abduction | 0.0 | 0 |
| Hip pitch | 1.05 | ~60 |
| Knee pitch | -2.10 | ~-120 |

### 2.11 Warmup Motor Control

| Parameter | Value | Unit |
|-----------|-------|------|
| Warmup steps | 1000 | — |
| Stiffness (kp) | 500.0 | Nm/rad |
| Damping (kd) | 50.0 | Nm·s/rad |
| Max force | 100.0 | N |

---

## 3. Gait Scheduler

**Source:** `crates/clankers-mpc/src/gait.rs`

### 3.1 Algorithm

Each foot has a phase offset in [0, 1). A global phase advances each tick:

```
phase = (phase + dt / cycle_time) % 1.0
foot_phase = (phase + offset[foot]) % 1.0
```

- **Stance:** `foot_phase < duty_factor`
- **Swing:** `foot_phase >= duty_factor`
- **Swing progress:** `(foot_phase - duty_factor) / (1 - duty_factor)` normalized to [0, 1]

The scheduler generates a `horizon x n_feet` contact matrix for the MPC via `contact_sequence(horizon, dt)`.

### 3.2 Gait Presets

| Gait | Foot Offsets [FL, FR, RL, RR] | Duty Factor | Cycle Time (s) | Description |
|------|-------------------------------|-------------|-----------------|-------------|
| Stand | [0.0, 0.0, 0.0, 0.0] | 1.0 | 1.0 | All feet always in stance |
| Trot | [0.0, 0.5, 0.5, 0.0] | 0.5 | 0.35 | Diagonal pairs alternate (FL+RR / FR+RL) |
| Walk | [0.0, 0.5, 0.25, 0.75] | 0.75 | 0.8 | One foot swings at a time, 3 in stance |
| Bound | [0.0, 0.0, 0.5, 0.5] | 0.5 | 0.4 | Front pair / rear pair alternate |

### 3.3 Trot Timing Detail

| Parameter | Value |
|-----------|-------|
| Stance duration | 0.5 x 0.35 = 0.175 s |
| Swing duration | 0.5 x 0.35 = 0.175 s |
| Phase pair 1 (FL+RR) | phase 0.0 – 0.5: stance; 0.5 – 1.0: swing |
| Phase pair 2 (FR+RL) | phase 0.5 – 1.0: stance; 0.0 – 0.5: swing |

---

## 4. Reference Trajectory

**Source:** `crates/clankers-mpc/src/types.rs`

A constant-velocity reference over the MPC horizon. For each horizon step k at time t = k * dt:

| State | Reference |
|-------|-----------|
| Roll, Pitch | 0.0 |
| Yaw | desired_yaw (constant) |
| px | x₀ + vx_des * t |
| py | y₀ + vy_des * t |
| pz | desired_height |
| ωx, ωy, ωz | 0.0 |
| vx, vy | desired velocity |
| vz | 0.0 |
| g | 9.81 |

### Velocity Ramp

During startup, the desired velocity ramps linearly over a configurable number of steps (default: 100 steps at the control rate).

---

## 5. Centroidal Dynamics Model

**Source:** `crates/clankers-mpc/src/centroidal.rs`

### 5.1 State Vector (13-dimensional)

```
x = [θ_roll, θ_pitch, θ_yaw,   (Euler angles, rad)
     p_x, p_y, p_z,             (CoM position, m)
     ω_x, ω_y, ω_z,            (angular velocity, rad/s)
     v_x, v_y, v_z,            (linear velocity, m/s)
     g]                         (gravity constant, m/s²)
```

### 5.2 Continuous-Time Dynamics

**Euler angle kinematics** (yaw-only rotation for convexity):

```
θ̇ = R_z(yaw)⁻¹ · ω
```

**Position:**

```
ṗ = v
```

**Angular acceleration:**

```
ω̇ = I_world⁻¹ · Σᵢ (rᵢ × fᵢ)
```

where `rᵢ = foot_posᵢ - CoM` and `I_world = R_z(yaw) · I_body · R_z(yaw)ᵀ`.

**Linear acceleration:**

```
v̇ = (1/m) · Σᵢ fᵢ + [0, 0, -g]
```

**Gravity:**

```
ġ = 0
```

### 5.3 MPC Dynamics Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Total mass | 9.0 | kg |
| Body-frame Ixx | 0.048 | kg·m² |
| Body-frame Iyy | 0.122 | kg·m² |
| Body-frame Izz | 0.135 | kg·m² |
| Gravity | 9.81 | m/s² |
| State dimension | 13 | — |

### 5.4 Discretization

**Method:** Matrix exponential (exact for LTI systems)

```
[A_d  B_d] = expm(dt · [A_c  B_c])
[ 0    I ]              [ 0    0 ]
```

The continuous A and B matrices are rebuilt at each solve using the current yaw angle and foot positions (relinearization).

---

## 6. Condensed QP Formulation

**Source:** `crates/clankers-mpc/src/solver.rs`

### 6.1 Decision Variables

```
U = [f₁, f₂, ..., f_H]    where fᵢ ∈ ℝ¹² (3D force × 4 feet)
```

Total: 12 × H variables (120 for H=10).

### 6.2 State Prediction (single-shooting condensation)

```
X = A_qp · x₀ + B_qp · U
```

- `A_qp` : (13H × 13) — powers of A_d stacked
- `B_qp` : (13H × 12H) — lower block triangular

### 6.3 Cost Function

```
J = ½ Uᵀ H U + gᵀ U

H = 2(B_qpᵀ S B_qp + α I)
g = 2 B_qpᵀ S (A_qp x₀ - X_ref)

S = blkdiag(Q, Q, ..., Q)   (H copies)
Q = diag(q_weights)
α = r_weight
```

### 6.4 Cost Weights

| Index | State | Weight | Role |
|-------|-------|--------|------|
| 0 | Roll | 25.0 | Orientation stability |
| 1 | Pitch | 25.0 | Orientation stability |
| 2 | Yaw | 10.0 | Heading tracking |
| 3 | px | 5.0 | Lateral position |
| 4 | py | 5.0 | Lateral position |
| 5 | pz | 10.0 | Height regulation |
| 6 | ωx | 1.0 | Angular rate damping |
| 7 | ωy | 1.0 | Angular rate damping |
| 8 | ωz | 0.3 | Yaw rate (low) |
| 9 | vx | 150.0 | Forward velocity (saturated) |
| 10 | vy | 150.0 | Lateral velocity |
| 11 | vz | 5.0 | Vertical velocity |

| Parameter | Value |
|-----------|-------|
| r_weight (force effort) | 1e-7 |

### 6.5 Constraints

**Swing feet (equality — ZeroCone):**

```
f_swing = 0     (3 constraints per swing foot per timestep)
```

**Friction pyramid (inequality — NonnegativeCone, 6 per stance foot per timestep):**

```
μ·fz - fx  ≥ 0
μ·fz + fx  ≥ 0
μ·fz - fy  ≥ 0
μ·fz + fy  ≥ 0
fz         ≥ 0
f_max - fz ≥ 0
```

### 6.6 Constraint Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Friction coefficient (μ) | 0.4 | — |
| Max normal force (f_max) | 120.0 | N |

### 6.7 Solver Settings (Clarabel)

| Parameter | Value |
|-----------|-------|
| Max iterations | 100 |
| Absolute gap tolerance | 1e-6 |
| Relative gap tolerance | 1e-6 |
| Feasibility tolerance | 1e-6 |
| Verbose | false |

### 6.8 MPC Horizon Parameters

| Parameter | Default | Unit |
|-----------|---------|------|
| Horizon (H) | 10 | steps |
| Timestep (dt) | 0.02 (50 Hz) or 0.01 (100 Hz) | s |
| Preview window | H × dt = 0.2 s or 0.1 s | s |

---

## 7. Stance Leg Control

**Source:** `examples/src/mpc_control.rs`

### 7.1 Pipeline

```
1. Negate MPC force:    f = -f_mpc  (body frame convention)
2. Compute Jacobian:    J (3×3 for 3-DOF leg)
3. Feedforward torque:  τ_ff = Jᵀ · f
4. IK solve:            q_ik = DLS(chain, foot_world, q_current)
5. Motor commands:      target_pos = q_ik, target_vel = τ_ff / kd
```

### 7.2 Velocity Encoding Trick

The key insight is encoding MPC forces into the motor velocity reference:

```
target_vel = τ_ff / kd
```

This cancels damping losses:

```
motor_torque = kp·(q_ik - q) + kd·(τ_ff/kd - q̇)
             = kp·(q_ik - q) + τ_ff - kd·q̇
             ≈ τ_ff          (since IK makes q_ik ≈ q and q̇ ≈ 0 in stance)
```

### 7.3 Stance Motor Gains

| Joint | kp (Nm/rad) | kd (Nm·s/rad) | max_force (N) |
|-------|-------------|----------------|---------------|
| Hip abduction | 200.0 | 20.0 | 200.0 |
| Hip pitch | 0.5 | 0.1 | 200.0 |
| Knee pitch | 0.5 | 0.1 | 200.0 |

### 7.4 IK Configuration (Damped Least Squares)

| Parameter | Value (stance) | Value (default) |
|-----------|----------------|-----------------|
| Max iterations | 10 | 100 |
| Position tolerance | 1e-3 m | 1e-4 m |
| Damping (λ) | 0.01 | 0.01 |

**DLS update rule:**

```
Δq = Jᵀ (J·Jᵀ + λ²I)⁻¹ · error
```

### 7.5 Jacobian Computation

**Source:** `crates/clankers-mpc/src/wbc.rs`

For revolute joint i:

```
J_col_i = axis_i × (foot_pos - joint_origin_i)
```

Result: 3×3 matrix (3 Cartesian axes, 3 joints per leg).

Torque mapping: `τ = Jᵀ · F`

---

## 8. Swing Leg Control

**Source:** `crates/clankers-mpc/src/swing.rs`, `examples/src/mpc_control.rs`

### 8.1 Raibert Foot Placement

Computed once at swing start (swing_phase < 0.05):

```
hip_at_td = hip_now + v_body · T_swing
sym_offset = 0.5 · v_body · T_stance
vel_correction = kv · (v_body - v_desired)

target = hip_at_td + sym_offset + vel_correction
```

Safety clamp: max 0.3 m from hip center.

### 8.2 Swing Trajectory (Min-Jerk)

**Position interpolation:**

```
s(t) = 10t³ - 15t⁴ + 6t⁵     where t ∈ [0, 1] is normalized swing phase
```

**Horizontal (XY):**

```
p_xy = start + (target - start) · s(t)
```

**Vertical (Z) with parabolic bump:**

```
bump = 64 · t³ · (1-t)³ · step_height     (peaks at t = 0.5)
p_z  = start_z + (target_z - start_z) · s(t) + bump
```

### 8.3 Cartesian PD Control

```
F = Kp · (p_des - p_foot) + Kd · (v_des - v_foot)
```

Where `v_foot` includes body linear velocity, body angular velocity cross-product, and joint-space contribution:

```
v_foot = v_body + ω_body × r_foot + J · q̇
```

### 8.4 Swing Configuration Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Step height | 0.10 | m |
| Default step length | 0.08 | m |
| Kp_cartesian | (500, 500, 500) | N/m |
| Kd_cartesian | (20, 20, 20) | N·s/m |
| Raibert kv | 0.15 | — |
| Max placement distance | 0.3 | m |

### 8.5 Swing Motor Gains

| Joint | kp (Nm/rad) | kd (Nm·s/rad) | max_force (N) |
|-------|-------------|----------------|---------------|
| Hip abduction | 80.0 | 0.5 | 60.0 |
| Hip pitch | 20.0 | 0.5 | 60.0 |
| Knee pitch | 20.0 | 0.5 | 60.0 |

Target position for all swing joints: standing configuration angles (q0).

---

## 9. Physics Engine Settings

**Engine:** Rapier

### 9.1 Integration Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Simulation dt | 0.02 or 0.01 | s |
| Gravity vector | (0, 0, -9.81) | m/s² |
| Solver iterations | **50** (CRITICAL) | — |
| Default solver iterations (Rapier) | 4 | — |

The solver iteration count is critical: 12 revolute joints create 60 locked-DOF constraints. The default of 4 iterations causes slow joint drift leading to robot tipping. Must be set to 50.

### 9.2 Ground Plane

| Parameter | Value | Unit |
|-----------|-------|------|
| Position (z) | -0.05 | m |
| Dimensions (cuboid) | 50 x 50 x 0.05 | m |
| Friction | 1.0 | — |
| Restitution | 0.0 | — |

### 9.3 Motor Model (impedance control)

```
τ = kp · (target_pos - pos) + kd · (target_vel - vel)
```

Clamped to `[-max_force, +max_force]`.

### 9.4 Startup Sequence

| Phase | Steps | Duration (at dt=0.02) | Action |
|-------|-------|-----------------------|--------|
| Warmup | 1000 | 20.0 s | Drive to standing angles with kp=500, kd=50 |
| Stabilization | 100 | 2.0 s | Stand gait (all feet contact) with MPC |
| Velocity ramp | 100 | 2.0 s | Linear ramp from 0 to target velocity |
| Locomotion | configurable | configurable | Full trot at target velocity |

---

## 10. Per-Tick Data Flow

At each control step (50 or 100 Hz):

```
1. Forward Kinematics
   For each leg: chain.forward_kinematics(q) → p_foot_body
   Transform to world: p_foot = R_body · p_foot_body + body_pos
   → foot_world[0..4]

2. Gait Advance
   gait.advance(dt)
   contacts_seq = gait.contact_sequence(horizon, dt)
   For each leg: swing_phase[i] = gait.swing_phase(i)

3. Reference Trajectory
   x_ref = constant_velocity(body_state, v_des, h_des, yaw_des, horizon, dt)

4. MPC Solve
   Build continuous A_c, B_c from current yaw + foot positions
   Discretize → A_d, B_d
   Fill A_qp, B_qp (single-shooting condensation)
   Build cost (H, g) and constraints (A_cone, b_cone)
   Solve QP with Clarabel
   → solution.forces[0..4]

5. Per-Leg Torque Computation
   Compute Jacobian J (3×3)

   IF is_contact(leg) AND solver converged:
     f = -solution.forces[leg]
     τ_ff = Jᵀ · f
     q_ik = IK_solve(chain, foot_world[leg], q_current)
     Motor: target_pos=q_ik, target_vel=τ_ff/kd, kp/kd/max_f per joint type

   ELSE (swing):
     IF swing_phase < 0.05: compute Raibert target
     p_des = swing_position(start, target, swing_phase, step_height)
     v_des = swing_velocity(start, target, swing_phase, ...)
     F_cart = Kp·(p_des-p) + Kd·(v_des-v)
     τ = Jᵀ · F_cart
     Motor: target_pos=q0, target_vel=τ/kd_swing, kp/kd/max_f per joint type

6. Send 12 MotorCommands to actuators
```

---

## 11. Performance Results

### 11.1 Trot at 50 Hz (dt = 0.02 s)

| Target Velocity | Achieved | Stability | Duration |
|-----------------|----------|-----------|----------|
| 0.3 m/s | 0.284 m/s | 5/5 stable | 60 s |
| 0.6 m/s | — | 1/3 stable | borderline |

### 11.2 Trot at 100 Hz (dt = 0.01 s)

| Target Velocity | Achieved | Stability | Duration |
|-----------------|----------|-----------|----------|
| 1.0 m/s | 0.955 m/s | 5/5 stable | 50 s |
| 1.0 m/s | — | tips | ~55 s (slow drift) |

### 11.3 Typical Solve Times

QP solve: ~100–500 μs per tick (Clarabel interior-point).

---

## 12. Tuning Insights

### 12.1 Critical Parameters

| Parameter | Value | Impact |
|-----------|-------|--------|
| Rapier solver iters | 50 | 4 → joint drift → tipping |
| MPC rate | 100 Hz | Essential for >0.5 m/s |
| Stance kd | 0.1 | Lower = less resistance to MPC forces |
| Stance kp | 0.5 | Small; IK nullifies position error |
| Hip ab kp | 200 | High stiffness prevents lateral tip |
| Hip ab kd | 20 | 10 → lateral tipping |
| Swing kd | 0.5 | 0.3 → unstable (0/5), 0.4 → borderline (4/5) |
| r_weight | 1e-7 | Near-zero → aggressive force usage |
| q_vx, q_vy | 150 | Saturated; 200+ gives no improvement |

### 12.2 Design Insights

1. **Velocity encoding (`target_vel = τ_ff / kd`)** converts impedance control into near-pure force feedforward. Without this, motor damping creates resistive losses that bottleneck speed.

2. **Yaw-only inertia rotation** maintains QP convexity while being reasonably accurate for small roll/pitch.

3. **Foot positions as parameters** (not decision variables) keeps the problem linear. Relinearized each solve.

4. **Gravity as state** makes the dynamics matrices truly linear (no explicit gravity term).

5. **Raibert target recomputed only at swing start** (phase < 0.05) to prevent chattering.

6. **Rapier non-determinism** means configs at the stability boundary give inconsistent results. Always validate with 5+ runs.

7. **Lower motor gains = faster locomotion** because kp/kd create damping losses that oppose MPC forces. IK keeps kp·(q_ik-q) ≈ 0, making the position term nearly zero.

8. **kp=0 is unstable** without high hip abduction damping — a small residual kp provides essential position regulation.

### 12.3 Gain Sensitivity

| Change | Speed Effect |
|--------|-------------|
| kp 5.0 → 0.5 | 0.155 → 0.230 m/s |
| kd 1.0 → 0.1 | 0.123 → 0.155 m/s |
| max_f 50 → 200 | 0.105 → 0.123 m/s |
| MPC dt 0.02 → 0.01 | ~0.3 m/s cap → ~1.0 m/s achievable |

---

## 13. File Reference

### 13.1 MPC Crate (`crates/clankers-mpc/src/`)

| File | Purpose |
|------|---------|
| `types.rs` | `MpcConfig`, `BodyState`, `MpcSolution`, `ReferenceTrajectory`, `ContactPlan` |
| `gait.rs` | `GaitScheduler` with stand/trot/walk/bound presets |
| `centroidal.rs` | Continuous + discrete centroidal dynamics (A, B matrices) |
| `solver.rs` | Condensed QP build + Clarabel solve, CSC constraint matrix |
| `swing.rs` | `SwingConfig`, `SwingPlanner`, min-jerk trajectory, Raibert targeting |
| `wbc.rs` | Leg Jacobian computation, J^T force-to-torque mapping |
| `plugin.rs` | Bevy ECS plugin wiring |

### 13.2 Examples

| File | Purpose |
|------|---------|
| `examples/src/mpc_control.rs` | Shared control pipeline: MPC → J^T → motor commands |
| `examples/src/bin/quadruped_mpc.rs` | Headless MPC runner |
| `examples/src/bin/quadruped_mpc_viz.rs` | 3D visualization runner |
| `examples/src/bin/quadruped_mpc_bench.rs` | Parameter sweep benchmark (CLI args) |
| `examples/tests/mpc_walk.rs` | Integration test for walk stability |
| `examples/urdf/quadruped.urdf` | Robot model definition |

### 13.3 Supporting Crates

| Crate | Purpose |
|-------|---------|
| `clankers-ik` | Damped least-squares IK solver |
| `clankers-physics` | Rapier physics bridge, integration parameters |

### 13.4 Benchmark CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--velocity` | 0.3 | Forward velocity target (m/s) |
| `--gait` | trot | Gait type: stand, trot, walk, bound |
| `--steps` | 500 | Locomotion simulation steps |
| `--stabilize` | 100 | Stabilization steps |
| `--ramp` | 100 | Velocity ramp steps |
| `--q-vx` / `--q-vy` | 150.0 | Velocity tracking weight |
| `--q-pz` | 10.0 | Height tracking weight |
| `--q-roll` | 25.0 | Roll/pitch weight |
| `--q-omega` | 1.0 | Angular velocity weight |
| `--r-weight` | 1e-7 | Force effort weight |
| `--horizon` | 10 | MPC prediction horizon |
| `--mu` | 0.4 | Friction coefficient |
| `--f-max` | 120.0 | Max force per foot (N) |
| `--raibert-kv` | 0.15 | Swing targeting gain |
| `--cycle-time` | — | Gait cycle time (s) |
| `--duty-factor` | — | Gait duty factor |
| `--step-height` | 0.10 | Swing step height (m) |
| `--mpc-dt` | 0.02 | MPC timestep (s) |
