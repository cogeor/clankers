# clankers-mpc

Centroidal convex MPC and whole-body controller for legged locomotion.

## Overview

`clankers-mpc` implements a classical Model Predictive Control pipeline for quadruped (and general multi-legged) robots. The pipeline runs at control frequency (typically 50 Hz) and produces joint torques from a desired body trajectory.

### Pipeline stages

1. **Gait Scheduler** -- generates contact sequences (trot, walk, stand, bound)
2. **Centroidal MPC** -- solves a convex QP for optimal ground reaction forces
3. **Whole Body Controller** -- maps foot forces to joint torques via Jacobian transpose
4. **Swing Leg Planner** -- generates Bezier foot trajectories for airborne legs

## Architecture

The MPC treats the robot as a single rigid body ("centroidal dynamics") with massless legs. This simplification yields a convex QP that can be solved in sub-millisecond time using the Clarabel interior-point solver.

### State representation

The centroidal state vector has 13 dimensions:

```
x = [roll, pitch, yaw,     -- body orientation (Euler ZYX)
     px, py, pz,            -- body position (CoM)
     wx, wy, wz,            -- angular velocity
     vx, vy, vz,            -- linear velocity
     g]                     -- gravity constant (9.81)
```

### Dynamics

Continuous-time centroidal dynamics:

```
d/dt [Theta] = R_z^{-1}(yaw) * omega
d/dt [p]     = v
d/dt [omega] = I^{-1} sum_i (r_i x f_i)
d/dt [v]     = (1/m) sum_i f_i + [0, 0, -g]
```

Where `r_i` is the vector from CoM to foot `i`, and `f_i` is the ground reaction force at foot `i`. Discretized via first-order Euler (ZOH): `A_d = I + A_c*dt`, `B_d = B_c*dt`.

### QP formulation

Decision variables: `z = [x_1, ..., x_H, u_0, ..., u_{H-1}]`

- **Cost**: `min 0.5 z^T P z + q^T z` with block-diagonal P (state weights Q, control weight R)
- **Equality constraints**: dynamics propagation `x_{k+1} = A_d x_k + B_d u_k`
- **Inequality constraints**: linearized friction cones (4 linear constraints per stance foot), unilateral contact (`f_z >= 0`), maximum force (`f_z <= f_max`)
- **Swing constraints**: zero force equality for swing feet

Solved by Clarabel (pure Rust, interior-point, sub-1ms in release mode).

## Module structure

```
clankers-mpc/src/
  lib.rs          -- module declarations and re-exports
  types.rs        -- MpcConfig, BodyState, ContactPlan, MpcSolution, ReferenceTrajectory
  centroidal.rs   -- continuous dynamics (A_c, B_c) and discretization
  gait.rs         -- GaitScheduler, GaitType (Stand, Trot, Walk, Bound)
  solver.rs       -- MpcSolver with QP construction and Clarabel integration
  swing.rs        -- SwingConfig, swing_foot_position (Bezier), raibert_foot_target
  wbc.rs          -- Jacobian transpose WBC, compute_leg_jacobian
  plugin.rs       -- [bevy feature] ClankersMpcPlugin, MpcPipelineConfig/State
```

## Key types

### `MpcConfig`

Solver configuration with defaults tuned for a small quadruped (8.6 kg):

| Field | Default | Description |
|-------|---------|-------------|
| `horizon` | 10 | QP prediction horizon (steps) |
| `dt` | 0.02 | Control timestep (seconds) |
| `mass` | 8.6 | Robot mass (kg) |
| `gravity` | 9.81 | Gravitational acceleration |
| `friction_coeff` | 0.6 | Coulomb friction coefficient |
| `f_max` | 200.0 | Maximum normal force per foot (N) |
| `q_weights` | see code | 12-element state error cost |
| `r_weight` | 1e-4 | Control effort cost |
| `max_solver_iters` | 100 | Clarabel iteration limit |

### `GaitScheduler`

Phase-based gait generation with configurable offsets and duty factor:

| Gait | Phase offsets [FL,FR,RL,RR] | Duty factor | Cycle time |
|------|---------------------------|-------------|------------|
| Stand | [0,0,0,0] | 1.0 | 1.0s |
| Trot | [0, 0.5, 0.5, 0] | 0.5 | 0.4s |
| Walk | [0, 0.5, 0.25, 0.75] | 0.75 | 0.8s |
| Bound | [0, 0, 0.5, 0.5] | 0.5 | 0.3s |

### `MpcSolver`

Constructs and solves the sparse QP each control step. Key method:

```rust
pub fn solve(
    &self,
    x0: &DVector<f64>,
    foot_positions: &[Vector3<f64>],
    contacts: &ContactPlan,
    reference: &ReferenceTrajectory,
) -> MpcSolution
```

Returns `MpcSolution` with per-foot force vectors, convergence flag, and solve time.

### Whole Body Controller

Maps optimal foot forces to joint torques using Jacobian transpose:

```rust
// Compute 3xN linear Jacobian for a leg chain
let jacobian = compute_leg_jacobian(&origins, &axes, &ee_pos, &is_prismatic);

// Map force to torques: tau = J^T * F
let torques = jacobian_transpose_torques(&jacobian, &force);
```

### Swing leg trajectory

Parabolic Bezier arc with Raibert heuristic foot placement:

```rust
// Target = hip + velocity * T_stance/2
let target = raibert_foot_target(&hip_pos, &body_vel, stance_duration, ground_height);

// Bezier position at phase t in [0,1], height = 4*h*t*(1-t)
let pos = swing_foot_position(&start, &target, phase, step_height);
```

## Feature flags

| Feature | Description |
|---------|-------------|
| `bevy` | Enables `ClankersMpcPlugin` and ECS integration (requires `clankers-core`, `clankers-actuator`, `clankers-ik`) |

## Dependencies

```toml
[dependencies]
nalgebra = "0.33"
clarabel = "0.11"
clankers-urdf = { path = "../clankers-urdf" }

# Optional (bevy feature)
bevy = { optional = true }
clankers-core = { optional = true }
clankers-actuator = { optional = true }
clankers-ik = { optional = true }
```

## Usage

### Headless (no Bevy plugin)

```rust
use clankers_mpc::*;

// Configure
let config = MpcConfig::default();
let mut gait = GaitScheduler::quadruped(GaitType::Trot);
let solver = MpcSolver::new(config.clone());

// Each control step:
gait.advance(config.dt);
let contacts = gait.contact_sequence(config.horizon, config.dt);
let x0 = body_state.to_state_vector(config.gravity);
let reference = ReferenceTrajectory::constant_velocity(
    &body_state, &desired_velocity, desired_height,
    desired_yaw, config.horizon, config.dt, config.gravity,
);
let solution = solver.solve(&x0, &foot_positions, &contacts, &reference);

// Apply forces via WBC
for (leg_idx, force) in solution.forces.iter().enumerate() {
    if gait.is_contact(leg_idx) {
        let torques = jacobian_transpose_torques(&jacobian, force);
        // Apply torques to joints
    }
}
```

### With Bevy plugin

```rust
use clankers_mpc::*;

app.add_plugins(ClankersMpcPlugin);
app.insert_resource(MpcPipelineConfig { ... });
app.insert_resource(MpcPipelineState::new_quadruped(config, GaitType::Trot));
// The plugin runs the full pipeline in ClankersSet::Decide
```

## Example

See `examples/src/bin/quadruped_mpc.rs` for a complete headless example with:
- Quadruped URDF (4 legs x 2 joints = 8 DOF)
- Rapier physics with floating base and ground plane
- Stand-to-trot gait transition
- MPC balancing and walking

Run: `cargo run -p clankers-examples --bin quadruped_mpc`

## Tests

32 unit tests covering:
- Centroidal dynamics matrices (A, B dimensions, gravity propagation)
- Gait scheduling (all patterns, phase wrapping, contact sequences)
- QP solver (standing balance, friction cones, swing zero-force, solve time)
- WBC (Jacobian computation, torque mapping, frame conversion)
- Swing trajectories (start/end, peak height, symmetry, Raibert heuristic)
- Plugin types (body state extraction, pipeline state creation)

Run: `cargo test -p clankers-mpc --all-features`
