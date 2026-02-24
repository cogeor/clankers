# TASK: Centroidal Convex MPC for Quadruped Locomotion

## Goal
Implement a Model Predictive Control (MPC) pipeline enabling a quadruped robot to walk
using classical control (no RL). The system must:
1. Create a quadruped URDF (4 legs × 2 joints = 8 DOF)
2. Implement centroidal dynamics MPC (QP-based, using Clarabel)
3. Implement a Whole Body Controller (foot forces → joint torques via J^T)
4. Implement a gait scheduler (trot, walk, stand)
5. Implement swing leg trajectory planning
6. Integrate as a Bevy plugin
7. Create a working headless example
8. Document the MPC module

## Architecture

### Pipeline (runs each control step in ClankersSet::Decide)
```
Body State → Gait Scheduler → Centroidal MPC → WBC → Joint Torques
   ↑              ↓                 ↓           ↓
Physics    Contact Sequence    Foot Forces   τ = J^T F
```

### Centroidal Dynamics Model
State x ∈ R^12: [θ_rpy(3), p_xyz(3), ω_xyz(3), v_xyz(3)]
Control u ∈ R^(3c): [f1_xyz, f2_xyz, ...] for c contact feet
Dynamics: x_{k+1} = A_d x_k + B_d u_k + g_d

### QP Formulation (Clarabel)
- Decision vars: z = [x_1,...,x_H, u_0,...,u_{H-1}] ∈ R^(12H + 3cH)
- Cost: Σ (x_i - x_ref)^T Q (x_i - x_ref) + u_i^T R u_i
- Constraints: dynamics (equality), friction cone (linearized ≤), unilateral f_z ≥ 0
- Problem size for H=10, c=4: ~240 vars, ~320 constraints → sub-1ms solve

### Whole Body Controller
τ = J_c^T F_stance (stance legs: map foot forces to torques)
Swing legs: Bézier trajectory + IK → position commands

### Quadruped URDF (8 DOF)
- Body: 0.4m × 0.2m × 0.1m box, 5kg
- 4 legs at body corners: FL, FR, RL, RR
- Per leg: hip_pitch (Y axis) + knee_pitch (Y axis)
- Leg segments: 0.15m each, standing height ~0.35m

## Dependencies
- clarabel: QP solver (pure Rust)
- nalgebra: linear algebra (already in workspace)
- clankers-ik: per-leg FK and Jacobians (already built)

## Reference
- .delegate/plans/11_MPC_WASD_DOG_WALKING.md (full spec)
