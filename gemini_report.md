# Quadruped MPC Walking: Root Cause Analysis

## Summary
The quadruped robot still stumbles and fails to advance properly despite recent fixes. After a thorough review of the Whole Body Control (WBC) and Cartesian PD logic, I've identified that the issue is not a fundamental failure of the MPC solver or gait scheduler, but rather a **frame-of-reference bug in the swing leg's Cartesian PD controller**.

## Evaluation of Previous Fixes
The previous agent made valuable corrections that were structurally sound but incomplete:
1. **Force Sign Convention (`-force`)**: The previous fix correctly negated the MPC ground reaction force before mapping it to joint torques via Jacobian transpose (`J^T * -F_GRF`). This is physically accurate—to get the ground to push up/forward, the leg must push down/backward.
2. **Missing Body Velocities**: Updating `plugin.rs` to propagate the real linear and angular body velocities into the `BodyState` was also correct and absolutely necessary for the MPC reference trajectory to work.

However, the robot still struggles because of a subtle math error in how the swing leg is controlled.

## Root Cause of Stumbling
During the swing phase, the robot uses a Cartesian PD controller to track the desired foot trajectory. The damping term relies on tracking the velocity error: `Kd * (v_des - v_actual)`.

1. **`v_des` (Desired Velocity)**: Computed by differentiating the trajectory from `swing_starts` to `swing_targets`. Both of these are defined in the **World Frame**. Thus, `v_des` is the desired foot velocity in the **World Frame**.
2. **`v_actual` (Actual Velocity)**: Computed purely as `J * q_dot` (the leg Jacobian multiplied by the joint velocities).
   
**The Bug:** The leg Jacobian `J` only maps joint velocities to foot velocity *relative to the robot's base*. By omitting the base velocity, `v_actual` is effectively the foot's velocity relative to the body, strictly expressed in world-aligned axes. 

### Why this breaks walking:
When the robot is successfully walking forward at `0.3 m/s`, the swing foot needs to move forward at roughly `0.6 m/s` (World Frame) to get ahead of the body. 
- The controller sees `v_des = 0.6 m/s`.
- Because `v_actual` is missing the base velocity, it thinks the foot is only moving at `0.3 m/s` (relative to the body).
- The PD controller calculates a massive artificial tracking error (`0.6 - 0.3 = 0.3 m/s`) and begins applying a huge damping force. 

This velocity mismatch causes the Cartesian PD to generate erratic forces that fight the natural trajectory of the swing leg, completely destroying the swing phase. The leg fails to lift and advance properly, causing the robot to stumble and abruptly lose momentum.

## Proposed Fix (Do not implement, per instructions)
To fix this, `v_actual` must be transformed into the **World Frame** by adding the body's linear and angular velocity:

```rust
// 1. Calculate the vector from the body CoM to the foot
let r_foot = p_actual - body_pos;

// 2. Add the base's full spatial velocity to the leg's relative velocity
let v_actual_world = body_state.linear_velocity 
                   + body_state.angular_velocity.cross(&r_foot) 
                   + v_actual_relative; // (J * q_dot)
```
Replacing `v_actual` with `v_actual_world` in the swing leg PD controller will correctly align the reference frames and allow the leg to swing unimpeded.

---

## Standing Pose Enforcement Analysis
You asked why the standing pose enforcement (a strong positional PD explicitly tracking the nominal joint angles `q0` alongside the MPC feedforward torque) is implemented during Stance phase, specifically targeting the hip abduction joint (`hip_ab`).

### Why it exists
In highly underactuated or multi-link systems, **Centroidal MPC only plans the 6D wrench (forces and moments) upon the Center of Mass**, treating the feet as idealized point contacts that can push in 3D space. 
However, for a 3-DOF leg (hip abduction, hip pitch, knee pitch), a desired 3D point force does not strictly constrain the internal configuration (the "null space") of the leg if external disturbances or modeling errors occur. Because the basic MPC doesn’t incorporate full joint-space rigid body dynamics, it does not explicitly penalize the mechanical limbs swaying, splaying, or drifting laterally as long as the foot continues to deliver the correct overall force to the body.

Without any joint-space regularization during stance, the continuous accumulation of small numerical errors and external forces will cause the robot's legs to gradually splay outward (abduct) or cave inward until they reach a joint limit or a kinematic singularity, instantly causing a simulation failure. 

### Comparison with SOTA (State of the Art)
This hack is **standard practice** in leading Quadruped controllers.
- **MIT Cheetah 3 / Mini-Cheetah**: The official MIT controller uses identical logic. It applies a joint-space PD control on top of the MPC feedforward torques specifically to stabilize the internal null space and dampen high-frequency vibrations. The Cheetah code literally specifies exactly this: $ \tau = J^T F_{MPC} + K_p (q_{nom} - q) + K_d (\dot{q}_{nom} - \dot{q}) $.
- **OCS2 / Legged Gym**: Advanced frameworks also utilize "posture tasks" or "joint space tracking tasks" mixed into the Whole Body Controller (WBC) at a lower priority than the Cartesian tracking tasks.
- **Why just Hip Abduction?** Here, the config sets `kp = 500` for `hip_ab` but `0` for the pitch/knee joints. This is because the pitch joints must move wildly to facilitate walking forward/backward, so enforcing a strict constant angle on them would fight the MPC's walking goal. However, `hip_ab` (lateral leg spread) mostly remains constant during straight-line walking, making it a perfect target for stiff positional enforcement to stop lateral collapse.
