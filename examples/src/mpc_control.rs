//! Shared MPC control logic for quadruped locomotion.
//!
//! Pipeline:
//!   Stance: tau = kp*(q_ik-q) + kd*(J^T*f_mpc/kd - qd)  (IK + feedforward)
//!   Swing:  tau = J^T * (Kp*(p_des-p) + Kd*(v_des-v))    (Cartesian PD)
//!
//! IK-derived q_desired makes kp*(q_ik-q) ≈ 0 during stance, so the
//! effective torque is dominated by the MPC feedforward term.

use bevy::prelude::{Entity, EulerRot};
use clankers_ik::{DlsConfig, DlsSolver, IkTarget};
use clankers_mpc::{
    AdaptiveGaitConfig, BodyState, DisturbanceEstimator, GaitScheduler, MpcConfig, MpcSolution,
    MpcSolver, ReferenceTrajectory, SwingConfig, raibert_foot_target, swing_foot_position,
    swing_foot_velocity,
    wbc::{
        compute_leg_jacobian, frames_f32_to_f64, jacobian_transpose_torques,
        transform_frames_to_world,
    },
};
use clankers_physics::rapier::RapierContext;
use nalgebra::{UnitQuaternion, Vector3};

/// Per-leg kinematic and entity data shared across all MPC consumers.
pub struct LegRuntime {
    pub chain: clankers_ik::KinematicChain,
    pub joint_entities: Vec<Entity>,
    pub is_prismatic: Vec<bool>,
    pub hip_offset: Vector3<f64>,
}

/// Mutable state carried across MPC steps.
pub struct MpcLoopState {
    pub gait: GaitScheduler,
    pub solver: MpcSolver,
    pub config: MpcConfig,
    pub swing_config: SwingConfig,
    pub adaptive_gait: Option<AdaptiveGaitConfig>,
    pub legs: Vec<LegRuntime>,
    pub swing_starts: Vec<Vector3<f64>>,
    pub swing_targets: Vec<Vector3<f64>>,
    pub prev_contacts: Vec<bool>,
    pub init_joint_angles: Vec<Vec<f32>>,
    /// Foot link names for contact detection (e.g., ["fl_foot", "fr_foot", ...]).
    /// When set, enables ground-truth contact feedback to override the gait schedule.
    pub foot_link_names: Option<Vec<String>>,
    /// Disturbance estimator for model-mismatch compensation.
    /// When set, corrects MPC initial state using estimated velocity biases.
    pub disturbance_estimator: Option<DisturbanceEstimator>,
}

/// A single joint motor command produced by the MPC step.
pub struct MotorCommand {
    pub entity: Entity,
    pub target_pos: f32,
    pub target_vel: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub max_force: f32,
}

/// Result of one MPC control step.
pub struct MpcStepResult {
    pub motor_commands: Vec<MotorCommand>,
    pub solution: MpcSolution,
    pub foot_world: Vec<Vector3<f64>>,
    pub contacts: Vec<bool>,
}

/// Stance pitch/knee stiffness (Nm/rad).
/// IK-derived q_desired makes this compatible with MPC feedforward:
/// kp*(q_ik - q) ≈ 0 during stance since q_ik tracks current foot position.
const STANCE_KP_JOINT: f32 = 0.5;
/// Stance pitch/knee damping (Nm*s/rad).
const STANCE_KD_JOINT: f32 = 0.1;
/// Stance pitch/knee max motor force (N).
const STANCE_MAX_F: f32 = 200.0;

/// Hip abduction gains — high stiffness for lateral stability.
const HIP_AB_KP: f32 = 200.0;
const HIP_AB_KD: f32 = 20.0;
const HIP_AB_MAX_F: f32 = 200.0;

/// Swing max motor force (N).
const SWING_MAX_F: f32 = 60.0;

/// Extract body state and quaternion from Rapier, applying floating origin offset.
///
/// The returned position is in true world coordinates (f64), computed as
/// `rapier_local_pos + world_origin` to preserve precision over long runs.
pub fn body_state_from_rapier(
    ctx: &RapierContext,
    link_name: &str,
) -> Option<(BodyState, UnitQuaternion<f64>)> {
    let handle = ctx.body_handles.get(link_name)?;
    let body = ctx.rigid_body_set.get(*handle)?;

    let r = body.rotation();
    let (yaw, pitch, roll) = r.to_euler(EulerRot::ZYX);

    let lv = body.linvel();
    let av = body.angvel();

    let body_quat = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        f64::from(r.w),
        f64::from(r.x),
        f64::from(r.y),
        f64::from(r.z),
    ));

    // Use world_position_f64 which adds the floating origin offset
    let world_pos = ctx.world_position_f64(*handle)?;

    Some((
        BodyState {
            orientation: Vector3::new(f64::from(roll), f64::from(pitch), f64::from(yaw)),
            position: Vector3::new(world_pos[0], world_pos[1], world_pos[2]),
            angular_velocity: Vector3::new(f64::from(av.x), f64::from(av.y), f64::from(av.z)),
            linear_velocity: Vector3::new(f64::from(lv.x), f64::from(lv.y), f64::from(lv.z)),
        },
        body_quat,
    ))
}

/// Query ground-truth foot contacts from physics.
///
/// Returns `None` if `foot_link_names` is not set on the state.
pub fn detect_foot_contacts(
    ctx: &RapierContext,
    state: &MpcLoopState,
) -> Option<Vec<bool>> {
    let names = state.foot_link_names.as_ref()?;
    Some(names.iter().map(|name| ctx.has_active_contacts(name)).collect())
}

/// Compute one full MPC control step and return motor commands for all joints.
///
/// Pipeline:
///   1. FK → foot positions
///   2. MPC solve → ground reaction forces
///   3. Stance: J^T * f_mpc (pure feedforward + joint damping)
///   4. Swing: Cartesian PD → J^T → torques
#[allow(clippy::too_many_arguments)]
pub fn compute_mpc_step(
    state: &mut MpcLoopState,
    body_state: &BodyState,
    body_quat: &UnitQuaternion<f64>,
    joint_positions: &[Vec<f32>],
    joint_velocities: &[Vec<f32>],
    desired_velocity: &Vector3<f64>,
    desired_height: f64,
    desired_yaw: f64,
    ground_height: f64,
    actual_contacts: Option<&[bool]>,
) -> MpcStepResult {
    let dt = state.config.dt;
    let body_pos = body_state.position;
    let n_feet = state.legs.len();

    // --- FK: compute foot positions in world frame ---
    let mut foot_world: Vec<Vector3<f64>> = Vec::with_capacity(n_feet);
    for leg in &state.legs {
        let q = &joint_positions[foot_world.len()];
        let ee_body = leg.chain.forward_kinematics(q);
        let ee_body_vec = Vector3::new(
            f64::from(ee_body.translation.x),
            f64::from(ee_body.translation.y),
            f64::from(ee_body.translation.z),
        );
        foot_world.push(body_quat * ee_body_vec + body_pos);
    }

    // --- Advance gait and adapt timing ---
    state.gait.advance(dt);
    if let Some(ref adaptive_cfg) = state.adaptive_gait {
        let speed = desired_velocity.xy().norm();
        state.gait.adapt_timing(speed, adaptive_cfg);
    }
    // Apply ground-truth contact feedback to override gait schedule
    if let Some(contacts) = actual_contacts {
        state.gait.apply_contact_feedback(contacts);
    }
    let contacts_seq = state.gait.contact_sequence(state.config.horizon, dt);

    // --- MPC reference trajectory ---
    let x0_raw = body_state.to_state_vector(state.config.gravity);

    // Apply disturbance compensation if estimator is active
    let x0 = if let Some(ref mut est) = state.disturbance_estimator {
        est.update(&x0_raw, None);
        est.compensate_state(&x0_raw)
    } else {
        x0_raw.clone()
    };

    let reference = ReferenceTrajectory::constant_velocity(
        body_state,
        desired_velocity,
        desired_height,
        desired_yaw,
        state.config.horizon,
        dt,
        state.config.gravity,
    );

    // --- Solve MPC ---
    let solution = state.solver.solve(&x0, &foot_world, &contacts_seq, &reference);

    // Store prediction for next disturbance estimate update
    if let Some(ref mut est) = state.disturbance_estimator {
        if solution.converged && solution.state_trajectory.len() >= 13 {
            let x_pred = solution.state_trajectory.rows(0, 13).into_owned();
            est.set_prediction(x_pred);
        }
    }
    let stance_duration = state.gait.duty_factor() * state.gait.cycle_time();

    // --- Generate motor commands ---
    let mut motor_commands: Vec<MotorCommand> = Vec::new();
    let mut contacts = Vec::with_capacity(n_feet);

    for (leg_idx, leg) in state.legs.iter().enumerate() {
        let is_contact = state.gait.is_contact(leg_idx);
        contacts.push(is_contact);

        // Detect liftoff transition
        if state.prev_contacts[leg_idx] && !is_contact {
            state.swing_starts[leg_idx] = foot_world[leg_idx];
        }

        // Compute Jacobian (needed for both stance and swing)
        let q = &joint_positions[leg_idx];
        let (origins, axes, ee_pos) = leg.chain.joint_frames(q);
        let (origins_f64, axes_f64, _) = frames_f32_to_f64(&origins, &axes, &ee_pos);
        let (origins_world, axes_world) =
            transform_frames_to_world(&origins_f64, &axes_f64, body_quat, &body_pos);
        let jacobian = compute_leg_jacobian(
            &origins_world,
            &axes_world,
            &foot_world[leg_idx],
            &leg.is_prismatic,
        );

        if is_contact && solution.converged {
            // --- Stance: J^T feedforward + IK position hold ---
            //
            // Feedforward with damping cancellation:
            //   target_vel = torque_ff/kd + qd
            //   tau = kp*(q_ik - q) + kd*(torque_ff/kd + qd - qd)
            //       = kp*(q_ik - q) + torque_ff
            //
            // Adding qd to target_vel cancels the motor damping term that
            // would otherwise oppose joint motion, allowing MPC forces to
            // pass through without resistive losses.

            let neg_force = -&solution.forces[leg_idx];
            let torques_ff = jacobian_transpose_torques(&jacobian, &neg_force);

            // IK: joint angles for current foot pos under desired body pose
            let desired_body_rot = UnitQuaternion::from_euler_angles(0.0, 0.0, desired_yaw);
            let desired_body_pos = Vector3::new(body_pos.x, body_pos.y, desired_height);
            let foot_in_body =
                desired_body_rot.inverse() * (foot_world[leg_idx] - desired_body_pos);
            let ik_target = IkTarget::Position(foot_in_body.cast::<f32>());
            let ik_solver = DlsSolver::new(DlsConfig {
                max_iterations: 10,
                position_tolerance: 1e-3,
                damping: 0.01,
                ..DlsConfig::default()
            });
            let ik_result = ik_solver.solve(&leg.chain, &ik_target, q);

            for (j, &entity) in leg.joint_entities.iter().enumerate() {
                let (kp_j, kd_j, max_f) = if j == 0 {
                    (HIP_AB_KP, HIP_AB_KD, HIP_AB_MAX_F)
                } else {
                    (STANCE_KP_JOINT, STANCE_KD_JOINT, STANCE_MAX_F)
                };

                #[allow(clippy::cast_possible_truncation)]
                let target_vel = (torques_ff[j] / f64::from(kd_j)) as f32;
                motor_commands.push(MotorCommand {
                    entity,
                    target_pos: ik_result.joint_positions[j],
                    target_vel,
                    stiffness: kp_j,
                    damping: kd_j,
                    max_force: max_f,
                });
            }
        } else {
            // --- Swing: Cartesian PD via J^T ---
            let swing_phase = state.gait.swing_phase(leg_idx);
            let swing_duration = (1.0 - state.gait.duty_factor()) * state.gait.cycle_time();

            if swing_phase < 0.05 {
                let hip_world = body_quat * leg.hip_offset + body_pos;
                state.swing_targets[leg_idx] = raibert_foot_target(
                    &hip_world,
                    &body_state.linear_velocity,
                    desired_velocity,
                    stance_duration,
                    swing_duration,
                    ground_height,
                    state.swing_config.cp_gain,
                    desired_height - ground_height,
                    state.config.gravity,
                    state.swing_config.max_reach,
                );
            }

            let p_des = swing_foot_position(
                &state.swing_starts[leg_idx],
                &state.swing_targets[leg_idx],
                swing_phase,
                state.swing_config.step_height,
            );
            let v_des = swing_foot_velocity(
                &state.swing_starts[leg_idx],
                &state.swing_targets[leg_idx],
                swing_phase,
                state.swing_config.step_height,
                swing_duration,
            );

            let p_actual = &foot_world[leg_idx];

            let qd_vals = &joint_velocities[leg_idx];
            let qd_f64: Vec<f64> = qd_vals.iter().map(|&v| f64::from(v)).collect();
            let v_actual_relative = Vector3::new(
                (0..jacobian.ncols())
                    .map(|c| jacobian[(0, c)] * qd_f64[c])
                    .sum::<f64>(),
                (0..jacobian.ncols())
                    .map(|c| jacobian[(1, c)] * qd_f64[c])
                    .sum::<f64>(),
                (0..jacobian.ncols())
                    .map(|c| jacobian[(2, c)] * qd_f64[c])
                    .sum::<f64>(),
            );

            let r_foot = p_actual - body_pos;
            let v_actual = body_state.linear_velocity
                + body_state.angular_velocity.cross(&r_foot)
                + v_actual_relative;

            let kp_cart = &state.swing_config.kp_cartesian;
            let kd_cart = &state.swing_config.kd_cartesian;
            let foot_force = Vector3::new(
                kp_cart.x * (p_des.x - p_actual.x) + kd_cart.x * (v_des.x - v_actual.x),
                kp_cart.y * (p_des.y - p_actual.y) + kd_cart.y * (v_des.y - v_actual.y),
                kp_cart.z * (p_des.z - p_actual.z) + kd_cart.z * (v_des.z - v_actual.z),
            );

            let torques = jacobian_transpose_torques(&jacobian, &foot_force);

            let kd_swing = 0.5_f32;

            for (j, &entity) in leg.joint_entities.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let target_vel = (torques[j] / f64::from(kd_swing)) as f32;

                let (kp_j, kd_j, max_f) = if j == 0 {
                    (80.0, kd_swing, SWING_MAX_F)
                } else {
                    (20.0, kd_swing, SWING_MAX_F)
                };

                motor_commands.push(MotorCommand {
                    entity,
                    target_pos: state.init_joint_angles[leg_idx][j],
                    target_vel,
                    stiffness: kp_j,
                    damping: kd_j,
                    max_force: max_f,
                });
            }
        }

        state.prev_contacts[leg_idx] = is_contact;
    }

    MpcStepResult {
        motor_commands,
        solution,
        foot_world,
        contacts,
    }
}
