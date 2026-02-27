//! Shared MPC control logic for quadruped locomotion.
//!
//! Extracts the common MPC step (gait advance, FK, MPC solve, stance/swing
//! motor command generation) into a pure function so that headless, viz, test,
//! and benchmark binaries all share identical control code.

use bevy::prelude::Entity;
use clankers_ik::{DlsConfig, DlsSolver, IkTarget, KinematicChain};
use clankers_mpc::{
    BodyState, GaitScheduler, MpcConfig, MpcSolution, MpcSolver, ReferenceTrajectory, SwingConfig,
    raibert_foot_target, swing_foot_position, swing_foot_velocity,
    wbc::{
        compute_leg_jacobian, frames_f32_to_f64, jacobian_transpose_torques,
        transform_frames_to_world,
    },
};
use nalgebra::{UnitQuaternion, Vector3};

/// Per-leg kinematic and entity data shared across all MPC consumers.
pub struct LegRuntime {
    pub chain: KinematicChain,
    pub joint_entities: Vec<Entity>,
    pub is_prismatic: Vec<bool>,
    pub hip_offset: Vector3<f64>,
}

/// Configurable stance PD gains, exposed for experiment sweeps.
pub struct StanceConfig {
    pub hip_ab_kp: f32,
    pub hip_ab_kd: f32,
    pub hip_ab_max_f: f32,
    pub pitch_knee_kp: f32,
    pub pitch_knee_kd: f32,
    pub pitch_knee_max_f: f32,
}

impl Default for StanceConfig {
    fn default() -> Self {
        Self {
            hip_ab_kp: 500.0,
            hip_ab_kd: 20.0,
            hip_ab_max_f: 200.0,
            pitch_knee_kp: 5.0,
            pitch_knee_kd: 1.0,
            pitch_knee_max_f: 50.0,
        }
    }
}

/// Mutable state carried across MPC steps.
pub struct MpcLoopState {
    pub gait: GaitScheduler,
    pub solver: MpcSolver,
    pub config: MpcConfig,
    pub swing_config: SwingConfig,
    pub stance_config: StanceConfig,
    pub legs: Vec<LegRuntime>,
    pub swing_starts: Vec<Vector3<f64>>,
    pub swing_targets: Vec<Vector3<f64>>,
    pub prev_contacts: Vec<bool>,
    pub init_joint_angles: Vec<Vec<f32>>,
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

/// Compute one full MPC control step and return motor commands for all joints.
///
/// This is a pure function: it reads body and joint state, advances the gait,
/// solves the MPC QP, and produces position-motor commands. The caller is
/// responsible for applying the commands to Rapier joints or `MotorOverrides`.
///
/// `desired_velocity` should already be ramped by the caller if desired.
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

    // --- Advance gait and build contact sequence ---
    state.gait.advance(dt);
    let contacts_seq = state.gait.contact_sequence(state.config.horizon, dt);

    // --- MPC reference trajectory ---
    let x0 = body_state.to_state_vector(state.config.gravity);
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

        if is_contact && solution.converged {
            // --- Stance: J^T feedforward + IK q_desired ---
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

            let neg_force = -&solution.forces[leg_idx];
            let torques_ff = jacobian_transpose_torques(&jacobian, &neg_force);

            // IK q_desired: joint angles for current foot pos under upright body
            let desired_body_rot = UnitQuaternion::from_euler_angles(0.0, 0.0, desired_yaw);
            let desired_body_pos = Vector3::new(body_pos.x, body_pos.y, desired_height);
            let foot_in_desired =
                desired_body_rot.inverse() * (foot_world[leg_idx] - desired_body_pos);
            let ik_target = IkTarget::Position(foot_in_desired.cast::<f32>());
            let ik_solver = DlsSolver::new(DlsConfig {
                max_iterations: 10,
                position_tolerance: 1e-3,
                damping: 0.01,
                ..DlsConfig::default()
            });
            let ik_result = ik_solver.solve(&leg.chain, &ik_target, q);

            let sc = &state.stance_config;
            for (j, &entity) in leg.joint_entities.iter().enumerate() {
                let (kp_j, kd_j, max_f) = if j == 0 {
                    (sc.hip_ab_kp, sc.hip_ab_kd, sc.hip_ab_max_f)
                } else {
                    (sc.pitch_knee_kp, sc.pitch_knee_kd, sc.pitch_knee_max_f)
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
            // --- Swing: Cartesian PD via J^T with gain blending ---
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
                    state.swing_config.raibert_kv,
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

            let q = &joint_positions[leg_idx];
            let qd_vals = &joint_velocities[leg_idx];
            let (origins, axes, ee_pos) = leg.chain.joint_frames(q);
            let (origins_f64, axes_f64, _) = frames_f32_to_f64(&origins, &axes, &ee_pos);
            let (origins_world, axes_world) =
                transform_frames_to_world(&origins_f64, &axes_f64, body_quat, &body_pos);
            let jacobian = compute_leg_jacobian(
                &origins_world,
                &axes_world,
                p_actual,
                &leg.is_prismatic,
            );

            let qd_f64: Vec<f64> = qd_vals.iter().map(|&v| f64::from(v)).collect();
            let v_actual_relative = Vector3::new(
                (0..jacobian.ncols())
                    .map(|j| jacobian[(0, j)] * qd_f64[j])
                    .sum::<f64>(),
                (0..jacobian.ncols())
                    .map(|j| jacobian[(1, j)] * qd_f64[j])
                    .sum::<f64>(),
                (0..jacobian.ncols())
                    .map(|j| jacobian[(2, j)] * qd_f64[j])
                    .sum::<f64>(),
            );

            // Add the base's full spatial velocity to the leg's relative velocity
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

            // Blend motor gains from stance -> swing over first 20% of swing.
            // target_pos uses q0 (standing angles) as a stabilizing return spring.
            // Swing IK was tested but found to destabilize: the joint-space PD
            // toward IK targets double-counts with the J^T Cartesian feedforward,
            // causing overshoot and body roll.
            let blend = (swing_phase / 0.2).min(1.0) as f32;

            for (j, &entity) in leg.joint_entities.iter().enumerate() {
                let kd_swing = 2.0_f32;
                #[allow(clippy::cast_possible_truncation)]
                let target_vel = (torques[j] / f64::from(kd_swing)) as f32;

                // Ramp gains: liftoff values -> swing values at 20% phase
                let (kp_j, kd_j, max_f) = if j == 0 {
                    // Hip abduction: keep more lateral stiffness during swing
                    (
                        500.0 * (1.0 - blend) + 80.0 * blend,
                        20.0 * (1.0 - blend) + kd_swing * blend,
                        200.0 * (1.0 - blend) + 60.0 * blend,
                    )
                } else {
                    // Hip pitch / knee
                    (
                        20.0 * blend,
                        5.0 * (1.0 - blend) + kd_swing * blend,
                        50.0 * (1.0 - blend) + 60.0 * blend,
                    )
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
