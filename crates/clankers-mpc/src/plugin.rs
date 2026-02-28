//! Bevy ECS plugin for the MPC locomotion pipeline.
//!
//! Provides [`ClankersMpcPlugin`] which runs the full control pipeline each
//! frame: gait scheduling, centroidal MPC, whole body control, swing planning.
//!
//! The system runs in [`ClankersSet::Decide`], before actuator step.

use bevy::prelude::*;
use nalgebra::Vector3;

use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::ClankersSet;
use clankers_ik::KinematicChain;
use clankers_urdf::{RobotModel, SpawnedRobot};

use crate::gait::{AdaptiveGaitConfig, GaitScheduler, GaitType};
use crate::solver::MpcSolver;
use crate::swing::{SwingConfig, raibert_foot_target, swing_foot_position, swing_foot_velocity};
use crate::types::{BodyState, MpcConfig, ReferenceTrajectory};
use crate::wbc::{
    compute_leg_jacobian, frames_f32_to_f64, jacobian_transpose_torques,
    stance_damping_torques, transform_frames_to_world,
};

/// Bevy plugin for MPC-based locomotion control.
pub struct ClankersMpcPlugin;

impl Plugin for ClankersMpcPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, mpc_control_system.in_set(ClankersSet::Decide));
    }
}

/// Per-leg configuration.
#[derive(Debug, Clone)]
pub struct LegConfig {
    /// IK chain from body to foot (for FK and Jacobian).
    pub chain: KinematicChain,
    /// Joint entities in chain order.
    pub joint_entities: Vec<Entity>,
    /// Whether joints in this leg are prismatic (false = revolute).
    pub is_prismatic: Vec<bool>,
    /// Nominal hip position relative to body center (used for Raibert targeting).
    pub hip_offset: Vector3<f64>,
}

/// Static configuration for the MPC pipeline.
#[derive(Resource, Debug)]
pub struct MpcPipelineConfig {
    /// MPC solver configuration.
    pub mpc_config: MpcConfig,
    /// Swing trajectory configuration.
    pub swing_config: SwingConfig,
    /// Per-leg configurations, ordered [FL, FR, RL, RR] (or as appropriate).
    pub legs: Vec<LegConfig>,
    /// Entity for the robot body (to read transform).
    pub body_entity: Entity,
    /// Desired body height above ground (meters).
    pub desired_height: f64,
    /// Desired body velocity (set by user/teleop).
    pub desired_velocity: Vector3<f64>,
    /// Desired body yaw (radians).
    pub desired_yaw: f64,
    /// Ground height (z coordinate of ground plane).
    pub ground_height: f64,
    /// Joint-space damping gain for stance legs (MIT uses 0.2).
    pub stance_kd_joint: f64,
    /// Adaptive gait timing configuration. `None` disables adaptation.
    pub adaptive_gait: Option<AdaptiveGaitConfig>,
}

/// Runtime state for the MPC pipeline.
#[derive(Resource)]
pub struct MpcPipelineState {
    /// Gait scheduler.
    pub gait: GaitScheduler,
    /// MPC solver.
    pub solver: MpcSolver,
    /// Last known foot positions (for swing start).
    pub foot_positions: Vec<Vector3<f64>>,
    /// Swing target positions (where feet should land).
    pub swing_targets: Vec<Vector3<f64>>,
    /// Swing start positions (where feet lifted off).
    pub swing_starts: Vec<Vector3<f64>>,
    /// Previous contact state per leg (for liftoff detection).
    pub prev_contacts: Vec<bool>,
    /// Body linear velocity in world frame (m/s).
    ///
    /// Must be updated each frame by the user (e.g., from physics engine).
    /// The MPC system reads this instead of estimating from transforms.
    pub body_linear_velocity: Vector3<f64>,
    /// Body angular velocity in world frame (rad/s).
    ///
    /// Must be updated each frame by the user (e.g., from physics engine).
    pub body_angular_velocity: Vector3<f64>,
    /// Last MPC solve time in microseconds.
    pub last_solve_time_us: u64,
    /// Whether the last MPC solve converged.
    pub last_converged: bool,
}

impl MpcPipelineState {
    /// Create initial state for a quadruped.
    pub fn new_quadruped(mpc_config: MpcConfig, gait: GaitType) -> Self {
        let n_feet = 4;
        Self {
            gait: GaitScheduler::quadruped(gait),
            solver: MpcSolver::new(mpc_config, 4),
            foot_positions: vec![Vector3::zeros(); n_feet],
            swing_targets: vec![Vector3::zeros(); n_feet],
            swing_starts: vec![Vector3::zeros(); n_feet],
            prev_contacts: vec![true; n_feet],
            body_linear_velocity: Vector3::zeros(),
            body_angular_velocity: Vector3::zeros(),
            last_solve_time_us: 0,
            last_converged: false,
        }
    }
}

/// Build leg configurations from a URDF model and spawned robot.
pub fn build_leg_configs(
    model: &RobotModel,
    spawned: &SpawnedRobot,
    foot_links: &[&str],
    hip_offsets: &[Vector3<f64>],
) -> Vec<LegConfig> {
    let mut legs = Vec::with_capacity(foot_links.len());

    for (i, &foot_link) in foot_links.iter().enumerate() {
        let chain = KinematicChain::from_model(model, foot_link)
            .unwrap_or_else(|| panic!("Failed to build chain to {foot_link}"));

        let joint_entities: Vec<Entity> = chain
            .joint_names()
            .iter()
            .map(|name| {
                spawned
                    .joint_entity(name)
                    .unwrap_or_else(|| panic!("Joint {name} not found"))
            })
            .collect();

        let is_prismatic = chain
            .joints()
            .iter()
            .map(|j| j.is_prismatic)
            .collect();

        legs.push(LegConfig {
            chain,
            joint_entities,
            is_prismatic,
            hip_offset: hip_offsets[i],
        });
    }

    legs
}

/// Extract body state and rotation quaternion from a Bevy Transform component.
///
/// Velocities must be provided by the caller (e.g., from `RapierContext`
/// or another physics backend). The `MpcPipelineState` fields
/// `body_linear_velocity` and `body_angular_velocity` are the intended source.
pub fn body_state_from_transform(
    transform: &GlobalTransform,
    linear_velocity: &Vector3<f64>,
    angular_velocity: &Vector3<f64>,
) -> (BodyState, nalgebra::UnitQuaternion<f64>) {
    let t = transform.translation();
    let bevy_rot = transform.to_scale_rotation_translation().1;
    let (yaw, pitch, roll) = bevy_rot.to_euler(EulerRot::ZYX);

    let body_quat = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        f64::from(bevy_rot.w),
        f64::from(bevy_rot.x),
        f64::from(bevy_rot.y),
        f64::from(bevy_rot.z),
    ));

    (
        BodyState {
            orientation: Vector3::new(f64::from(roll), f64::from(pitch), f64::from(yaw)),
            position: Vector3::new(f64::from(t.x), f64::from(t.y), f64::from(t.z)),
            angular_velocity: *angular_velocity,
            linear_velocity: *linear_velocity,
        },
        body_quat,
    )
}

/// The main MPC control system.
///
/// Runs each frame in `ClankersSet::Decide`:
/// 1. Read body state from transform
/// 2. Read joint states for each leg
/// 3. Compute FK for each foot
/// 4. Advance gait and generate contact sequence
/// 5. Solve centroidal MPC for optimal forces
/// 6. WBC: stance legs get J^T torques + damping, swing legs get Cartesian PD
#[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
fn mpc_control_system(
    config: Option<Res<MpcPipelineConfig>>,
    state: Option<ResMut<MpcPipelineState>>,
    transforms: Query<&GlobalTransform>,
    mut joints: Query<(&JointState, &mut JointCommand)>,
) {
    let (Some(config), Some(mut state)) = (config, state) else {
        return;
    };

    // 1. Read body state
    let Ok(body_tf) = transforms.get(config.body_entity) else {
        return;
    };
    let (body_state, body_quat) = body_state_from_transform(
        body_tf,
        &state.body_linear_velocity,
        &state.body_angular_velocity,
    );
    let body_pos = body_state.position;

    // 2. Read joint states and compute foot positions via FK
    let n_feet = config.legs.len();
    let mut all_joint_positions: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
    let mut all_joint_velocities: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
    let mut foot_positions_world: Vec<Vector3<f64>> = Vec::with_capacity(n_feet);

    for leg in &config.legs {
        let mut q = Vec::with_capacity(leg.joint_entities.len());
        let mut qd = Vec::with_capacity(leg.joint_entities.len());
        for &entity in &leg.joint_entities {
            if let Ok((js, _)) = joints.get(entity) {
                q.push(js.position);
                qd.push(js.velocity);
            } else {
                q.push(0.0);
                qd.push(0.0);
            }
        }

        let ee_body = leg.chain.forward_kinematics(&q);
        let ee_body_vec = Vector3::new(
            f64::from(ee_body.translation.x),
            f64::from(ee_body.translation.y),
            f64::from(ee_body.translation.z),
        );
        let foot_world = body_quat * ee_body_vec + body_pos;

        foot_positions_world.push(foot_world);
        all_joint_positions.push(q);
        all_joint_velocities.push(qd);
    }

    state.foot_positions.clone_from(&foot_positions_world);

    // 3. Advance gait and adapt timing
    let dt = config.mpc_config.dt;
    state.gait.advance(dt);
    if let Some(ref adaptive_cfg) = config.adaptive_gait {
        let speed = config.desired_velocity.xy().norm();
        state.gait.adapt_timing(speed, adaptive_cfg);
    }

    // 4. Generate contact sequence
    let contacts = state.gait.contact_sequence(config.mpc_config.horizon, dt);

    // 5. Build reference trajectory
    let x0 = body_state.to_state_vector(config.mpc_config.gravity);
    let reference = ReferenceTrajectory::constant_velocity(
        &body_state,
        &config.desired_velocity,
        config.desired_height,
        config.desired_yaw,
        config.mpc_config.horizon,
        dt,
        config.mpc_config.gravity,
    );

    // 6. Solve MPC
    let solution = state
        .solver
        .solve(&x0, &foot_positions_world, &contacts, &reference);
    state.last_solve_time_us = solution.solve_time_us;
    state.last_converged = solution.converged;

    // 7. Apply control
    let stance_duration = state.gait.duty_factor() * state.gait.cycle_time();

    for (leg_idx, leg) in config.legs.iter().enumerate() {
        let is_contact = state.gait.is_contact(leg_idx);

        // Detect liftoff transition: set swing_starts at the exact moment
        if state.prev_contacts[leg_idx] && !is_contact {
            state.swing_starts[leg_idx] = foot_positions_world[leg_idx];
        }

        if is_contact && solution.converged {
            // Stance leg: J^T force feedforward + joint damping
            let q = &all_joint_positions[leg_idx];
            let qd = &all_joint_velocities[leg_idx];
            let (origins, axes, ee_pos) = leg.chain.joint_frames(q);
            let (origins_f64, axes_f64, _) = frames_f32_to_f64(&origins, &axes, &ee_pos);

            let (origins_world, axes_world) =
                transform_frames_to_world(&origins_f64, &axes_f64, &body_quat, &body_pos);

            let foot_world = &foot_positions_world[leg_idx];
            let jacobian =
                compute_leg_jacobian(&origins_world, &axes_world, foot_world, &leg.is_prismatic);

            // Negate: MPC gives ground reaction forces (ground pushes up on
            // foot); the body must apply -F through the foot (Newton's 3rd law).
            let neg_force = -solution.forces[leg_idx];
            let torques_ff = jacobian_transpose_torques(&jacobian, &neg_force);

            // Joint-space damping (MIT Cheetah: Kd = 0.2 * I)
            let qd_f64: Vec<f64> = qd.iter().map(|&v| f64::from(v)).collect();
            let torques_damp = stance_damping_torques(&qd_f64, config.stance_kd_joint);

            for (j, &entity) in leg.joint_entities.iter().enumerate() {
                if let Ok((_, mut cmd)) = joints.get_mut(entity) {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        cmd.value = (torques_ff[j] + torques_damp[j]) as f32;
                    }
                }
            }
        } else {
            // Swing leg: min-jerk trajectory with Cartesian PD via J^T
            let swing_phase = state.gait.swing_phase(leg_idx);

            let swing_duration = (1.0 - state.gait.duty_factor()) * state.gait.cycle_time();

            if swing_phase < 0.05 {
                let hip_world = body_quat * leg.hip_offset + body_pos;
                state.swing_targets[leg_idx] = raibert_foot_target(
                    &hip_world,
                    &body_state.linear_velocity,
                    &config.desired_velocity,
                    stance_duration,
                    swing_duration,
                    config.ground_height,
                    config.swing_config.cp_gain,
                    config.desired_height - config.ground_height,
                    config.mpc_config.gravity,
                    config.swing_config.max_reach,
                );
            }

            let p_des = swing_foot_position(
                &state.swing_starts[leg_idx],
                &state.swing_targets[leg_idx],
                swing_phase,
                config.swing_config.step_height,
            );
            let v_des = swing_foot_velocity(
                &state.swing_starts[leg_idx],
                &state.swing_targets[leg_idx],
                swing_phase,
                config.swing_config.step_height,
                swing_duration,
            );

            let p_actual = &foot_positions_world[leg_idx];

            let q = &all_joint_positions[leg_idx];
            let qd = &all_joint_velocities[leg_idx];
            let (origins, axes, ee_pos) = leg.chain.joint_frames(q);
            let (origins_f64, axes_f64, _) = frames_f32_to_f64(&origins, &axes, &ee_pos);
            let (origins_world, axes_world) =
                transform_frames_to_world(&origins_f64, &axes_f64, &body_quat, &body_pos);
            let jacobian =
                compute_leg_jacobian(&origins_world, &axes_world, p_actual, &leg.is_prismatic);

            let qd_vec: Vec<f64> = qd.iter().map(|&v| f64::from(v)).collect();
            let v_actual = Vector3::new(
                (0..jacobian.ncols())
                    .map(|j| jacobian[(0, j)] * qd_vec[j])
                    .sum::<f64>(),
                (0..jacobian.ncols())
                    .map(|j| jacobian[(1, j)] * qd_vec[j])
                    .sum::<f64>(),
                (0..jacobian.ncols())
                    .map(|j| jacobian[(2, j)] * qd_vec[j])
                    .sum::<f64>(),
            );

            let kp = &config.swing_config.kp_cartesian;
            let kd = &config.swing_config.kd_cartesian;
            let foot_force = Vector3::new(
                kp.x * (p_des.x - p_actual.x) + kd.x * (v_des.x - v_actual.x),
                kp.y * (p_des.y - p_actual.y) + kd.y * (v_des.y - v_actual.y),
                kp.z * (p_des.z - p_actual.z) + kd.z * (v_des.z - v_actual.z),
            );

            let torques = jacobian_transpose_torques(&jacobian, &foot_force);

            // Blend: fade in swing torques + fade out stance damping over first 20%
            let blend = (swing_phase / 0.2).min(1.0);
            let damp_fade = 1.0 - blend;

            for (j, &entity) in leg.joint_entities.iter().enumerate() {
                if let Ok((_, mut cmd)) = joints.get_mut(entity) {
                    let damp = -config.stance_kd_joint * f64::from(qd[j]);
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        cmd.value = (torques[j] + damp_fade * damp) as f32;
                    }
                }
            }
        }

        state.prev_contacts[leg_idx] = is_contact;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::STATE_DIM;

    #[test]
    fn body_state_extraction() {
        let state = BodyState {
            orientation: Vector3::zeros(),
            position: Vector3::new(1.0, 2.0, 0.35),
            angular_velocity: Vector3::zeros(),
            linear_velocity: Vector3::zeros(),
        };
        let x = state.to_state_vector(9.81);
        assert_eq!(x.len(), STATE_DIM);
        assert!((x[3] - 1.0).abs() < 1e-10);
        assert!((x[5] - 0.35).abs() < 1e-10);
    }

    #[test]
    fn pipeline_state_creation() {
        let config = MpcConfig::default();
        let state = MpcPipelineState::new_quadruped(config, GaitType::Trot);
        assert_eq!(state.foot_positions.len(), 4);
        assert_eq!(state.swing_targets.len(), 4);
    }
}
