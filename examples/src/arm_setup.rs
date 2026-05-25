//! Shared 6-DOF arm setup: URDF, physics, position-mode actuators, sensors, IK.
//!
//! Both arm binaries (`arm_ik`, `arm_manipulation`, `arm_bench`, `arm_gym`)
//! share identical setup code.  This module extracts it into a single function
//! so each binary calls `setup_arm(config)` and gets back everything it needs.

use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
use std::sync::Arc;

use bevy::prelude::*;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_actuator_core::prelude::ControlMode;
use clankers_core::types::RobotGroup;
use clankers_env::prelude::*;
use clankers_ik::{DlsConfig, DlsSolver, IkTarget, KinematicChain};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_physics::rapier::systems::validate_motor_coverage;
use clankers_physics::rapier::{
    MotorOverrideParams, MotorOverrides, RapierBackend, RapierBackendFixed, RapierContext,
    bridge::register_robot,
};
use clankers_sim::{SceneBuilder, SpawnedScene};
use clankers_urdf::RobotModel;
use nalgebra::Vector3;

use crate::SIX_DOF_ARM_URDF;

/// Default resting pose for the 6-DOF arm.
///
/// j2 tilts shoulder 45° forward, j3 bends elbow 90°, j5 pitches wrist down.
pub const REST_POSE: [f32; 6] = [0.0, FRAC_PI_4, FRAC_PI_2, 0.0, FRAC_PI_4, 0.0];

/// Configuration for arm setup — knobs that differ between binaries.
#[derive(Clone)]
pub struct ArmSetupConfig {
    /// Maximum episode length (500 for bench, 50_000 for viz).
    pub max_episode_steps: u32,
    /// When true, register physics on `FixedUpdate` instead of `Update`.
    pub use_fixed_update: bool,
    /// Sensor DOF: 6 for arm-only, 8 for arm + gripper.
    pub sensor_dof: usize,
    /// Initial joint positions for the 6 arm joints (chain order).
    /// Defaults to [`REST_POSE`].
    pub initial_positions: [f32; 6],
}

impl Default for ArmSetupConfig {
    fn default() -> Self {
        Self {
            max_episode_steps: 500,
            use_fixed_update: false,
            sensor_dof: 6,
            initial_positions: REST_POSE,
        }
    }
}

/// Everything produced by [`setup_arm`] that callers need.
pub struct ArmSetup {
    pub scene: SpawnedScene,
    pub model: RobotModel,
    pub chain: KinematicChain,
    pub joint_entities: Vec<Entity>,
    pub arm_joint_names: Vec<String>,
    /// Layout bound to ALL spawned arm joint entities (incl. gripper
    /// fingers when present). Suitable for layout-bound sensors and for
    /// [`validate_motor_coverage`].
    pub joint_layout: Arc<clankers_core::layout::JointLayout>,
}

/// Build a fully configured arm scene: URDF, physics, position-mode actuators,
/// sensors, IK chain, and joint entity mapping.
#[allow(clippy::needless_pass_by_value)]
pub fn setup_arm(config: ArmSetupConfig) -> ArmSetup {
    // 1. Parse URDF
    let model =
        clankers_urdf::parse_string(SIX_DOF_ARM_URDF).expect("failed to parse six_dof_arm URDF");

    // 2. Build IK chain (needed for joint names before spawning)
    let chain = KinematicChain::from_model(&model, "end_effector")
        .expect("failed to build IK chain to end_effector");

    // 3. Build initial positions map from rest pose
    let initial_positions: HashMap<String, f32> = chain
        .joint_names()
        .iter()
        .zip(config.initial_positions.iter())
        .map(|(name, &pos)| (name.to_string(), pos))
        .collect();

    // 4. Build scene with position-controlled actuators
    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(config.max_episode_steps)
        .with_robot(model.clone(), initial_positions)
        .build();

    let spawned = &scene.robots["six_dof_arm"];

    // 3. Switch all actuators to position mode (PID controller)
    for entity in spawned.joints.values() {
        let mut actuator = scene
            .app
            .world_mut()
            .get_mut::<clankers_actuator::components::Actuator>(*entity)
            .unwrap();
        *actuator = clankers_actuator::components::Actuator::new(
            actuator.motor.clone(),
            actuator.transmission.clone(),
            actuator.friction.clone(),
            ControlMode::Position {
                kp: 100.0,
                ki: 0.0,
                kd: 10.0,
            },
        );
    }

    // 4. Add Rapier physics
    if config.use_fixed_update {
        scene
            .app
            .add_plugins(ClankersPhysicsPlugin::new(RapierBackendFixed));
    } else {
        scene
            .app
            .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));
    }

    {
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        register_robot(&mut ctx, &model, spawned, world, true);
        ctx.integration_parameters.num_solver_iterations = 50;

        // Set initial motor targets so the first physics step doesn't jerk
        // from zero to REST_POSE.
        for (i, name) in chain.joint_names().iter().enumerate() {
            if i >= config.initial_positions.len() {
                break;
            }
            let q0 = config.initial_positions[i];
            if let Some(&entity) = spawned.joints.get(*name)
                && let Some(&jh) = ctx.joint_handles.get(&entity)
                && let Some(joint) = ctx.impulse_joint_set.get_mut(jh, true)
            {
                let axis = if ctx
                    .joint_info
                    .get(&entity)
                    .is_some_and(|info| info.is_prismatic)
                {
                    rapier3d::prelude::JointAxis::LinX
                } else {
                    rapier3d::prelude::JointAxis::AngX
                };
                let max_f = if i < EFFORT_LIMITS.len() {
                    EFFORT_LIMITS[i]
                } else {
                    EFFORT_LIMITS[EFFORT_LIMITS.len() - 1]
                };
                joint
                    .data
                    .set_motor(axis, q0, 0.0, ARM_STIFFNESS, ARM_DAMPING);
                joint.data.set_motor_max_force(axis, max_f);
            }
        }

        world.insert_resource(ctx);
    }

    // 6. Map chain joint order to entities
    let arm_joint_names: Vec<String> = chain
        .joint_names()
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let spawned = &scene.robots["six_dof_arm"];
    let joint_entities: Vec<Entity> = chain
        .joint_names()
        .iter()
        .map(|name| {
            spawned
                .joint_entity(name)
                .unwrap_or_else(|| panic!("joint {name} not found in spawned robot"))
        })
        .collect();

    // 7. Build the JointLayout bound to every spawned arm joint.
    let joint_layout = {
        let spawned = &scene.robots["six_dof_arm"];
        let mut layout = model.to_layout();
        let entities: Vec<Entity> = layout
            .joints()
            .iter()
            .map(|spec| {
                spawned
                    .joint_entity(&spec.name)
                    .unwrap_or_else(|| panic!("joint {} not in spawned arm", spec.name))
            })
            .collect();
        layout.bind_entities(&entities);
        Arc::new(layout)
    };
    // Some bins (e.g. arm_gym) only register a sensor over the 6 arm
    // joints; build a chain-order layout for that path.
    let chain_layout = {
        let mut builder = clankers_core::layout::JointLayoutBuilder::default();
        for (i, name) in chain
            .joint_names()
            .iter()
            .take(config.sensor_dof.min(joint_entities.len()))
            .enumerate()
        {
            builder = builder.push(clankers_core::layout::JointSpec {
                name: (*name).to_string(),
                entity: None,
                joint_type: clankers_core::layout::JointKind::Revolute,
                limits: clankers_core::layout::JointSpecLimits::default(),
                axis: [0.0, 0.0, 1.0],
            });
            let _ = i;
        }
        let mut l = builder.build();
        let chain_entities: Vec<Entity> = joint_entities.iter().take(l.len()).copied().collect();
        l.bind_entities(&chain_entities);
        Arc::new(l)
    };

    // 8. Register sensors using the chain-order layout (matches the
    // pre-PR2 ArmApplicator action ordering).
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(chain_layout)), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    ArmSetup {
        scene,
        model,
        chain,
        joint_entities,
        arm_joint_names,
        joint_layout,
    }
}

// ---------------------------------------------------------------------------
// Shared IK control
// ---------------------------------------------------------------------------

/// 6 reachable workspace target positions for the arm end-effector.
#[must_use]
pub fn arm_ik_targets() -> Vec<Vector3<f32>> {
    vec![
        Vector3::new(0.3, 0.0, 0.5),  // forward
        Vector3::new(0.0, 0.3, 0.5),  // left
        Vector3::new(-0.3, 0.0, 0.5), // back
        Vector3::new(0.0, -0.3, 0.5), // right
        Vector3::new(0.2, 0.2, 0.7),  // up-left
        Vector3::new(0.0, 0.0, 0.91), // straight up (home)
    ]
}

/// Resource holding the IK chain, solver, and target-cycling state.
#[derive(Resource)]
pub struct ArmIkState {
    pub chain: KinematicChain,
    pub solver: DlsSolver,
    pub joint_entities: Vec<Entity>,
    pub targets: Vec<Vector3<f32>>,
    pub current_target: usize,
    pub steps_at_target: u32,
    pub steps_per_target: u32,
}

/// System that sets `JointCommand` from IK solutions each step.
#[allow(clippy::needless_pass_by_value)]
pub fn arm_ik_control_system(
    mut ik: ResMut<ArmIkState>,
    episode: Res<Episode>,
    mut query: Query<(&JointState, &mut JointCommand)>,
) {
    if !episode.is_running() {
        return;
    }

    // Advance target
    ik.steps_at_target += 1;
    if ik.steps_at_target >= ik.steps_per_target {
        ik.steps_at_target = 0;
        ik.current_target = (ik.current_target + 1) % ik.targets.len();
    }

    let target_pos = ik.targets[ik.current_target];
    let target = IkTarget::Position(target_pos);

    // Read current joint positions
    let mut q_current = Vec::with_capacity(ik.joint_entities.len());
    for &entity in &ik.joint_entities {
        if let Ok((state, _)) = query.get(entity) {
            q_current.push(state.position);
        }
    }

    if q_current.len() != ik.chain.dof() {
        return;
    }

    // Solve IK
    let result = ik.solver.solve(&ik.chain, &target, &q_current);

    // Write joint commands (position mode)
    for (i, &entity) in ik.joint_entities.iter().enumerate() {
        if let Ok((_, mut cmd)) = query.get_mut(entity) {
            cmd.value = result.joint_positions[i];
        }
    }

    // Print status every 10 steps
    if ik.steps_at_target.is_multiple_of(10) {
        let ee = ik.chain.forward_kinematics(&q_current);
        let err = (target_pos - ee.translation.vector).norm();
        println!(
            "  target [{:.2}, {:.2}, {:.2}]  ee [{:.3}, {:.3}, {:.3}]  err={:.4}m  conv={} iters={}",
            target_pos.x,
            target_pos.y,
            target_pos.z,
            ee.translation.x,
            ee.translation.y,
            ee.translation.z,
            err,
            result.converged,
            result.iterations,
        );
    }
}

/// Create a default DLS IK solver for the arm.
#[must_use]
pub const fn arm_ik_solver() -> DlsSolver {
    DlsSolver::new(DlsConfig {
        max_iterations: 100,
        position_tolerance: 1e-4,
        angle_tolerance: 1e-3,
        damping: 0.01,
    })
}

// ---------------------------------------------------------------------------
// Motor override constants & helpers
// ---------------------------------------------------------------------------

/// Per-joint effort limits for the 6 arm joints.
/// With `AccelerationBased` motor model these are max accelerations (rad/s²),
/// so they can be very high to make the arm essentially rigid.
pub const EFFORT_LIMITS: [f32; 6] = [5000.0, 5000.0, 3000.0, 2000.0, 1000.0, 500.0];

/// Default arm joint PD stiffness (acceleration per radian of error).
/// High values make the arm essentially rigid under gravity.
pub const ARM_STIFFNESS: f32 = 50_000.0;
/// Default arm joint PD damping (acceleration per rad/s of velocity error).
/// Slightly overdamped (ζ ≈ 1.1) to prevent oscillation.
pub const ARM_DAMPING: f32 = 500.0;

/// Default gripper finger travel in meters (prismatic range 0..0.03).
pub const FINGER_TRAVEL: f32 = 0.03;
/// Default gripper PD stiffness.
pub const GRIPPER_STIFFNESS: f32 = 500.0;
/// Default gripper PD damping.
pub const GRIPPER_DAMPING: f32 = 50.0;
/// Default gripper max force.
pub const GRIPPER_MAX_FORCE: f32 = 100.0;

/// Build [`MotorOverrides`] pre-populated with REST_POSE targets for all arm
/// and gripper joints.
///
/// This ensures motors are active from the very first physics step, preventing
/// the arm from collapsing under gravity before the controller starts.
/// Follows the Isaac Sim pattern: configure joint drives to hold initial
/// position before physics begins.
///
/// # Panics
///
/// Panics via [`Result::expect`] if the constructed overrides do not
/// cover every joint in `setup.joint_layout` — the WS2 PR2 promotion of
/// the "ALL joints must be overridden" invariant from prose to a
/// runtime check (see [`validate_motor_coverage`] and MEMORY.md
/// "MotorOverrides — ALL Joints Must Be Overridden").
#[must_use]
pub fn initial_motor_overrides(setup: &ArmSetup, gripper_entities: &[Entity]) -> MotorOverrides {
    let mut overrides = MotorOverrides {
        layout: Some(setup.joint_layout.clone()),
        ..MotorOverrides::default()
    };

    // Arm joints: hold at REST_POSE
    for (i, &entity) in setup.joint_entities.iter().enumerate() {
        if i >= 6 {
            break;
        }
        overrides.joints.insert(
            entity,
            MotorOverrideParams {
                target_pos: REST_POSE[i],
                target_vel: 0.0,
                stiffness: ARM_STIFFNESS,
                damping: ARM_DAMPING,
                max_force: EFFORT_LIMITS[i],
            },
        );
    }

    // Gripper fingers: hold open
    for &entity in gripper_entities {
        overrides.joints.insert(
            entity,
            MotorOverrideParams {
                target_pos: FINGER_TRAVEL,
                target_vel: 0.0,
                stiffness: GRIPPER_STIFFNESS,
                damping: GRIPPER_DAMPING,
                max_force: GRIPPER_MAX_FORCE,
            },
        );
    }

    // PR2 invariant: every layout joint must have an override entry.
    // Without this check, missing joints silently fall through to the
    // ZOH torque motor trick and the arm flails wildly.
    validate_motor_coverage(&RobotGroup::default(), &setup.joint_layout, &overrides)
        .expect("initial_motor_overrides missing joint coverage — see validate_motor_coverage");

    overrides
}
