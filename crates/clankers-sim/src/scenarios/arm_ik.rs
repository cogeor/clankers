//! `arm_ik` scenario — headless 6-DOF arm scene for IK-driven control.
//!
//! Lifts the headless subset of `examples/src/arm_setup.rs::setup_arm`
//! into a [`ScenarioBuilder`]: URDF parse → [`crate::SceneBuilder`] →
//! position-mode PID actuators → Rapier physics with
//! `num_solver_iterations = 50` → initial motor targets → bound
//! [`clankers_core::layout::JointLayout`] →
//! [`JointStateSensor`].
//!
//! IK logic (chain solver, target cycling, control system) intentionally
//! stays in the example bin — `clankers-sim` does NOT depend on
//! `clankers-ik`. The scenario hands callers everything they need
//! (joint entities, bound layout) to wire IK from above.
//!
//! # MEMORY.md invariants
//!
//! - `num_solver_iterations = 50` (W8 PR1 gate item 10).
//! - Arm joints: stiffness 50000, damping 500, `AccelerationBased`.
//! - The scenario does NOT pre-populate
//!   [`MotorOverrides`](clankers_physics::rapier::MotorOverrides);
//!   callers wanting overrides build them via
//!   `examples::arm_setup::initial_motor_overrides` for now.

use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
use std::sync::Arc;

use bevy::prelude::*;
use clankers_actuator::components::Actuator;
use clankers_actuator_core::prelude::ControlMode;
use clankers_env::prelude::*;
use clankers_physics::ClankersPhysicsPlugin;
use clankers_physics::rapier::{
    RapierBackend, RapierBackendFixed, RapierContext, bridge::register_robot,
};

use crate::SceneBuilder;
use crate::SpawnedScene;
use crate::scenarios::{ScenarioBuilder, ScenarioConfig, ScenarioHandle};

/// URDF source for the 6-DOF arm + gripper.
const SIX_DOF_ARM_URDF: &str = include_str!("../../../../examples/urdf/six_dof_arm.urdf");

/// Default resting pose for the 6-DOF arm (j2 tilts shoulder 45°,
/// j3 bends elbow 90°, j5 pitches wrist down).
pub const REST_POSE: [f32; 6] = [0.0, FRAC_PI_4, FRAC_PI_2, 0.0, FRAC_PI_4, 0.0];

/// Per-joint effort limits for the 6 arm joints (`AccelerationBased`
/// motor model — these are max accelerations).
pub const EFFORT_LIMITS: [f32; 6] = [5000.0, 5000.0, 3000.0, 2000.0, 1000.0, 500.0];

/// Arm joint PD stiffness (acceleration per radian of error).
pub const ARM_STIFFNESS: f32 = 50_000.0;
/// Arm joint PD damping (acceleration per rad/s of velocity error).
pub const ARM_DAMPING: f32 = 500.0;

/// Joint names in chain (URDF kinematic) order. Matches the IK chain
/// order built by `clankers_ik::KinematicChain::from_model`.
pub const ARM_JOINT_NAMES: [&str; 6] = [
    "j1_base_yaw",
    "j2_shoulder_pitch",
    "j3_elbow_pitch",
    "j4_forearm_roll",
    "j5_wrist_pitch",
    "j6_wrist_roll",
];

/// Per-scenario knobs for the `arm_ik` scenario.
///
/// Per W8 PR1 Design choice B, none of these belong on the field-locked
/// [`ScenarioConfig`]; the scenario owns its own config struct consumed
/// by [`ArmIkScenario::build_with_artifacts`].
#[derive(Debug, Clone)]
pub struct ArmIkConfig {
    /// Hard cap on episode steps. Loops 7+ pin this to `max_steps` if
    /// the caller doesn't override it.
    pub max_episode_steps: u32,
    /// When `true`, install [`RapierBackendFixed`] (physics runs on
    /// `FixedUpdate`); otherwise install [`RapierBackend`] (physics on
    /// `Update`).
    pub use_fixed_update: bool,
    /// Sensor DOF: 6 for the arm joints only, 8 to include the gripper
    /// fingers. The sensor is registered over a chain-order
    /// [`clankers_core::layout::JointLayout`] — matches the pre-PR1
    /// `ArmApplicator` action ordering.
    pub sensor_dof: usize,
    /// Initial joint positions for the 6 arm joints. Defaults to
    /// [`REST_POSE`].
    pub initial_positions: [f32; 6],
}

impl Default for ArmIkConfig {
    fn default() -> Self {
        Self {
            max_episode_steps: 500,
            use_fixed_update: false,
            sensor_dof: 6,
            initial_positions: REST_POSE,
        }
    }
}

/// Handle returned by [`ArmIkScenario::build_with_artifacts`] alongside
/// the standard [`ScenarioHandle`]. Carries the data callers need to
/// drive IK or layout-bound sensors from above.
pub struct ArmIkArtifacts {
    /// The spawned-scene wrapper (re-exported through the handle for
    /// callers that need direct world access).
    pub scene: SpawnedScene,
    /// The parsed URDF model (cheap to clone — `RobotModel` is small).
    pub model: clankers_urdf::RobotModel,
    /// Chain-order joint entities (the 6 arm joints; index = chain pos).
    pub arm_joint_entities: Vec<Entity>,
    /// Bound layout over **all** spawned joint entities (arm + gripper
    /// fingers when present), suitable for layout-bound sensors and
    /// [`clankers_physics::rapier::systems::validate_motor_coverage`].
    pub joint_layout: Arc<clankers_core::layout::JointLayout>,
}

/// Builder for the `arm_ik` scenario.
///
/// Implements [`ScenarioBuilder`]; registered into the registry by
/// [`super::register_builtin`] alongside the rest of the arm family.
pub struct ArmIkScenario;

impl ScenarioBuilder for ArmIkScenario {
    fn name(&self) -> &'static str {
        "arm_ik"
    }

    fn build(&self, app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle {
        let ik_cfg = ArmIkConfig {
            max_episode_steps: cfg.max_steps,
            ..ArmIkConfig::default()
        };
        let artefacts = Self::build_with_artifacts(cfg, &ik_cfg);
        let handle = ScenarioHandle {
            layout: Some(artefacts.joint_layout),
            max_steps: cfg.max_steps,
        };
        // Move the SceneBuilder-owned App into the caller's slot.
        let mut scene = artefacts.scene;
        std::mem::swap(app, &mut scene.app);
        handle
    }
}

impl ArmIkScenario {
    /// Parametrised build path used by `arm_ik.rs`, `arm_gym.rs`,
    /// `arm_ik_viz.rs`. Returns the assembled artefacts so the caller
    /// can install IK + control systems before draining the scene.
    #[must_use]
    #[allow(clippy::too_many_lines)] // single linear setup pipeline, matches arm_pick.rs precedent
    pub fn build_with_artifacts(cfg: &ScenarioConfig, ik_cfg: &ArmIkConfig) -> ArmIkArtifacts {
        let _ = cfg.seed; // honoured by the caller's RNG; nothing random here
        let model = clankers_urdf::parse_string(SIX_DOF_ARM_URDF)
            .expect("failed to parse six_dof_arm URDF (compile-time include)");

        // 1. Initial positions map (chain order; only arm joints get
        //    rest values, fingers start at 0).
        let initial_positions: HashMap<String, f32> = ARM_JOINT_NAMES
            .iter()
            .zip(ik_cfg.initial_positions.iter())
            .map(|(name, &pos)| ((*name).to_string(), pos))
            .collect();

        // 2. Build the scene.
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(ik_cfg.max_episode_steps.max(1))
            .with_robot(model.clone(), initial_positions)
            .build();

        // 3. Switch all actuators (arm + gripper) to position-mode PID.
        let spawned = &scene.robots["six_dof_arm"];
        let joint_entity_ids: Vec<Entity> = spawned.joints.values().copied().collect();
        for entity in &joint_entity_ids {
            let mut actuator = scene
                .app
                .world_mut()
                .get_mut::<Actuator>(*entity)
                .expect("Actuator component on spawned joint");
            *actuator = Actuator::new(
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

        // 4. Add Rapier physics (fixed-update variant if requested).
        if ik_cfg.use_fixed_update {
            scene
                .app
                .add_plugins(ClankersPhysicsPlugin::new(RapierBackendFixed));
        } else {
            scene
                .app
                .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));
        }

        // 5. Register robot bodies + set initial motor targets +
        //    `num_solver_iterations = 50` (MEMORY.md invariant).
        {
            let spawned = &scene.robots["six_dof_arm"];
            let world = scene.app.world_mut();
            let mut ctx = world
                .remove_resource::<RapierContext>()
                .expect("RapierContext present after ClankersPhysicsPlugin");
            register_robot(&mut ctx, &model, spawned, world, true);
            ctx.integration_parameters.num_solver_iterations = 50;

            for (i, name) in ARM_JOINT_NAMES.iter().enumerate() {
                let q0 = ik_cfg.initial_positions[i];
                let Some(&entity) = spawned.joints.get(*name) else {
                    continue;
                };
                let Some(&jh) = ctx.joint_handles.get(&entity) else {
                    continue;
                };
                let Some(joint) = ctx.impulse_joint_set.get_mut(jh, true) else {
                    continue;
                };
                let axis = if ctx
                    .joint_info
                    .get(&entity)
                    .is_some_and(|info| info.is_prismatic)
                {
                    rapier3d::prelude::JointAxis::LinX
                } else {
                    rapier3d::prelude::JointAxis::AngX
                };
                let max_f = EFFORT_LIMITS.get(i).copied().unwrap_or(500.0);
                joint
                    .data
                    .set_motor(axis, q0, 0.0, ARM_STIFFNESS, ARM_DAMPING);
                joint.data.set_motor_max_force(axis, max_f);
            }

            world.insert_resource(ctx);
        }

        // 6. Map chain joint order to entities (the arm-only 6 joints).
        let spawned = &scene.robots["six_dof_arm"];
        let arm_joint_entities: Vec<Entity> = ARM_JOINT_NAMES
            .iter()
            .map(|name| {
                spawned
                    .joint_entity(name)
                    .unwrap_or_else(|| panic!("joint {name} not found in spawned arm"))
            })
            .collect();

        // 7. Build the JointLayout bound to every spawned arm joint
        //    (arm + gripper fingers when present).
        let joint_layout = {
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

        // 8. Chain-order sensor layout (matches pre-PR1 ArmApplicator
        //    action ordering). Sized to `sensor_dof`.
        let sensor_dof = ik_cfg.sensor_dof.min(arm_joint_entities.len());
        let chain_layout = {
            let mut builder = clankers_core::layout::JointLayoutBuilder::default();
            for name in ARM_JOINT_NAMES.iter().take(sensor_dof) {
                builder = builder.push(clankers_core::layout::JointSpec {
                    name: (*name).to_string(),
                    entity: None,
                    joint_type: clankers_core::layout::JointKind::Revolute,
                    limits: clankers_core::layout::JointSpecLimits::default(),
                    axis: [0.0, 0.0, 1.0],
                });
            }
            let mut l = builder.build();
            let chain_entities: Vec<Entity> =
                arm_joint_entities.iter().take(l.len()).copied().collect();
            l.bind_entities(&chain_entities);
            Arc::new(l)
        };

        // 9. Register the joint-state sensor on the chain-order layout.
        {
            let world = scene.app.world_mut();
            let mut registry = world
                .remove_resource::<SensorRegistry>()
                .expect("SensorRegistry present after ClankersEnvPlugin");
            let mut buffer = world
                .remove_resource::<ObservationBuffer>()
                .expect("ObservationBuffer present after ClankersEnvPlugin");
            registry.register(Box::new(JointStateSensor::new(chain_layout)), &mut buffer);
            world.insert_resource(buffer);
            world.insert_resource(registry);
        }

        ArmIkArtifacts {
            scene,
            model,
            arm_joint_entities,
            joint_layout,
        }
    }
}
