//! Bevy entity spawning from a parsed [`RobotModel`].
//!
//! Creates one entity per actuated joint with [`Actuator`], [`JointCommand`],
//! [`JointState`], and [`JointTorque`] components.

use std::collections::HashMap;
use std::hash::BuildHasher;

use bevy::prelude::*;
use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
use clankers_actuator_core::motor::IdealMotor;
use clankers_actuator_core::prelude::MotorType;

use crate::types::{JointData, RobotModel};

// ---------------------------------------------------------------------------
// JointName component
// ---------------------------------------------------------------------------

/// Component storing the URDF joint name on a joint entity.
#[derive(Component, Clone, Debug)]
pub struct JointName(pub String);

// ---------------------------------------------------------------------------
// SpawnedRobot
// ---------------------------------------------------------------------------

/// Result of spawning a robot — maps joint names to their Bevy entities.
#[derive(Debug, Clone)]
pub struct SpawnedRobot {
    /// Robot name from the URDF.
    pub name: String,
    /// Map from joint name to the spawned entity ID.
    pub joints: HashMap<String, Entity>,
}

impl SpawnedRobot {
    /// Get the entity for a joint by name.
    pub fn joint_entity(&self, name: &str) -> Option<Entity> {
        self.joints.get(name).copied()
    }

    /// Number of spawned joint entities.
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }
}

// ---------------------------------------------------------------------------
// spawn_robot
// ---------------------------------------------------------------------------

/// Spawn actuated joint entities for a [`RobotModel`].
///
/// Creates one entity per actuated joint (revolute, continuous, prismatic).
/// Each entity receives [`Actuator`], [`JointCommand`], [`JointState`],
/// [`JointTorque`], and [`JointName`] components.
///
/// The [`Actuator`] is configured from the joint's URDF limits:
/// - `effort` → `IdealMotor::max_torque`
/// - `velocity` → `IdealMotor::max_velocity`
/// - `dynamics.damping` → `FrictionModel::viscous`
/// - `dynamics.friction` → `FrictionModel::coulomb`
///
/// Initial joint positions can be provided via `initial_positions`.
pub fn spawn_robot<S: BuildHasher>(
    world: &mut World,
    model: &RobotModel,
    initial_positions: &HashMap<String, f32, S>,
) -> SpawnedRobot {
    let mut joints = HashMap::new();

    for joint in model.actuated_joints() {
        let entity = spawn_joint_entity(world, joint, initial_positions);
        joints.insert(joint.name.clone(), entity);
    }

    SpawnedRobot {
        name: model.name.clone(),
        joints,
    }
}

fn spawn_joint_entity<S: BuildHasher>(
    world: &mut World,
    joint: &JointData,
    initial_positions: &HashMap<String, f32, S>,
) -> Entity {
    let motor = MotorType::Ideal(IdealMotor::new(
        joint.limits.effort.max(1.0),
        joint.limits.velocity.max(1.0),
    ));

    let friction = clankers_actuator_core::friction::FrictionModel {
        coulomb: joint.dynamics.friction,
        viscous: joint.dynamics.damping,
        ..Default::default()
    };

    let actuator = Actuator::default().with_friction(friction);

    // Override motor via the public field
    let mut actuator = actuator;
    actuator.motor = motor;

    let position = initial_positions.get(&joint.name).copied().unwrap_or(0.0);

    world
        .spawn((
            JointName(joint.name.clone()),
            actuator,
            JointCommand::default(),
            JointState {
                position,
                velocity: 0.0,
            },
            JointTorque::default(),
        ))
        .id()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_string;

    const ARM_URDF: &str = r#"
        <robot name="arm">
            <link name="base"/>
            <link name="link1"/>
            <link name="link2"/>
            <link name="link3"/>
            <joint name="shoulder" type="revolute">
                <parent link="base"/>
                <child link="link1"/>
                <axis xyz="0 0 1"/>
                <limit lower="-1.57" upper="1.57" effort="50" velocity="3"/>
                <dynamics damping="0.5" friction="0.1"/>
            </joint>
            <joint name="elbow" type="revolute">
                <parent link="link1"/>
                <child link="link2"/>
                <axis xyz="0 1 0"/>
                <limit lower="-2.0" upper="2.0" effort="30" velocity="5"/>
            </joint>
            <joint name="wrist_fixed" type="fixed">
                <parent link="link2"/>
                <child link="link3"/>
            </joint>
        </robot>
    "#;

    #[test]
    fn spawn_creates_actuated_joints_only() {
        let model = parse_string(ARM_URDF).unwrap();
        let mut world = World::new();
        let result = spawn_robot(&mut world, &model, &HashMap::new());

        assert_eq!(result.name, "arm");
        assert_eq!(result.joint_count(), 2); // shoulder + elbow, not wrist_fixed
        assert!(result.joint_entity("shoulder").is_some());
        assert!(result.joint_entity("elbow").is_some());
        assert!(result.joint_entity("wrist_fixed").is_none());
    }

    #[test]
    fn spawned_entities_have_all_components() {
        let model = parse_string(ARM_URDF).unwrap();
        let mut world = World::new();
        let result = spawn_robot(&mut world, &model, &HashMap::new());

        for entity in result.joints.values() {
            assert!(world.get::<Actuator>(*entity).is_some());
            assert!(world.get::<JointCommand>(*entity).is_some());
            assert!(world.get::<JointState>(*entity).is_some());
            assert!(world.get::<JointTorque>(*entity).is_some());
            assert!(world.get::<JointName>(*entity).is_some());
        }
    }

    #[test]
    fn joint_name_component_matches() {
        let model = parse_string(ARM_URDF).unwrap();
        let mut world = World::new();
        let result = spawn_robot(&mut world, &model, &HashMap::new());

        let entity = result.joint_entity("shoulder").unwrap();
        let name = world.get::<JointName>(entity).unwrap();
        assert_eq!(name.0, "shoulder");
    }

    #[test]
    fn initial_positions_applied() {
        let model = parse_string(ARM_URDF).unwrap();
        let mut world = World::new();
        let mut positions = HashMap::new();
        positions.insert("shoulder".into(), 0.5);
        positions.insert("elbow".into(), -1.0);

        let result = spawn_robot(&mut world, &model, &positions);

        let shoulder = result.joint_entity("shoulder").unwrap();
        let state = world.get::<JointState>(shoulder).unwrap();
        assert!((state.position - 0.5).abs() < f32::EPSILON);

        let elbow = result.joint_entity("elbow").unwrap();
        let state = world.get::<JointState>(elbow).unwrap();
        assert!((state.position - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn default_position_is_zero() {
        let model = parse_string(ARM_URDF).unwrap();
        let mut world = World::new();
        let result = spawn_robot(&mut world, &model, &HashMap::new());

        let entity = result.joint_entity("shoulder").unwrap();
        let state = world.get::<JointState>(entity).unwrap();
        assert!(state.position.abs() < f32::EPSILON);
        assert!(state.velocity.abs() < f32::EPSILON);
    }

    #[test]
    fn motor_configured_from_limits() {
        let model = parse_string(ARM_URDF).unwrap();
        let mut world = World::new();
        let result = spawn_robot(&mut world, &model, &HashMap::new());

        let entity = result.joint_entity("shoulder").unwrap();
        let actuator = world.get::<Actuator>(entity).unwrap();
        if let MotorType::Ideal(motor) = &actuator.motor {
            assert!((motor.max_torque - 50.0).abs() < f32::EPSILON);
            assert!((motor.max_velocity - 3.0).abs() < f32::EPSILON);
        } else {
            panic!("expected IdealMotor");
        }
    }

    #[test]
    fn friction_configured_from_dynamics() {
        let model = parse_string(ARM_URDF).unwrap();
        let mut world = World::new();
        let result = spawn_robot(&mut world, &model, &HashMap::new());

        let entity = result.joint_entity("shoulder").unwrap();
        let actuator = world.get::<Actuator>(entity).unwrap();
        assert!((actuator.friction.viscous - 0.5).abs() < f32::EPSILON);
        assert!((actuator.friction.coulomb - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn spawned_robot_entity_lookup() {
        let model = parse_string(ARM_URDF).unwrap();
        let mut world = World::new();
        let result = spawn_robot(&mut world, &model, &HashMap::new());

        assert!(result.joint_entity("nonexistent").is_none());
        assert!(result.joint_entity("shoulder").is_some());
    }
}
