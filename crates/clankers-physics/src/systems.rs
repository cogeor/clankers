//! Placeholder system stubs for physics integration.
//!
//! These will be fleshed out in Loop 02 when the Rapier backend is implemented.
//! They exist here to define the system signatures that backends will register
//! in [`ClankersSet::Simulate`](clankers_core::ClankersSet::Simulate).

use bevy::prelude::*;
use clankers_actuator::components::{JointState, JointTorque};

use crate::components::PhysicsJoint;

/// Read [`JointTorque`] from actuator entities and apply to the physics engine.
///
/// This system runs in `ClankersSet::Simulate` and bridges the actuator
/// pipeline output to the physics backend.
#[allow(unused_variables)]
pub fn apply_joint_torques(query: Query<(&PhysicsJoint, &JointTorque)>) {
    // Stub: will be implemented by the concrete backend in Loop 02.
}

/// Read physics engine joint state and write [`JointState`] on actuator entities.
///
/// This system runs in `ClankersSet::Simulate` after the physics step,
/// feeding position/velocity back to the actuator pipeline.
#[allow(unused_variables)]
pub fn read_joint_states(query: Query<(&PhysicsJoint, &mut JointState)>) {
    // Stub: will be implemented by the concrete backend in Loop 02.
}
