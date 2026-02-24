//! URDF-to-Rapier bridge: converts [`RobotModel`] into rapier rigid bodies
//! and impulse joints, inserting physics marker components on entities.

use std::collections::{HashMap, VecDeque};

use bevy::prelude::{Vec3, World};
use rapier3d::prelude::{
    FixedJointBuilder, GenericJoint, ImpulseJointHandle, JointAxis, MassProperties, MotorModel,
    PrismaticJointBuilder, RevoluteJointBuilder, RigidBodyBuilder,
};

use clankers_urdf::spawner::SpawnedRobot;
use clankers_urdf::types::{JointType, RobotModel};

use crate::components::{PhysicsBody, PhysicsJoint};

use super::context::{JointInfo, RapierContext};

// ---------------------------------------------------------------------------
// register_robot
// ---------------------------------------------------------------------------

/// Register a robot's links and joints with the rapier physics context.
///
/// Creates rigid bodies for each link and impulse joints for each actuated
/// joint. Inserts [`PhysicsBody`] and [`PhysicsJoint`] marker components
/// on the corresponding Bevy entities.
pub fn register_robot(
    context: &mut RapierContext,
    model: &RobotModel,
    spawned: &SpawnedRobot,
    world: &mut World,
    fixed_base: bool,
) {
    // 1. Create root link rigid body
    let root_body = if fixed_base {
        RigidBodyBuilder::fixed()
    } else {
        RigidBodyBuilder::dynamic()
    };
    let root_handle = context.rigid_body_set.insert(root_body.build());
    context
        .body_handles
        .insert(model.root_link.clone(), root_handle);

    // Track accumulated world positions through the kinematic chain.
    // URDF joint origins are relative to the parent link, so each child's
    // world position = parent world position + joint origin offset.
    let mut link_world_pos: HashMap<String, Vec3> = HashMap::new();
    link_world_pos.insert(model.root_link.clone(), Vec3::ZERO);

    // 2. BFS through the kinematic tree
    let mut queue = VecDeque::new();
    queue.push_back(model.root_link.clone());

    while let Some(parent_link_name) = queue.pop_front() {
        let parent_world = link_world_pos[&parent_link_name];

        for joint_data in model.joints.values() {
            if joint_data.parent != parent_link_name {
                continue;
            }

            let child_link_name = &joint_data.child;
            let origin_pos = Vec3::new(
                joint_data.origin.xyz[0],
                joint_data.origin.xyz[1],
                joint_data.origin.xyz[2],
            );

            // Child world position = parent world + joint origin offset
            let child_world = parent_world + origin_pos;
            link_world_pos.insert(child_link_name.clone(), child_world);

            // Create child rigid body at accumulated world position
            let child_link = model.links.get(child_link_name);
            let child_body = create_link_body(child_link, child_world);
            let child_handle = context.rigid_body_set.insert(child_body);
            context
                .body_handles
                .insert(child_link_name.clone(), child_handle);

            let parent_handle = context.body_handles[&parent_link_name];

            // Create rapier joint for actuated joints
            if joint_data.joint_type.is_actuated() {
                let axis = Vec3::new(
                    joint_data.axis[0],
                    joint_data.axis[1],
                    joint_data.axis[2],
                );

                let rapier_joint = build_rapier_joint(
                    joint_data.joint_type,
                    axis,
                    origin_pos,
                    joint_data.limits.lower,
                    joint_data.limits.upper,
                );

                let joint_handle: ImpulseJointHandle = context.impulse_joint_set.insert(
                    parent_handle,
                    child_handle,
                    rapier_joint,
                    true,
                );

                // Store mapping if this joint has an ECS entity
                if let Some(&entity) = spawned.joints.get(&joint_data.name) {
                    context.joint_handles.insert(entity, joint_handle);
                    context.joint_info.insert(
                        entity,
                        JointInfo {
                            parent_body: parent_handle,
                            child_body: child_handle,
                            axis,
                            is_prismatic: joint_data.joint_type == JointType::Prismatic,
                        },
                    );

                    // Insert marker components
                    world.entity_mut(entity).insert((
                        PhysicsBody::Dynamic,
                        PhysicsJoint {
                            parent_body: entity,
                            child_body: entity,
                        },
                    ));
                }
            } else if joint_data.joint_type == JointType::Fixed {
                let fixed: GenericJoint = FixedJointBuilder::new()
                    .local_anchor1(origin_pos)
                    .build()
                    .into();
                context
                    .impulse_joint_set
                    .insert(parent_handle, child_handle, fixed, true);
            }

            queue.push_back(child_link_name.clone());
        }
    }

    // Snapshot initial body positions for episode reset.
    context.snapshot_initial_state();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a dynamic rigid body for a URDF link.
fn create_link_body(
    link: Option<&clankers_urdf::types::LinkData>,
    position: Vec3,
) -> rapier3d::prelude::RigidBody {
    let mut builder = rapier3d::prelude::RigidBodyBuilder::dynamic()
        .translation(position)
        .can_sleep(false);

    if let Some(link) = link {
        if let Some(ref inertial) = link.inertial {
            if inertial.mass > 0.0 {
                builder = builder.additional_mass_properties(MassProperties::new(
                    Vec3::new(
                        inertial.origin.xyz[0],
                        inertial.origin.xyz[1],
                        inertial.origin.xyz[2],
                    ),
                    inertial.mass,
                    Vec3::new(
                        inertial.inertia[0], // ixx
                        inertial.inertia[3], // iyy
                        inertial.inertia[5], // izz
                    ),
                ));
            }
        }
    }

    builder.build()
}

/// Build a rapier `GenericJoint` from URDF joint data.
fn build_rapier_joint(
    joint_type: JointType,
    axis: Vec3,
    anchor: Vec3,
    lower: Option<f32>,
    upper: Option<f32>,
) -> GenericJoint {
    match joint_type {
        JointType::Revolute | JointType::Continuous => {
            let mut joint: GenericJoint = RevoluteJointBuilder::new(axis)
                .local_anchor1(anchor)
                .build()
                .into();

            if joint_type == JointType::Revolute {
                if let (Some(lo), Some(hi)) = (lower, upper) {
                    joint.set_limits(JointAxis::AngX, [lo, hi]);
                }
            }

            joint.set_motor_model(JointAxis::AngX, MotorModel::ForceBased);
            joint.set_motor(JointAxis::AngX, 0.0, 0.0, 0.0, 0.0);
            joint
        }
        JointType::Prismatic => {
            let mut joint: GenericJoint = PrismaticJointBuilder::new(axis)
                .local_anchor1(anchor)
                .build()
                .into();

            if let (Some(lo), Some(hi)) = (lower, upper) {
                joint.set_limits(JointAxis::LinX, [lo, hi]);
            }

            joint.set_motor_model(JointAxis::LinX, MotorModel::ForceBased);
            joint.set_motor(JointAxis::LinX, 0.0, 0.0, 0.0, 0.0);
            joint
        }
        _ => {
            FixedJointBuilder::new()
                .local_anchor1(anchor)
                .build()
                .into()
        }
    }
}
