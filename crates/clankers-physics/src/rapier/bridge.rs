//! URDF-to-Rapier bridge: converts [`RobotModel`] into rapier rigid bodies
//! and impulse joints, inserting physics marker components on entities.

use std::collections::{HashMap, VecDeque};

use bevy::prelude::{EulerRot, Quat, Vec3, World};
use rapier3d::dynamics::{GenericJointBuilder, JointAxesMask};
use rapier3d::prelude::{
    ColliderBuilder, GenericJoint, ImpulseJointHandle, JointAxis, MassProperties, MotorModel,
    RigidBodyBuilder,
};

use clankers_urdf::spawner::SpawnedRobot;
use clankers_urdf::types::{Geometry, JointType, RobotModel};

use clankers_core::physics::ContactData;

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
///
/// Initial joint positions from the spawned robot's `JointState` components
/// are used to compute FK, so rigid bodies are placed at the correct
/// initial configuration (not zero-angle). This prevents the first physics
/// step from generating large motor forces to drive from zero to the
/// desired pose.
#[allow(clippy::too_many_lines)]
pub fn register_robot(
    context: &mut RapierContext,
    model: &RobotModel,
    spawned: &SpawnedRobot,
    world: &mut World,
    fixed_base: bool,
) {
    // 1. Create root link rigid body
    let root_body_type = if fixed_base {
        PhysicsBody::Fixed
    } else {
        PhysicsBody::Dynamic
    };
    let root_body = if fixed_base {
        RigidBodyBuilder::fixed()
    } else {
        RigidBodyBuilder::dynamic()
    };
    let root_handle = context.rigid_body_set.insert(root_body.build());
    context
        .body_handles
        .insert(model.root_link.clone(), root_handle);

    // Create colliders for the root link.
    if let Some(root_link) = model.links.get(&model.root_link) {
        create_link_colliders(context, root_handle, root_link, &Quat::IDENTITY);
    }

    // Spawn a root entity so PhysicsJoint can reference it as parent_body.
    let root_entity = world.spawn((root_body_type, ContactData::default())).id();

    // Track link name → ECS entity for PhysicsJoint parent/child fields.
    let mut link_to_entity: HashMap<String, bevy::prelude::Entity> = HashMap::new();
    link_to_entity.insert(model.root_link.clone(), root_entity);

    // Track accumulated world poses through the kinematic chain.
    // URDF joint origins are relative to the parent link, so each child's
    // world pose = parent world pose * joint origin transform.
    let mut link_world_pos: HashMap<String, Vec3> = HashMap::new();
    let mut link_world_rot: HashMap<String, Quat> = HashMap::new();
    link_world_pos.insert(model.root_link.clone(), Vec3::ZERO);
    link_world_rot.insert(model.root_link.clone(), Quat::IDENTITY);

    // 2. BFS through the kinematic tree
    let mut queue = VecDeque::new();
    queue.push_back(model.root_link.clone());

    while let Some(parent_link_name) = queue.pop_front() {
        let parent_world_pos = link_world_pos[&parent_link_name];
        let parent_world_rot = link_world_rot[&parent_link_name];

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
            let origin_rpy = joint_data.origin.rpy;
            let origin_rot =
                Quat::from_euler(EulerRot::ZYX, origin_rpy[2], origin_rpy[1], origin_rpy[0]);

            // Read initial joint position from JointState (set by URDF spawner).
            // This lets us place bodies at the correct initial FK pose instead
            // of zero-angle, avoiding a violent first-step motor correction.
            let initial_q = spawned
                .joints
                .get(&joint_data.name)
                .and_then(|&entity| {
                    world
                        .get::<clankers_actuator::components::JointState>(entity)
                        .map(|s| s.position)
                })
                .unwrap_or(0.0);

            // Apply initial joint angle: rotate around the joint axis
            let joint_axis = Vec3::new(joint_data.axis[0], joint_data.axis[1], joint_data.axis[2]);
            let joint_rotation = if joint_data.joint_type.is_actuated()
                && joint_data.joint_type != JointType::Prismatic
                && initial_q.abs() > f32::EPSILON
            {
                Quat::from_axis_angle(joint_axis, initial_q)
            } else {
                Quat::IDENTITY
            };

            // Child world pose = parent * joint_origin * joint_rotation
            // For prismatic joints, add translation along joint axis instead.
            let joint_frame_pos = parent_world_pos + parent_world_rot * origin_pos;
            let joint_frame_rot = parent_world_rot * origin_rot * joint_rotation;

            let child_world_pos = if joint_data.joint_type == JointType::Prismatic
                && initial_q.abs() > f32::EPSILON
            {
                joint_frame_pos + parent_world_rot * (joint_axis * initial_q)
            } else {
                joint_frame_pos
            };
            let child_world_rot = joint_frame_rot;

            link_world_pos.insert(child_link_name.clone(), child_world_pos);
            link_world_rot.insert(child_link_name.clone(), child_world_rot);

            // Create child rigid body at accumulated world pose
            let child_link = model.links.get(child_link_name);
            let child_body = create_link_body(child_link, child_world_pos, child_world_rot);
            let child_handle = context.rigid_body_set.insert(child_body);
            context
                .body_handles
                .insert(child_link_name.clone(), child_handle);

            // Create colliders from URDF collision geometry.
            if let Some(link) = child_link {
                create_link_colliders(context, child_handle, link, &child_world_rot);
            }

            let parent_handle = context.body_handles[&parent_link_name];

            // Create rapier joint for actuated joints
            if joint_data.joint_type.is_actuated() {
                let axis = Vec3::new(joint_data.axis[0], joint_data.axis[1], joint_data.axis[2]);

                let rapier_joint = build_rapier_joint(
                    joint_data.joint_type,
                    axis,
                    origin_pos,
                    &origin_rpy,
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
                    context.body_to_entity.insert(child_handle, entity);

                    // Map child link name to this entity for downstream PhysicsJoint lookups.
                    link_to_entity.insert(child_link_name.clone(), entity);

                    // Look up parent body entity from link_to_entity mapping.
                    let parent_entity = link_to_entity
                        .get(&parent_link_name)
                        .copied()
                        .unwrap_or(entity);

                    // Insert marker components
                    world.entity_mut(entity).insert((
                        PhysicsBody::Dynamic,
                        PhysicsJoint {
                            parent_body: parent_entity,
                            child_body: entity,
                        },
                        ContactData::default(),
                    ));
                }
            } else if joint_data.joint_type == JointType::Fixed {
                let fixed = build_rapier_joint(
                    JointType::Fixed,
                    Vec3::ZERO,
                    origin_pos,
                    &origin_rpy,
                    None,
                    None,
                );
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
    rotation: Quat,
) -> rapier3d::prelude::RigidBody {
    let mut builder = rapier3d::prelude::RigidBodyBuilder::dynamic()
        .translation(position)
        .rotation(quat_to_angvector(rotation))
        .can_sleep(false);

    if let Some(link) = link
        && let Some(ref inertial) = link.inertial
        && inertial.mass > 0.0
    {
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

    builder.build()
}

/// Convert a bevy `Quat` to a Rapier `AngVector` (scaled-axis representation).
fn quat_to_angvector(q: Quat) -> Vec3 {
    let (axis, angle) = q.to_axis_angle();
    if angle.abs() < f32::EPSILON {
        Vec3::ZERO
    } else {
        axis * angle
    }
}

/// Build a rapier `GenericJoint` from URDF joint data.
///
/// Uses `GenericJointBuilder` with explicit anchor/axis setup, matching the
/// approach from the official `rapier3d-urdf` crate. This avoids the pitfall
/// of `RevoluteJointBuilder::new(axis)` + `set_local_frame1()` which
/// overwrites the internal axis frames and causes constraint explosions.
fn build_rapier_joint(
    joint_type: JointType,
    axis: Vec3,
    origin_pos: Vec3,
    origin_rpy: &[f32; 3],
    lower: Option<f32>,
    upper: Option<f32>,
) -> GenericJoint {
    let origin_rot = Quat::from_euler(EulerRot::ZYX, origin_rpy[2], origin_rpy[1], origin_rpy[0]);

    let locked_axes = match joint_type {
        JointType::Revolute | JointType::Continuous => JointAxesMask::LOCKED_REVOLUTE_AXES,
        JointType::Prismatic => JointAxesMask::LOCKED_PRISMATIC_AXES,
        _ => JointAxesMask::LOCKED_FIXED_AXES,
    };

    let mut builder = GenericJointBuilder::new(locked_axes)
        .local_anchor1(origin_pos)
        .contacts_enabled(false);

    // Set joint axis in both parent and child frames.
    // In the parent frame, the axis is rotated by the joint origin RPY.
    // In the child frame, the axis is unrotated (child body is aligned with the joint).
    let normalized_axis = axis.normalize_or_zero();
    if normalized_axis != Vec3::ZERO {
        builder = builder
            .local_axis1(origin_rot * normalized_axis)
            .local_axis2(normalized_axis);
    }

    match joint_type {
        JointType::Revolute => {
            if let (Some(lo), Some(hi)) = (lower, upper) {
                builder = builder.limits(JointAxis::AngX, [lo, hi]);
            }
        }
        JointType::Prismatic => {
            if let (Some(lo), Some(hi)) = (lower, upper) {
                builder = builder.limits(JointAxis::LinX, [lo, hi]);
            }
        }
        _ => {}
    }

    let mut joint = builder.build();

    // Set acceleration-based motor model for actuated joints.
    // AccelerationBased is mass-independent, so same gains work for heavy
    // shoulder and light wrist joints.
    match joint_type {
        JointType::Revolute | JointType::Continuous => {
            joint.set_motor_model(JointAxis::AngX, MotorModel::AccelerationBased);
            joint.set_motor(JointAxis::AngX, 0.0, 0.0, 0.0, 0.0);
        }
        JointType::Prismatic => {
            joint.set_motor_model(JointAxis::LinX, MotorModel::AccelerationBased);
            joint.set_motor(JointAxis::LinX, 0.0, 0.0, 0.0, 0.0);
        }
        _ => {}
    }

    joint
}

/// Rotation to convert URDF's Z-up cylinder to Rapier's Y-up cylinder.
///
/// URDF specifies cylinders with their axis along Z, but Rapier's
/// `ColliderBuilder::cylinder` creates Y-up cylinders.  Rotating by
/// π/2 around X maps Y→Z, matching the URDF convention.
/// See `rapier3d-urdf` for the canonical approach.
const CYLINDER_Z_UP: Quat = Quat::from_xyzw(
    std::f32::consts::FRAC_1_SQRT_2, // sin(π/4)
    0.0,
    0.0,
    std::f32::consts::FRAC_1_SQRT_2, // cos(π/4)
);

/// Create Rapier colliders from a link's URDF collision geometry.
///
/// Each collision element is converted to a Rapier collider shape and
/// attached to the link's rigid body. Mesh geometries are skipped with
/// a warning since trimesh import is not yet supported.
fn create_link_colliders(
    context: &mut RapierContext,
    body_handle: rapier3d::prelude::RigidBodyHandle,
    link: &clankers_urdf::types::LinkData,
    _link_world_rot: &Quat,
) {
    for collision in &link.collisions {
        let Some(collider_builder) = geometry_to_collider(&collision.geometry) else {
            continue;
        };

        // Apply collision origin offset relative to the link body.
        // For cylinders, compose with CYLINDER_Z_UP to convert Rapier's
        // Y-up cylinder to URDF's Z-up convention.
        let col_origin = &collision.origin;
        let col_pos = Vec3::new(col_origin.xyz[0], col_origin.xyz[1], col_origin.xyz[2]);
        let mut col_rot = Quat::from_euler(
            EulerRot::ZYX,
            col_origin.rpy[2],
            col_origin.rpy[1],
            col_origin.rpy[0],
        );
        if matches!(collision.geometry, Geometry::Cylinder { .. }) {
            col_rot *= CYLINDER_Z_UP;
        }
        let col_frame = rapier3d::glamx::Pose3::from_parts(col_pos, col_rot);

        let collider = collider_builder.position(col_frame).build();
        context
            .collider_set
            .insert_with_parent(collider, body_handle, &mut context.rigid_body_set);
    }
}

/// Convert a URDF [`Geometry`] to a Rapier [`ColliderBuilder`].
///
/// Returns `None` for mesh geometries (not yet supported).
fn geometry_to_collider(geometry: &Geometry) -> Option<ColliderBuilder> {
    match geometry {
        Geometry::Sphere { radius } => Some(ColliderBuilder::ball(*radius)),
        // URDF Box size is full extents; Rapier cuboid takes half-extents.
        Geometry::Box { size } => Some(ColliderBuilder::cuboid(
            size[0] / 2.0,
            size[1] / 2.0,
            size[2] / 2.0,
        )),
        // URDF Cylinder length is full length; Rapier cylinder takes half-height.
        Geometry::Cylinder { radius, length } => {
            Some(ColliderBuilder::cylinder(*length / 2.0, *radius))
        }
        Geometry::Mesh { .. } => {
            bevy::log::warn!("URDF mesh collision geometry not yet supported; skipping collider");
            None
        }
    }
}
