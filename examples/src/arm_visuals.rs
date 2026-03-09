//! Shared arm visualization: link meshes, scene setup, and physics-to-visual sync.
//!
//! Extracts common code from `arm_ik_viz`, `arm_pick_record`, and `arm_pick_replay`
//! so each binary can reuse the same mesh definitions, materials, and sync systems.

use bevy::prelude::*;
use clankers_physics::rapier::RapierContext;

use crate::arm_setup::ArmSetup;
use clankers_viz::{phys_rot_to_vis, phys_to_vis};

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Marker for a visual entity whose transform is driven by a named rigid body
/// in the Rapier physics context.
#[derive(Component)]
pub struct LinkVisual(pub &'static str);

/// Two gripper finger joint entities (left, right).
#[derive(Resource)]
pub struct GripperEntities(pub [Entity; 2]);

impl GripperEntities {
    /// Extract gripper finger entities from a spawned arm scene.
    #[must_use]
    pub fn from_setup(setup: &ArmSetup) -> Self {
        let spawned = &setup.scene.robots["six_dof_arm"];
        Self([
            spawned
                .joint_entity("j_finger_left")
                .expect("j_finger_left not found"),
            spawned
                .joint_entity("j_finger_right")
                .expect("j_finger_right not found"),
        ])
    }
}

// ---------------------------------------------------------------------------
// Startup: spawn arm link meshes with materials
// ---------------------------------------------------------------------------

/// Spawn primitive meshes for each arm link, gripper, and end-effector.
///
/// Each link is a parent entity with a `LinkVisual` marker (for physics sync)
/// and a child mesh at the appropriate local offset. This matches the pattern
/// used across all arm visualization binaries.
pub fn spawn_arm_link_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let base_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.3, 0.35),
        ..default()
    });
    let link_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.5, 0.8),
        ..default()
    });
    let forearm_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.7, 0.3),
        ..default()
    });
    let wrist_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.8, 0.2),
        ..default()
    });
    let ee_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.4, 0.1),
        ..default()
    });
    let gripper_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.6, 0.6, 0.65),
        ..default()
    });

    // Cylindrical link segments
    for (name, radius, height, y_off, mat) in [
        ("base", 0.08, 0.1, 0.0, &base_mat),
        ("shoulder_link", 0.04, 0.2, 0.1, &link_mat),
        ("upper_arm", 0.035, 0.3, 0.15, &link_mat),
        ("elbow_link", 0.03, 0.1, 0.05, &forearm_mat),
        ("forearm", 0.025, 0.2, 0.1, &forearm_mat),
        ("wrist_link", 0.02, 0.06, 0.03, &wrist_mat),
    ] {
        commands
            .spawn((
                LinkVisual(name),
                Visibility::default(),
                Transform::default(),
            ))
            .with_children(|p| {
                p.spawn((
                    Mesh3d(meshes.add(Cylinder::new(radius, height))),
                    MeshMaterial3d(mat.clone()),
                    Transform::from_xyz(0.0, y_off, 0.0),
                ));
            });
    }

    // End-effector sphere
    commands
        .spawn((
            LinkVisual("end_effector"),
            Visibility::default(),
            Transform::default(),
        ))
        .with_children(|p| {
            p.spawn((
                Mesh3d(meshes.add(Sphere::new(0.025))),
                MeshMaterial3d(ee_mat),
                Transform::IDENTITY,
            ));
        });

    // Gripper base
    commands
        .spawn((
            LinkVisual("gripper_base"),
            Visibility::default(),
            Transform::default(),
        ))
        .with_children(|p| {
            p.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.06, 0.02, 0.04))),
                MeshMaterial3d(gripper_mat.clone()),
                Transform::IDENTITY,
            ));
        });

    // Gripper fingers
    for finger_name in ["finger_left", "finger_right"] {
        commands
            .spawn((
                LinkVisual(finger_name),
                Visibility::default(),
                Transform::default(),
            ))
            .with_children(|p| {
                p.spawn((
                    Mesh3d(meshes.add(Cuboid::new(0.01, 0.04, 0.01))),
                    MeshMaterial3d(gripper_mat.clone()),
                    Transform::from_xyz(0.0, 0.02, 0.0),
                ));
            });
    }
}

// ---------------------------------------------------------------------------
// Startup: spawn table, lighting, and ground for arm scenes
// ---------------------------------------------------------------------------

/// Spawn a tabletop scene with ground plane, directional light, ambient light,
/// and a wooden table surface. Used by arm viz, record, and replay binaries.
pub fn spawn_arm_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(5.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.38, 0.35),
            ..default()
        })),
    ));

    // Directional light (sun)
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.4, 0.0)),
    ));

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
        ..default()
    });

    // Table surface
    let table_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.6, 0.42),
        ..default()
    });
    commands.spawn((
        LinkVisual("table"),
        Mesh3d(meshes.add(Cuboid::new(0.6, 0.025, 0.4))),
        MeshMaterial3d(table_mat),
        Visibility::default(),
        Transform::default(),
    ));

    // Red cube
    let red_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.15, 0.1),
        ..default()
    });
    commands.spawn((
        LinkVisual("red_cube"),
        Mesh3d(meshes.add(Cuboid::new(0.025, 0.025, 0.025))),
        MeshMaterial3d(red_mat),
        Visibility::default(),
        Transform::default(),
    ));
}

// ---------------------------------------------------------------------------
// Runtime: sync link visuals from physics
// ---------------------------------------------------------------------------

/// Sync `LinkVisual` entity transforms from Rapier rigid body state.
///
/// Looks up each `LinkVisual`'s name in `RapierContext::body_handles`, reads
/// the body's translation and rotation, and applies the physics-to-visual
/// coordinate transform.
#[allow(clippy::needless_pass_by_value)]
pub fn sync_link_visuals(ctx: Res<RapierContext>, mut query: Query<(&LinkVisual, &mut Transform)>) {
    for (link, mut tf) in &mut query {
        let Some(&handle) = ctx.body_handles.get(link.0) else {
            continue;
        };
        let Some(body) = ctx.rigid_body_set.get(handle) else {
            continue;
        };
        tf.translation = phys_to_vis(body.translation());
        tf.rotation = phys_rot_to_vis(body.rotation());
    }
}
