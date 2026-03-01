//! Tabletop manipulation scene with a 6-DOF arm and graspable objects.
//!
//! Demonstrates: URDF with gripper, collision groups, segmentation tags,
//! physics objects, teleop control, and position-controlled joints.
//!
//! ## Input separation
//!
//! - **Scene camera** (orbit): mouse only -- left-drag orbit, right-drag pan,
//!   scroll zoom.
//! - **Joint control** (teleop): keyboard only -- Q/A, W/S, E/D, R/F, T/G,
//!   Y/H = arm joints 1-6. U/J = gripper open/close.
//!
//! Run: `cargo run -p clankers-examples --bin arm_manipulation`

use bevy::prelude::*;
use clankers_core::ClankersSet;
use clankers_core::types::SegmentationClass;
use clankers_env::prelude::*;
use clankers_examples::arm_setup::{ArmSetupConfig, setup_arm};
use clankers_physics::rapier::RapierContext;
use clankers_teleop::prelude::*;
use clankers_viz::ClankersVizPlugin;
use clankers_viz::input::{KeyboardJointBinding, KeyboardTeleopMap};
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyHandle};

// ---------------------------------------------------------------------------
// Visual markers
// ---------------------------------------------------------------------------

/// Marker for the table visual mesh.
#[derive(Component)]
struct TableVisual;

/// Marker for manipulable object visuals. Index into `ObjectHandles::bodies`.
#[derive(Component)]
struct ObjectVisual(usize);

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Rigid-body handles for dynamic scene objects (not the robot).
#[derive(Resource)]
struct ObjectHandles {
    bodies: Vec<RigidBodyHandle>,
}

// ---------------------------------------------------------------------------
// Startup: spawn visual meshes for table and objects
// ---------------------------------------------------------------------------

fn spawn_scene_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Table: wooden brown, static
    commands.spawn((
        TableVisual,
        SegmentationClass::TABLE,
        Mesh3d(meshes.add(Cuboid::new(0.6, 0.4, 0.025))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.4, 0.2),
            ..default()
        })),
        Transform::from_xyz(0.35, 0.0, 0.4),
    ));

    // Red cube
    commands.spawn((
        ObjectVisual(0),
        SegmentationClass::OBJECT,
        Mesh3d(meshes.add(Cuboid::new(0.025, 0.025, 0.025))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.1, 0.1),
            ..default()
        })),
        Transform::from_xyz(0.3, 0.0, 0.425),
    ));

    // Blue cube
    commands.spawn((
        ObjectVisual(1),
        SegmentationClass::OBJECT,
        Mesh3d(meshes.add(Cuboid::new(0.03, 0.03, 0.03))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.1, 0.9),
            ..default()
        })),
        Transform::from_xyz(0.4, 0.05, 0.425),
    ));

    // Green cylinder
    commands.spawn((
        ObjectVisual(2),
        SegmentationClass::OBJECT,
        Mesh3d(meshes.add(Cylinder::new(0.02, 0.03))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.8, 0.1),
            ..default()
        })),
        Transform::from_xyz(0.35, -0.05, 0.43),
    ));
}

// ---------------------------------------------------------------------------
// Visual sync: read rapier body transforms -> mesh transforms
// ---------------------------------------------------------------------------

/// Sync object visual transforms from rapier rigid-body state.
#[allow(clippy::needless_pass_by_value)]
fn sync_objects_system(
    ctx: Res<RapierContext>,
    handles: Res<ObjectHandles>,
    mut objects: Query<(&ObjectVisual, &mut Transform)>,
) {
    for (obj, mut transform) in &mut objects {
        if let Some(&handle) = handles.bodies.get(obj.0)
            && let Some(body) = ctx.rigid_body_set.get(handle)
        {
            let pos = body.translation();
            let rot = body.rotation();
            transform.translation = Vec3::new(pos.x, pos.y, pos.z);
            transform.rotation = *rot;
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // 1. Setup arm with shared module (8 DOF for arm + gripper, viz mode)
    let setup = setup_arm(ArmSetupConfig {
        max_episode_steps: 50_000,
        use_fixed_update: true,
        sensor_dof: 8,
    });
    let mut scene = setup.scene;

    // 2. Add table and dynamic objects to rapier context
    {
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();

        // Table body (fixed, at z=0.4)
        let table_body = ctx.rigid_body_set.insert(
            RigidBodyBuilder::fixed()
                .translation(Vec3::new(0.35, 0.0, 0.4))
                .build(),
        );
        let table_collider = ColliderBuilder::cuboid(0.3, 0.2, 0.0125)
            .friction(0.6)
            .build();
        ctx.collider_set
            .insert_with_parent(table_collider, table_body, &mut ctx.rigid_body_set);

        // Dynamic objects on the table
        let object_configs: Vec<(&str, Vec3, ColliderBuilder)> = vec![
            (
                "red_cube",
                Vec3::new(0.3, 0.0, 0.425),
                ColliderBuilder::cuboid(0.0125, 0.0125, 0.0125),
            ),
            (
                "blue_cube",
                Vec3::new(0.4, 0.05, 0.425),
                ColliderBuilder::cuboid(0.015, 0.015, 0.015),
            ),
            (
                "green_cylinder",
                Vec3::new(0.35, -0.05, 0.43),
                ColliderBuilder::cylinder(0.015, 0.02),
            ),
        ];

        let mut object_bodies = Vec::new();
        for (_name, pos, collider_builder) in &object_configs {
            let body = ctx.rigid_body_set.insert(
                RigidBodyBuilder::dynamic()
                    .translation(*pos)
                    .can_sleep(false)
                    .build(),
            );
            let collider = collider_builder
                .clone()
                .density(500.0)
                .friction(0.8)
                .build();
            ctx.collider_set
                .insert_with_parent(collider, body, &mut ctx.rigid_body_set);
            object_bodies.push(body);
        }

        world.insert_resource(ObjectHandles {
            bodies: object_bodies,
        });
        world.insert_resource(ctx);
    }

    // 3. Teleop: map 6 arm joints + 2 gripper fingers to keyboard channels
    let spawned = &scene.robots["six_dof_arm"];
    let arm_joints = &setup.joint_entities;

    let finger_left = spawned.joint_entity("j_finger_left");
    let finger_right = spawned.joint_entity("j_finger_right");

    let mut teleop_config = TeleopConfig::new();
    for (i, &entity) in arm_joints.iter().enumerate() {
        teleop_config = teleop_config.with_mapping(
            format!("joint_{i}"),
            JointMapping::new(entity).with_scale(2.0),
        );
    }
    // Gripper: both fingers on channels 6 and 7
    if let Some(fl) = finger_left {
        teleop_config = teleop_config
            .with_mapping("joint_6".to_string(), JointMapping::new(fl).with_scale(0.5));
    }
    if let Some(fr) = finger_right {
        teleop_config = teleop_config
            .with_mapping("joint_7".to_string(), JointMapping::new(fr).with_scale(0.5));
    }
    scene.app.insert_resource(teleop_config);

    // Custom keyboard map: Q/A..Y/H = arm joints, U/J = gripper
    let mut bindings = Vec::new();
    let arm_key_pairs = [
        (KeyCode::KeyQ, KeyCode::KeyA),
        (KeyCode::KeyW, KeyCode::KeyS),
        (KeyCode::KeyE, KeyCode::KeyD),
        (KeyCode::KeyR, KeyCode::KeyF),
        (KeyCode::KeyT, KeyCode::KeyG),
        (KeyCode::KeyY, KeyCode::KeyH),
    ];
    for (i, &(pos, neg)) in arm_key_pairs.iter().enumerate() {
        bindings.push(KeyboardJointBinding {
            channel: format!("joint_{i}"),
            key_positive: pos,
            key_negative: neg,
        });
    }
    // U/J controls both gripper fingers together
    bindings.push(KeyboardJointBinding {
        channel: "joint_6".into(),
        key_positive: KeyCode::KeyU,
        key_negative: KeyCode::KeyJ,
    });
    bindings.push(KeyboardJointBinding {
        channel: "joint_7".into(),
        key_positive: KeyCode::KeyU,
        key_negative: KeyCode::KeyJ,
    });

    scene.app.insert_resource(KeyboardTeleopMap {
        bindings,
        increment: 0.05,
    });

    // 4. Windowed rendering
    scene.app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Clankers -- Arm Manipulation".to_string(),
            resolution: (1280, 720).into(),
            ..default()
        }),
        ..default()
    }));

    // 5. Teleop + viz plugins
    scene.app.add_plugins(ClankersTeleopPlugin);
    scene.app.add_plugins(ClankersVizPlugin::default());

    // 6. Systems
    scene.app.add_systems(Startup, spawn_scene_meshes);
    scene
        .app
        .add_systems(Update, sync_objects_system.after(ClankersSet::Simulate));

    // 7. Start episode and run
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("Arm manipulation scene");
    println!("  Scene camera: mouse (orbit/pan/zoom)");
    println!("  Joint control: Q/A, W/S, E/D, R/F, T/G, Y/H = arm joints 1-6");
    println!("  Gripper: U/J = open/close");
    println!("  Physics: Rapier3D with table + 3 objects");
    scene.app.run();
}
