//! Multi-robot visualization with robot selection and teleop switching.
//!
//! Demonstrates the full multi-robot viz pipeline: two robots (cart-pole and
//! two-link arm) rendered side by side with Rapier physics, per-robot visual
//! sync, and the robot selector GUI for dynamic teleop rebinding.
//!
//! ## Robots
//!
//! - **Cartpole** (x = 0): prismatic cart + revolute pole (2 joints)
//! - **Two-link arm** (x = 3): shoulder + elbow revolute joints (2 joints)
//!
//! ## Input separation
//!
//! - **Scene camera** (orbit): mouse only -- left-drag orbit, right-drag pan,
//!   scroll zoom. Does NOT respond to keyboard.
//! - **Joint control** (teleop): keyboard only -- bindings auto-switch when
//!   selecting a robot in the GUI panel.
//!
//! No policy or ONNX model is needed -- this is teleop-only.
//!
//! Run: `cargo run -p clankers-examples --bin multi_robot_viz`

use std::collections::HashMap;

use bevy::prelude::*;
use clankers_actuator::components::JointState;
use clankers_core::ClankersSet;
use clankers_env::prelude::*;
use clankers_examples::{CARTPOLE_URDF, TWO_LINK_ARM_URDF};
use clankers_physics::rapier::{bridge::register_robot, RapierBackend, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_sim::SceneBuilder;
use clankers_teleop::prelude::*;
use clankers_viz::ClankersVizPlugin;

// ---------------------------------------------------------------------------
// Visual markers -- these tag meshes, NOT joint data entities.
// Joint data lives on separate ECS entities (Actuator/JointCommand/etc).
// ---------------------------------------------------------------------------

// -- Cartpole visuals (at x = 0) --

/// Static rail mesh.
#[derive(Component)]
struct RailVisual;

/// Cart mesh -- translates along X from the cart_slide JointState.
#[derive(Component)]
struct CartVisual;

/// Pivot sphere -- follows the cart position.
#[derive(Component)]
struct PivotVisual;

/// Pole parent -- follows cart X, rotates from pole_hinge JointState.
#[derive(Component)]
struct PoleVisual;

// -- Two-link arm visuals (at x = 3) --

/// Fixed base cylinder for the arm.
#[derive(Component)]
struct ArmBaseVisual;

/// Upper arm parent -- rotates by shoulder angle.
#[derive(Component)]
struct UpperArmVisual;

/// Forearm parent -- positioned at upper arm end, rotates by shoulder + elbow.
#[derive(Component)]
struct ForearmVisual;

/// End-effector sphere at the forearm tip.
#[derive(Component)]
struct EEVisual;

// ---------------------------------------------------------------------------
// Resources: joint entity references (data entities, NOT visual entities)
// ---------------------------------------------------------------------------

/// References to cartpole joint DATA entities.
#[derive(Resource)]
struct CartPoleJoints {
    /// Prismatic joint entity -- command drives X translation.
    cart: Entity,
    /// Revolute joint entity -- command drives Y rotation.
    pole: Entity,
}

/// References to two-link arm joint DATA entities.
#[derive(Resource)]
struct ArmJoints {
    /// Revolute shoulder joint entity (Z-axis).
    shoulder: Entity,
    /// Revolute elbow joint entity (Z-axis).
    elbow: Entity,
}

// ---------------------------------------------------------------------------
// Arm geometry constants
// ---------------------------------------------------------------------------

/// X offset for the two-link arm (separate from cartpole).
const ARM_X_OFFSET: f32 = 3.0;

/// Height of the arm base above ground (half the base cylinder height).
const ARM_BASE_HEIGHT: f32 = 0.05;

/// Length of the upper arm segment (matches URDF).
const UPPER_ARM_LEN: f32 = 0.3;

/// Length of the forearm segment (matches URDF).
const FOREARM_LEN: f32 = 0.25;

// ---------------------------------------------------------------------------
// Startup: spawn cartpole meshes (same as pendulum_viz)
// ---------------------------------------------------------------------------

fn spawn_cartpole_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let rail_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.5, 0.55),
        ..default()
    });
    let cart_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.5, 0.8),
        ..default()
    });
    let pole_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.3, 0.2),
        ..default()
    });
    let joint_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.8, 0.2),
        ..default()
    });

    // Rail: static, never moves
    commands.spawn((
        RailVisual,
        Mesh3d(meshes.add(Cuboid::new(5.0, 0.04, 0.06))),
        MeshMaterial3d(rail_mat),
        Transform::from_xyz(0.0, 0.02, 0.0),
    ));

    // Cart: slides along X based on cart_slide JointState
    commands.spawn((
        CartVisual,
        Mesh3d(meshes.add(Cuboid::new(0.25, 0.12, 0.15))),
        MeshMaterial3d(cart_mat),
        Transform::from_xyz(0.0, 0.1, 0.0),
    ));

    // Pivot sphere: follows cart position
    commands.spawn((
        PivotVisual,
        Mesh3d(meshes.add(Sphere::new(0.04))),
        MeshMaterial3d(joint_mat),
        Transform::from_xyz(0.0, 0.16, 0.0),
    ));

    // Pole: parent at pivot, children rotate. Needs Visibility for children.
    commands
        .spawn((
            PoleVisual,
            Visibility::default(),
            Transform::from_xyz(0.0, 0.16, 0.0),
        ))
        .with_children(|parent| {
            parent.spawn((
                Mesh3d(meshes.add(Cylinder::new(0.02, 1.0))),
                MeshMaterial3d(pole_mat.clone()),
                Transform::from_xyz(0.0, 0.5, 0.0),
            ));
            parent.spawn((
                Mesh3d(meshes.add(Sphere::new(0.03))),
                MeshMaterial3d(pole_mat),
                Transform::from_xyz(0.0, 1.0, 0.0),
            ));
        });
}

// ---------------------------------------------------------------------------
// Startup: spawn two-link arm meshes
// ---------------------------------------------------------------------------

fn spawn_arm_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let base_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.15, 0.35, 0.15),
        ..default()
    });
    let upper_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.7, 0.3),
        ..default()
    });
    let forearm_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.8, 0.2),
        ..default()
    });
    let ee_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.5, 0.1),
        ..default()
    });

    // Base: fixed cylinder at arm offset position
    commands.spawn((
        ArmBaseVisual,
        Mesh3d(meshes.add(Cylinder::new(0.05, 0.1))),
        MeshMaterial3d(base_mat),
        Transform::from_xyz(ARM_X_OFFSET, ARM_BASE_HEIGHT, 0.0),
    ));

    // Upper arm: parent at shoulder pivot, rotates by shoulder angle.
    // Child cylinder is offset upward by half the arm length.
    commands
        .spawn((
            UpperArmVisual,
            Visibility::default(),
            Transform::from_xyz(ARM_X_OFFSET, ARM_BASE_HEIGHT * 2.0, 0.0),
        ))
        .with_children(|parent| {
            parent.spawn((
                Mesh3d(meshes.add(Cylinder::new(0.03, UPPER_ARM_LEN))),
                MeshMaterial3d(upper_mat),
                Transform::from_xyz(0.0, UPPER_ARM_LEN / 2.0, 0.0),
            ));
        });

    // Forearm: parent at elbow pivot, rotates by shoulder+elbow.
    // Position is computed in sync system. Child cylinder offset upward.
    commands
        .spawn((
            ForearmVisual,
            Visibility::default(),
            Transform::from_xyz(ARM_X_OFFSET, ARM_BASE_HEIGHT * 2.0 + UPPER_ARM_LEN, 0.0),
        ))
        .with_children(|parent| {
            parent.spawn((
                Mesh3d(meshes.add(Cylinder::new(0.025, FOREARM_LEN))),
                MeshMaterial3d(forearm_mat),
                Transform::from_xyz(0.0, FOREARM_LEN / 2.0, 0.0),
            ));
        });

    // End-effector sphere (synced in forearm system)
    commands.spawn((
        EEVisual,
        Mesh3d(meshes.add(Sphere::new(0.03))),
        MeshMaterial3d(ee_mat),
        Transform::from_xyz(ARM_X_OFFSET, ARM_BASE_HEIGHT * 2.0 + UPPER_ARM_LEN + FOREARM_LEN, 0.0),
    ));
}

// ---------------------------------------------------------------------------
// Cartpole visual sync (same as pendulum_viz)
// ---------------------------------------------------------------------------

/// Move cart mesh along X based on cart_slide physics state.
#[allow(clippy::needless_pass_by_value)]
fn sync_cart_visual(
    joints: Res<CartPoleJoints>,
    states: Query<&JointState>,
    mut cart: Query<&mut Transform, With<CartVisual>>,
) {
    let Ok(state) = states.get(joints.cart) else {
        return;
    };
    for mut t in &mut cart {
        t.translation.x = state.position.clamp(-1.0, 1.0);
    }
}

/// Move pivot sphere to follow cart.
#[allow(clippy::needless_pass_by_value)]
fn sync_pivot_visual(
    joints: Res<CartPoleJoints>,
    states: Query<&JointState>,
    mut pivot: Query<&mut Transform, With<PivotVisual>>,
) {
    let Ok(state) = states.get(joints.cart) else {
        return;
    };
    for mut t in &mut pivot {
        t.translation.x = state.position.clamp(-1.0, 1.0);
    }
}

/// Position pole at cart X and rotate from pole_hinge physics state.
#[allow(clippy::needless_pass_by_value)]
fn sync_pole_visual(
    joints: Res<CartPoleJoints>,
    states: Query<&JointState>,
    mut pole: Query<&mut Transform, With<PoleVisual>>,
) {
    let (Ok(cart_state), Ok(pole_state)) =
        (states.get(joints.cart), states.get(joints.pole))
    else {
        return;
    };
    for mut t in &mut pole {
        t.translation.x = cart_state.position.clamp(-1.0, 1.0);
        // Revolute: physics angle -> Z rotation (visible swing)
        t.rotation = Quat::from_rotation_z(pole_state.position);
    }
}

// ---------------------------------------------------------------------------
// Two-link arm visual sync
// ---------------------------------------------------------------------------

/// Rotate upper arm parent by shoulder angle (Z-axis revolute).
#[allow(clippy::needless_pass_by_value)]
fn sync_upper_arm_visual(
    joints: Res<ArmJoints>,
    states: Query<&JointState>,
    mut upper: Query<&mut Transform, With<UpperArmVisual>>,
) {
    let Ok(shoulder_state) = states.get(joints.shoulder) else {
        return;
    };
    for mut t in &mut upper {
        t.rotation = Quat::from_rotation_z(shoulder_state.position);
    }
}

/// Position forearm at upper arm end and rotate by shoulder + elbow angles.
#[allow(clippy::needless_pass_by_value)]
fn sync_forearm_visual(
    joints: Res<ArmJoints>,
    states: Query<&JointState>,
    mut forearm: Query<&mut Transform, (With<ForearmVisual>, Without<EEVisual>)>,
    mut ee: Query<&mut Transform, (With<EEVisual>, Without<ForearmVisual>)>,
) {
    let (Ok(shoulder_state), Ok(elbow_state)) =
        (states.get(joints.shoulder), states.get(joints.elbow))
    else {
        return;
    };

    let shoulder_angle = shoulder_state.position;
    let elbow_angle = elbow_state.position;

    // The shoulder rotates around Z, so the upper arm tip is at:
    //   base + upper_arm_len * rotate(shoulder_angle) applied to (0, 1, 0)
    let base_y = ARM_BASE_HEIGHT * 2.0;
    let upper_tip_x = ARM_X_OFFSET + UPPER_ARM_LEN * (-shoulder_angle).sin();
    let upper_tip_y = base_y + UPPER_ARM_LEN * shoulder_angle.cos();

    // Forearm rotation = shoulder + elbow
    let total_angle = shoulder_angle + elbow_angle;

    for mut t in &mut forearm {
        t.translation.x = upper_tip_x;
        t.translation.y = upper_tip_y;
        t.rotation = Quat::from_rotation_z(total_angle);
    }

    // End-effector at forearm tip
    let ee_x = upper_tip_x + FOREARM_LEN * (-total_angle).sin();
    let ee_y = upper_tip_y + FOREARM_LEN * total_angle.cos();

    for mut t in &mut ee {
        t.translation.x = ee_x;
        t.translation.y = ee_y;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // 1. Parse both URDFs (keep models for physics registration)
    let cartpole_model =
        clankers_urdf::parse_string(CARTPOLE_URDF).expect("failed to parse cartpole URDF");
    let arm_model =
        clankers_urdf::parse_string(TWO_LINK_ARM_URDF).expect("failed to parse two_link_arm URDF");

    // 2. Build scene with both robots
    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(10_000)
        .with_robot(cartpole_model.clone(), HashMap::new())
        .with_robot(arm_model.clone(), HashMap::new())
        .build();

    // 3. Extract joint entities
    let cart = scene.robots["cartpole"]
        .joint_entity("cart_slide")
        .expect("missing cart_slide joint");
    let pole = scene.robots["cartpole"]
        .joint_entity("pole_hinge")
        .expect("missing pole_hinge joint");

    let shoulder = scene.robots["two_link_arm"]
        .joint_entity("shoulder")
        .expect("missing shoulder joint");
    let elbow = scene.robots["two_link_arm"]
        .joint_entity("elbow")
        .expect("missing elbow joint");

    // 4. Add Rapier physics (must come after build so SimConfig exists)
    scene
        .app
        .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

    // 5. Register both robots with the rapier context (both fixed base)
    {
        let cartpole_spawned = &scene.robots["cartpole"];
        let arm_spawned = &scene.robots["two_link_arm"];
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        register_robot(&mut ctx, &cartpole_model, cartpole_spawned, world, true);
        register_robot(&mut ctx, &arm_model, arm_spawned, world, true);
        world.insert_resource(ctx);
    }

    // 6. Register sensors (joint state for all 4 joints)
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(4)), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // 7. Joint entity references (data entities, NOT visual entities)
    scene
        .app
        .insert_resource(CartPoleJoints { cart, pole });
    scene
        .app
        .insert_resource(ArmJoints { shoulder, elbow });

    // 8. Windowed rendering (scene camera added by ClankersVizPlugin)
    scene.app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Clankers -- Multi-Robot Viz".to_string(),
            resolution: (1280, 720).into(),
            ..default()
        }),
        ..default()
    }));

    // 9. Teleop plugin (keyboard -> TeleopCommander -> JointCommand)
    //    Viz plugin (orbit camera + egui panel + mode gating + robot selector)
    //    NOTE: No manual TeleopConfig/KeyboardTeleopMap needed --
    //    sync_teleop_to_robot rebuilds them automatically from RobotGroup.
    scene.app.add_plugins(ClankersTeleopPlugin);
    scene.app.add_plugins(ClankersVizPlugin::default());

    // 10. Robot visual meshes (separate from joint data entities)
    scene.app.add_systems(Startup, (spawn_cartpole_meshes, spawn_arm_meshes));

    // 11. Visual sync: JointState (physics) -> mesh transforms
    //     Must run after physics so we read up-to-date JointState.
    scene.app.add_systems(
        Update,
        (
            sync_cart_visual,
            sync_pivot_visual,
            sync_pole_visual,
            sync_upper_arm_visual,
            sync_forearm_visual,
        )
            .after(ClankersSet::Simulate),
    );

    // 12. Start episode
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("Multi-robot viz: cartpole (x=0) + two-link arm (x=3)");
    println!("  Scene camera: mouse (orbit/pan/zoom)");
    println!("  Joint control: keyboard (auto-switches via robot selector)");
    println!("  Select robot in GUI panel to rebind teleop keys");
    scene.app.run();
}
