//! Cart-pole visualization driven by an ONNX policy model.
//!
//! Loads a trained ONNX policy and runs it in the Bevy simulation loop
//! to balance the cart-pole without any manual input.
//!
//! Export a model first:
//!     py -3.12 python/examples/export_sb3_to_onnx.py
//!
//! Then run:
//!     cargo run -p clankers-examples --bin cartpole_policy_viz --release -- --model python/examples/cartpole_ppo.onnx

use std::collections::HashMap;
use std::path::PathBuf;

use bevy::prelude::*;
use clap::Parser;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::ClankersSet;
use clankers_env::prelude::*;
use clankers_examples::CARTPOLE_URDF;
use clankers_physics::rapier::{bridge::register_robot, RapierBackend, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_policy::prelude::*;
use clankers_sim::SceneBuilder;
use clankers_viz::ClankersVizPlugin;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Cart-pole visualization driven by an ONNX policy.
#[derive(Parser)]
#[command(name = "cartpole_policy_viz")]
#[command(about = "Visualize a trained ONNX policy on the cart-pole")]
struct Cli {
    /// Path to the ONNX policy model file.
    #[arg(long)]
    model: PathBuf,
}

// ---------------------------------------------------------------------------
// Visual markers -- these tag meshes, NOT joint data entities.
// Joint data lives on separate ECS entities (Actuator/JointCommand/etc).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// References to joint DATA entities (not visual entities).
/// These hold Actuator, JointCommand, JointState, JointTorque components.
#[derive(Resource)]
struct CartPoleJoints {
    /// Prismatic joint entity -- command drives X translation.
    cart: Entity,
    /// Revolute joint entity -- command drives Z rotation.
    pole: Entity,
}

// ---------------------------------------------------------------------------
// Startup: spawn robot meshes (visual only, no joint data)
// ---------------------------------------------------------------------------

fn spawn_robot_meshes(
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
// Visual sync: read JointState (physics output) to drive mesh transforms
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
// Policy action applicator
// ---------------------------------------------------------------------------

/// Copies PolicyRunner's current action to JointCommand components.
///
/// The cartpole PPO policy produces a 2-element action:
///   action[0] -> cart_slide force (JointCommand on cart entity)
///   action[1] -> pole_hinge torque (JointCommand on pole entity)
///
/// If the policy produces only 1 action (discrete cart force), only
/// the cart joint is driven.
#[allow(clippy::needless_pass_by_value)]
fn apply_policy_action(
    runner: Res<PolicyRunner>,
    joints: Res<CartPoleJoints>,
    mut commands: Query<&mut JointCommand>,
) {
    let action = runner.action().as_slice();

    // Cart force (action index 0)
    if let Ok(mut cmd) = commands.get_mut(joints.cart) {
        cmd.value = action.first().copied().unwrap_or(0.0);
    }

    // Pole torque (action index 1), if present
    if action.len() > 1 {
        if let Ok(mut cmd) = commands.get_mut(joints.pole) {
            cmd.value = action.get(1).copied().unwrap_or(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    // 1. Load the ONNX policy
    println!("Loading ONNX policy from {}...", cli.model.display());
    let onnx_policy = OnnxPolicy::from_file(&cli.model)
        .unwrap_or_else(|e| panic!("failed to load ONNX model: {e}"));
    let action_dim = onnx_policy.action_dim();
    println!(
        "  Policy loaded: obs_dim={}, action_dim={}",
        onnx_policy.obs_dim(),
        action_dim,
    );

    // 2. Parse URDF
    let model =
        clankers_urdf::parse_string(CARTPOLE_URDF).expect("failed to parse cartpole URDF");

    // 3. Build scene
    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(10_000)
        .with_robot(model.clone(), HashMap::new())
        .build();

    let cart = scene.robots["cartpole"]
        .joint_entity("cart_slide")
        .expect("missing cart_slide joint");
    let pole = scene.robots["cartpole"]
        .joint_entity("pole_hinge")
        .expect("missing pole_hinge joint");

    // 4. Add Rapier physics (must come after build so SimConfig exists)
    scene
        .app
        .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

    // 5. Register robot with the rapier context (fixed base = rail anchored)
    {
        let spawned = &scene.robots["cartpole"];
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        register_robot(&mut ctx, &model, spawned, world, true);
        world.insert_resource(ctx);
    }

    // 6. Register sensors (JointStateSensor fills ObservationBuffer
    //    with [cart_pos, cart_vel, pole_pos, pole_vel] = 4 obs)
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(
            Box::new(JointStateSensor::new(2)),
            &mut buffer,
        );
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // 7. Joint entity references (data entities, NOT visual entities)
    scene
        .app
        .insert_resource(CartPoleJoints { cart, pole });

    // 8. PolicyRunner + ClankersPolicyPlugin
    let runner = PolicyRunner::new(Box::new(onnx_policy), action_dim);
    scene.app.insert_resource(runner);
    scene.app.add_plugins(ClankersPolicyPlugin);

    // 9. Windowed rendering (scene camera added by ClankersVizPlugin)
    scene.app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Clankers - Cart-Pole Policy Viz".to_string(),
            resolution: (1280, 720).into(),
            ..default()
        }),
        ..default()
    }));

    // 10. Viz plugin (orbit camera + egui panel, NO teleop)
    scene.app.add_plugins(ClankersVizPlugin);

    // 11. Robot visual meshes (separate from joint data entities)
    scene.app.add_systems(Startup, spawn_robot_meshes);

    // 12. Visual sync: JointState (physics) -> mesh transforms
    //     Must run after physics so we read up-to-date JointState.
    scene.app.add_systems(
        Update,
        (sync_cart_visual, sync_pivot_visual, sync_pole_visual)
            .after(ClankersSet::Simulate),
    );

    // 13. Action applicator: PolicyRunner -> JointCommand
    //     Must run after Decide (policy has produced action)
    //     and before Act (actuator reads JointCommand).
    scene.app.add_systems(
        Update,
        apply_policy_action
            .after(ClankersSet::Decide)
            .before(ClankersSet::Act),
    );

    // 14. Start episode
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("Cart-pole policy viz with Rapier physics");
    println!("  Scene camera: mouse (orbit/pan/zoom)");
    println!("  Joint control: ONNX policy (no keyboard)");
    println!("  Model: {}", cli.model.display());
    scene.app.run();
}
