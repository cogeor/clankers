//! Interactive cart-pole visualization with physics simulation.
//!
//! Demonstrates the full pipeline: URDF parsing, prismatic + revolute joints,
//! Rapier3D physics, sensor noise (encoder + force), teleop control, and 3D
//! visualization.
//!
//! ## Input separation
//!
//! - **Scene camera** (orbit): mouse only — left-drag orbit, right-drag pan,
//!   scroll zoom. Does NOT respond to keyboard.
//! - **Joint control** (teleop): keyboard only — A/D = cart slide, W/S = pole
//!   hinge. Also controllable via egui sliders in Teleop mode.
//! - **Robot camera**: not present in this example, but could be added as an
//!   optional `Camera3d` parented to a robot link entity.
//!
//! ## Visual sync
//!
//! Robot meshes are driven from `JointState.position` (physics output).
//! Teleop sets `JointCommand`, which the `IdealMotor` converts to torque,
//! and the Rapier physics engine integrates into joint state.
//!
//! Run: `cargo run -p clankers-examples --bin pendulum_viz`

use std::collections::HashMap;

use bevy::prelude::*;
use clankers_actuator::components::JointState;
use clankers_core::ClankersSet;
use clankers_env::prelude::*;
use clankers_examples::CARTPOLE_URDF;
use clankers_noise::prelude::*;
use clankers_physics::rapier::{bridge::register_robot, RapierBackend, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_sim::SceneBuilder;
use clankers_teleop::prelude::*;
use clankers_viz::input::{KeyboardJointBinding, KeyboardTeleopMap};
use clankers_viz::ClankersVizPlugin;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// Visual markers — these tag meshes, NOT joint data entities.
// Joint data lives on separate ECS entities (Actuator/JointCommand/etc).
// ---------------------------------------------------------------------------

/// Static rail mesh.
#[derive(Component)]
struct RailVisual;

/// Cart mesh — translates along X from the cart_slide JointState.
#[derive(Component)]
struct CartVisual;

/// Pivot sphere — follows the cart position.
#[derive(Component)]
struct PivotVisual;

/// Pole parent — follows cart X, rotates from pole_hinge JointState.
#[derive(Component)]
struct PoleVisual;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// References to joint DATA entities (not visual entities).
/// These hold Actuator, JointCommand, JointState, JointTorque components.
#[derive(Resource)]
struct CartPoleJoints {
    /// Prismatic joint entity — command drives X translation.
    cart: Entity,
    /// Revolute joint entity — command drives Z rotation.
    pole: Entity,
}

/// Sensor noise applied to observations each frame.
#[derive(Resource)]
struct SensorNoise {
    cart_noise: IndependentAxesNoise,
    pole_noise: IndependentAxesNoise,
    force_noise: NoiseModel,
    rng: ChaCha8Rng,
    slot_index: usize,
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
        // Revolute: physics angle → Z rotation (visible swing)
        t.rotation = Quat::from_rotation_z(pole_state.position);
    }
}

// ---------------------------------------------------------------------------
// Sensor noise: applies encoder/force noise to a dedicated obs buffer slot
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn apply_sensor_noise(
    joints: Res<CartPoleJoints>,
    states: Query<&JointState>,
    mut noise: ResMut<SensorNoise>,
    mut obs_buf: ResMut<ObservationBuffer>,
) {
    let (Ok(cart_state), Ok(pole_state)) =
        (states.get(joints.cart), states.get(joints.pole))
    else {
        return;
    };

    let SensorNoise {
        cart_noise,
        pole_noise,
        force_noise,
        rng,
        slot_index,
    } = &mut *noise;

    let cart_clean = [cart_state.position, cart_state.velocity];
    let pole_clean = [pole_state.position, pole_state.velocity];

    let cart_noisy = cart_noise.apply_vec(&cart_clean, rng);
    let pole_noisy = pole_noise.apply_vec(&pole_clean, rng);
    let force_noisy = force_noise.apply(0.0, rng);

    let noisy_obs = [
        cart_noisy[0],
        cart_noisy[1],
        pole_noisy[0],
        pole_noisy[1],
        force_noisy,
    ];
    obs_buf.write(*slot_index, &noisy_obs);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // 1. Parse URDF (keep model for physics registration)
    let model =
        clankers_urdf::parse_string(CARTPOLE_URDF).expect("failed to parse cartpole URDF");

    // 2. Build scene from parsed model
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

    // 3. Add Rapier physics (must come after build so SimConfig exists)
    scene
        .app
        .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

    // 4. Register robot with the rapier context (fixed base = rail anchored)
    {
        let spawned = &scene.robots["cartpole"];
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        register_robot(&mut ctx, &model, spawned, world, true);
        world.insert_resource(ctx);
    }

    // 5. Register sensors + noisy observation slot
    let noisy_slot;
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(2)), &mut buffer);
        registry.register(Box::new(JointCommandSensor::new(2)), &mut buffer);
        noisy_slot = buffer.register("noisy_obs", 5);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // 6. Joint entity references (data entities, NOT visual entities)
    scene
        .app
        .insert_resource(CartPoleJoints { cart, pole });

    // 7. Sensor noise: encoder on pos/vel, force on torque readings
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut cart_noise = IndependentAxesNoise::new(vec![
        presets::encoder_position().expect("encoder position noise"),
        presets::encoder_velocity().expect("encoder velocity noise"),
    ]);
    let mut pole_noise = IndependentAxesNoise::new(vec![
        presets::encoder_position().expect("encoder position noise"),
        presets::encoder_velocity().expect("encoder velocity noise"),
    ]);
    let mut force_noise = presets::force_sensor(0.1).expect("force sensor noise");
    cart_noise.reset(&mut rng);
    pole_noise.reset(&mut rng);
    force_noise.reset(&mut rng);

    scene.app.insert_resource(SensorNoise {
        cart_noise,
        pole_noise,
        force_noise,
        rng,
        slot_index: noisy_slot,
    });

    // 8. Teleop: keyboard drives joints, NOT the camera.
    //    A/D → joint_0 (cart_slide), W/S → joint_1 (pole_hinge)
    //    Scene camera is mouse-only (PanOrbitCamera).
    let teleop_config = TeleopConfig::new()
        .with_mapping(
            "joint_0",
            JointMapping::new(cart).with_scale(10.0).with_dead_zone(0.05),
        )
        .with_mapping(
            "joint_1",
            JointMapping::new(pole).with_scale(5.0).with_dead_zone(0.05),
        );
    scene.app.insert_resource(teleop_config);

    // Custom keyboard map: A/D = cart, W/S = pole (WASD-style).
    scene.app.insert_resource(KeyboardTeleopMap {
        bindings: vec![
            KeyboardJointBinding {
                channel: "joint_0".into(),
                key_positive: KeyCode::KeyD,
                key_negative: KeyCode::KeyA,
            },
            KeyboardJointBinding {
                channel: "joint_1".into(),
                key_positive: KeyCode::KeyW,
                key_negative: KeyCode::KeyS,
            },
        ],
        increment: 0.05,
    });

    // 9. Windowed rendering (scene camera added by ClankersVizPlugin)
    scene.app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Clankers — Cart-Pole Viz".to_string(),
            resolution: (1280, 720).into(),
            ..default()
        }),
        ..default()
    }));

    // 10. Teleop plugin (keyboard → TeleopCommander → JointCommand)
    //     Viz plugin (orbit camera + egui panel + mode gating)
    scene.app.add_plugins(ClankersTeleopPlugin);
    scene.app.add_plugins(ClankersVizPlugin::default());

    // 11. Robot visual meshes (separate from joint data entities)
    scene.app.add_systems(Startup, spawn_robot_meshes);

    // 12. Visual sync: JointState (physics) → mesh transforms
    //     Must run after physics so we read up-to-date JointState.
    scene.app.add_systems(
        Update,
        (sync_cart_visual, sync_pivot_visual, sync_pole_visual)
            .after(ClankersSet::Simulate),
    );

    // 13. Sensor noise: noisy readings in obs buffer
    scene.app.add_systems(Update, apply_sensor_noise);

    // 14. Start episode
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("Cart-pole viz with Rapier physics + sensor noise");
    println!("  Scene camera: mouse (orbit/pan/zoom)");
    println!("  Joint control: A/D = cart, W/S = pole");
    println!("  Physics: Rapier3D, gravity [0,0,-9.81], 20 substeps/frame");
    scene.app.run();
}
