//! 6-DOF arm gym server.
//!
//! Headless arm environment serving a single GymEnv over TCP for Python RL
//! training and data collection.
//!
//! Run: `cargo run -p clankers-examples --bin arm_gym`
//! Then connect with: `python python/examples/arm_imitation_learning.py --online`

use bevy::prelude::*;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::types::{Action, ActionSpace, ObservationSpace};
use clankers_examples::arm_setup::{ArmSetupConfig, setup_arm};
use clankers_gym::prelude::*;
use clankers_physics::rapier::RapierContext;

/// Joint entities for the 6-DOF arm, stored as a resource so the applicator
/// can target specific joints rather than iterating all joints in query order.
#[derive(Resource)]
struct ArmJointEntities(Vec<Entity>);

/// Maps 6-dim action to `JointCommand` on the arm's joint entities.
struct ArmApplicator;

impl clankers_core::traits::ActionApplicator for ArmApplicator {
    fn apply(&self, world: &mut World, action: &Action) {
        let values = action.as_slice();
        let entities = world.resource::<ArmJointEntities>().0.clone();
        for (i, entity) in entities.iter().enumerate() {
            if i < values.len()
                && let Some(mut cmd) = world.get_mut::<JointCommand>(*entity)
            {
                cmd.value = values[i];
            }
        }
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "ArmApplicator"
    }
}

fn main() {
    println!("=== 6-DOF Arm Gym Server ===\n");

    let max_steps: u32 = 300;
    let num_joints: usize = 6;
    let address = "127.0.0.1:9879";

    // 1. Setup arm
    let setup = setup_arm(ArmSetupConfig {
        max_episode_steps: max_steps,
        sensor_dof: num_joints,
        ..ArmSetupConfig::default()
    });
    let mut scene = setup.scene;

    // Store joint entities as a resource for the applicator
    scene
        .app
        .insert_resource(ArmJointEntities(setup.joint_entities.clone()));

    println!("Robot: six_dof_arm");
    println!("DOF:   {num_joints}");
    println!("Joint names: {:?}", setup.arm_joint_names);

    // 2. Create gym environment
    let obs_dim = num_joints * 2; // 6 pos + 6 vel = 12
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };
    let act_space = ActionSpace::Box {
        low: vec![-std::f32::consts::PI; num_joints],
        high: vec![std::f32::consts::PI; num_joints],
    };

    let joint_entities = setup.joint_entities;

    let mut env = GymEnv::new(scene.app, obs_space, act_space, Box::new(ArmApplicator))
        .with_reset_fn(move |world: &mut World| {
            // Reset rapier rigid body positions and velocities
            if let Some(mut ctx) = world.remove_resource::<RapierContext>() {
                ctx.reset_to_initial();
                world.insert_resource(ctx);
            }
            // Reset joint states and commands for arm joints
            for &entity in &joint_entities {
                if let Some(mut state) = world.get_mut::<JointState>(entity) {
                    state.position = 0.0;
                    state.velocity = 0.0;
                }
                if let Some(mut cmd) = world.get_mut::<JointCommand>(entity) {
                    cmd.value = 0.0;
                }
            }
        });

    // 3. Start server
    let server = GymServer::bind(address).expect("failed to bind server");
    let addr = server.local_addr().expect("failed to get address");
    println!("\nArm gym server listening on {addr}");
    println!("joints={num_joints}, obs_dim={obs_dim}, act_dim={num_joints}, max_steps={max_steps}");
    println!("Physics: Rapier3D, position-mode PID (kp=100, kd=10)");
    println!(
        "Connect with: python python/examples/arm_imitation_learning.py --online --port 9879\n"
    );

    loop {
        println!("waiting for client...");
        match server.serve_one(&mut env) {
            Ok(()) => println!("client disconnected cleanly"),
            Err(e) => eprintln!("client error: {e}"),
        }
    }
}
