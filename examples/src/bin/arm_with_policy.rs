//! 2-link arm driven by different policies.
//!
//! Tests: URDF with multiple joints, policy plugin integration,
//! Zero / Constant / Random / Scripted policies, observation buffer,
//! action-to-command bridging via a custom Bevy system.
//!
//! Run: `cargo run -p clankers-examples --bin arm_with_policy`

use std::collections::HashMap;

use bevy::prelude::*;
use clankers_actuator::components::{JointCommand, JointState, JointTorque};
use clankers_core::types::{Action, ActionSpace};
use clankers_core::ClankersSet;
use clankers_env::prelude::*;
use clankers_examples::TWO_LINK_ARM_URDF;
use clankers_policy::prelude::*;
use clankers_sim::{EpisodeStats, SceneBuilder};

/// System that copies the policy action to joint commands each step.
#[allow(clippy::needless_pass_by_value)]
fn apply_action_system(
    runner: Res<PolicyRunner>,
    episode: Res<Episode>,
    mut commands: Query<&mut JointCommand>,
) {
    if !episode.is_running() {
        return;
    }
    let values = runner.action().as_slice();
    for (i, mut cmd) in commands.iter_mut().enumerate() {
        if i < values.len() {
            cmd.value = values[i];
        }
    }
}

/// Build a scene, add a policy, run one episode, return final joint states.
fn run_with_policy(policy: Box<dyn clankers_core::traits::Policy>, name: &str) {
    println!("\n--- Policy: {name} ---");

    let max_steps = 20;
    let num_joints = 2;

    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(max_steps)
        .with_robot_urdf(TWO_LINK_ARM_URDF, HashMap::new())
        .expect("failed to parse arm URDF")
        .build();

    // Register sensors so the policy has observations to work with
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(
            Box::new(JointStateSensor::new(num_joints)),
            &mut buffer,
        );
        registry.register(
            Box::new(JointCommandSensor::new(num_joints)),
            &mut buffer,
        );
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // Add policy plugin, runner, and action applicator system
    scene.app.add_plugins(ClankersPolicyPlugin);
    scene
        .app
        .insert_resource(PolicyRunner::new(policy, num_joints));
    scene.app.add_systems(
        Update,
        apply_action_system.in_set(ClankersSet::Act),
    );

    // Run episode
    scene.app.world_mut().resource_mut::<Episode>().reset(None);
    for step in 0..max_steps {
        scene.app.update();

        if step % 5 == 0 || step == max_steps - 1 {
            let bot = &scene.robots["two_link_arm"];
            let shoulder = bot.joint_entity("shoulder").unwrap();
            let elbow = bot.joint_entity("elbow").unwrap();

            let s_state = scene.app.world().get::<JointState>(shoulder).unwrap();
            let e_state = scene.app.world().get::<JointState>(elbow).unwrap();
            let s_cmd = scene.app.world().get::<JointCommand>(shoulder).unwrap();
            let e_cmd = scene.app.world().get::<JointCommand>(elbow).unwrap();
            let s_torque = scene.app.world().get::<JointTorque>(shoulder).unwrap();
            let e_torque = scene.app.world().get::<JointTorque>(elbow).unwrap();

            println!(
                "  step {:2}: shoulder(cmd={:+6.2} pos={:+5.3} trq={:+6.2})  elbow(cmd={:+6.2} pos={:+5.3} trq={:+6.2})",
                step,
                s_cmd.value, s_state.position, s_torque.value,
                e_cmd.value, e_state.position, e_torque.value,
            );
        }

        if scene.app.world().resource::<Episode>().is_done() {
            break;
        }
    }

    let runner = scene.app.world().resource::<PolicyRunner>();
    println!("  Final action: {:?}", runner.action().as_slice());

    let stats = scene.app.world().resource::<EpisodeStats>();
    println!(
        "  Stats: {} ep, {} steps",
        stats.episodes_completed, stats.total_steps
    );
}

fn main() {
    println!("=== Arm With Policy Example ===");

    // 1. Zero policy (all commands = 0)
    run_with_policy(Box::new(ZeroPolicy::new(2)), "ZeroPolicy");

    // 2. Constant policy (fixed torque)
    let action = Action::from(vec![10.0, -5.0]);
    run_with_policy(Box::new(ConstantPolicy::new(action)), "ConstantPolicy");

    // 3. Random policy (sample from action space)
    let space = ActionSpace::Box {
        low: vec![-1.0, -1.0],
        high: vec![1.0, 1.0],
    };
    run_with_policy(
        Box::new(RandomPolicy::new(space, 42)),
        "RandomPolicy",
    );

    // 4. Scripted policy (cycling sequence)
    let actions = vec![
        Action::from(vec![5.0, 0.0]),
        Action::from(vec![0.0, 5.0]),
        Action::from(vec![-5.0, 0.0]),
        Action::from(vec![0.0, -5.0]),
    ];
    run_with_policy(Box::new(ScriptedPolicy::new(actions)), "ScriptedPolicy");

    println!("\nArm with policy example PASSED");
}
