//! Multi-robot scene with independent control.
//!
//! Tests: SceneBuilder with multiple URDFs, RobotGroup, RobotId tagging,
//! robot-scoped sensors, independent joint commands per robot.
//!
//! Run: `cargo run -p clankers-examples --bin multi_robot`

use std::collections::HashMap;

use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::traits::Sensor;
use clankers_core::types::RobotGroup;
use clankers_env::prelude::*;
use clankers_examples::{PENDULUM_URDF, SIX_DOF_ARM_URDF, TWO_LINK_ARM_URDF};
use clankers_sim::{EpisodeStats, SceneBuilder};

fn main() {
    println!("=== Multi-Robot Scene Example ===\n");

    // ---------------------------------------------------------------
    // 1. Build scene with 3 different robots
    // ---------------------------------------------------------------
    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(30)
        .with_robot_urdf(PENDULUM_URDF, HashMap::new())
        .expect("failed to parse pendulum URDF")
        .with_robot_urdf(TWO_LINK_ARM_URDF, HashMap::new())
        .expect("failed to parse arm URDF")
        .with_robot_urdf(SIX_DOF_ARM_URDF, HashMap::new())
        .expect("failed to parse 6-DOF arm URDF")
        .build();

    println!("Robots in scene: {}", scene.robots.len());
    for (name, bot) in &scene.robots {
        println!("  {} — {} joints", name, bot.joint_count());
    }

    // ---------------------------------------------------------------
    // 2. Verify RobotGroup assignment
    // ---------------------------------------------------------------
    let group = scene.app.world().resource::<RobotGroup>();
    println!("\nRobotGroup ({} robots):", group.len());
    for id in 0..group.len() {
        let info = group.get(clankers_core::types::RobotId(id as u32)).unwrap();
        println!(
            "  RobotId({id}): '{}' — {} joints",
            info.name(),
            info.joint_count()
        );
    }

    // ---------------------------------------------------------------
    // 3. Run episode with independent commands per robot
    // ---------------------------------------------------------------
    println!("\n--- Running episode ---");
    scene
        .app
        .world_mut()
        .resource_mut::<Episode>()
        .reset(Some(42));

    for step in 0..30 {
        let t = step as f32 * 0.02;

        // Pendulum: sinusoidal torque
        let pivot = scene.robots["pendulum"].joint_entity("pivot").unwrap();
        scene
            .app
            .world_mut()
            .get_mut::<JointCommand>(pivot)
            .unwrap()
            .value = 8.0 * (t * 5.0).sin();

        // 2-link arm: constant shoulder, oscillating elbow
        let shoulder = scene.robots["two_link_arm"]
            .joint_entity("shoulder")
            .unwrap();
        scene
            .app
            .world_mut()
            .get_mut::<JointCommand>(shoulder)
            .unwrap()
            .value = 10.0;

        let elbow = scene.robots["two_link_arm"]
            .joint_entity("elbow")
            .unwrap();
        scene
            .app
            .world_mut()
            .get_mut::<JointCommand>(elbow)
            .unwrap()
            .value = 5.0 * (t * 2.0).cos();

        // 6-DOF arm: small constant commands on all joints
        for joint_name in ["j1_base_yaw", "j2_shoulder_pitch", "j3_elbow_pitch",
                           "j4_forearm_roll", "j5_wrist_pitch", "j6_wrist_roll"] {
            if let Some(entity) = scene.robots["six_dof_arm"].joint_entity(joint_name) {
                scene
                    .app
                    .world_mut()
                    .get_mut::<JointCommand>(entity)
                    .unwrap()
                    .value = 2.0;
            }
        }

        scene.app.update();

        if step % 10 == 0 {
            let pend_state = scene.app.world().get::<JointState>(pivot).unwrap();
            let sh_state = scene.app.world().get::<JointState>(shoulder).unwrap();
            let el_state = scene.app.world().get::<JointState>(elbow).unwrap();
            println!(
                "  step {:2}: pendulum={:+5.3}rad  shoulder={:+5.3}rad  elbow={:+5.3}rad",
                step, pend_state.position, sh_state.position, el_state.position,
            );
        }

        if scene.app.world().resource::<Episode>().is_done() {
            println!("  -> episode terminated at step {step}");
            break;
        }
    }

    // ---------------------------------------------------------------
    // 4. Test robot-scoped sensors
    // ---------------------------------------------------------------
    println!("\n--- Robot-scoped sensor reads ---");

    // Pendulum (RobotId 0): 1 joint -> 2 state values
    let pend_sensor = RobotJointStateSensor::new(clankers_core::types::RobotId(0), 1);
    let pend_obs = pend_sensor.read(scene.app.world_mut());
    println!(
        "Pendulum state sensor:  {} values — pos={:.3} vel={:.3}",
        pend_obs.len(),
        pend_obs[0],
        pend_obs[1],
    );

    // 2-link arm (RobotId 1): 2 joints -> 4 state values
    let arm_sensor = RobotJointStateSensor::new(clankers_core::types::RobotId(1), 2);
    let arm_obs = arm_sensor.read(scene.app.world_mut());
    println!(
        "Arm state sensor:       {} values — {:?}",
        arm_obs.len(),
        arm_obs.as_slice(),
    );

    // 6-DOF arm (RobotId 2): 6 joints -> 12 state values
    let six_sensor = RobotJointStateSensor::new(clankers_core::types::RobotId(2), 6);
    let six_obs = six_sensor.read(scene.app.world_mut());
    println!(
        "6-DOF arm state sensor: {} values",
        six_obs.len(),
    );

    // Robot-scoped command sensors
    let pend_cmd_sensor = RobotJointCommandSensor::new(clankers_core::types::RobotId(0), 1);
    let pend_cmd = pend_cmd_sensor.read(scene.app.world_mut());
    println!(
        "Pendulum cmd sensor:    {} values — cmd={:.2}",
        pend_cmd.len(),
        pend_cmd[0],
    );

    // Robot-scoped torque sensors
    let arm_torque_sensor = RobotJointTorqueSensor::new(clankers_core::types::RobotId(1), 2);
    let arm_torques = arm_torque_sensor.read(scene.app.world_mut());
    println!(
        "Arm torque sensor:      {} values — {:?}",
        arm_torques.len(),
        arm_torques.as_slice(),
    );

    // ---------------------------------------------------------------
    // 5. Stats
    // ---------------------------------------------------------------
    let stats = scene.app.world().resource::<EpisodeStats>();
    println!("\n=== Summary ===");
    println!("Episodes: {}  Total steps: {}", stats.episodes_completed, stats.total_steps);

    println!("\nMulti-robot scene example PASSED");
}
