//! Multi-robot scene with independent control. Thin wrapper over
//! [`clankers_sim::scenarios::multi_robot::MultiRobotScenario`].
//!
//! Run: `cargo run -p clankers-examples --bin multi_robot`

use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::traits::Sensor;
use clankers_core::types::{RobotGroup, RobotId};
use clankers_env::prelude::{Episode, RobotJointStateSensor};
use clankers_sim::scenarios::multi_robot::{MultiRobotConfig, MultiRobotScenario};
use clankers_sim::{EpisodeStats, ScenarioConfig};

fn main() {
    println!("=== Multi-Robot Scene Example ===\n");
    let mut art =
        MultiRobotScenario::build_with(&ScenarioConfig::default(), &MultiRobotConfig::default());
    println!("Robots in scene: {}", art.scene.robots.len());
    for (nm, bot) in &art.scene.robots {
        println!("  {nm} — {} joints", bot.joint_count());
    }
    let grp = art.scene.app.world().resource::<RobotGroup>();
    for id in 0..grp.len() {
        let info = grp.get(RobotId(id as u32)).unwrap();
        println!(
            "  RobotId({id}): '{}' — {} joints",
            info.name(),
            info.joint_count()
        );
    }
    art.scene
        .app
        .world_mut()
        .resource_mut::<Episode>()
        .reset(Some(42));
    let piv = art.scene.robots["pendulum"].joint_entity("pivot").unwrap();
    let sho = art.scene.robots["two_link_arm"]
        .joint_entity("shoulder")
        .unwrap();
    let elb = art.scene.robots["two_link_arm"]
        .joint_entity("elbow")
        .unwrap();
    let six_names = [
        "j1_base_yaw",
        "j2_shoulder_pitch",
        "j3_elbow_pitch",
        "j4_forearm_roll",
        "j5_wrist_pitch",
        "j6_wrist_roll",
    ];
    for step in 0..30 {
        let tm = step as f32 * 0.02;
        art.scene
            .app
            .world_mut()
            .get_mut::<JointCommand>(piv)
            .unwrap()
            .value = 8.0 * (tm * 5.0).sin();
        art.scene
            .app
            .world_mut()
            .get_mut::<JointCommand>(sho)
            .unwrap()
            .value = 10.0;
        art.scene
            .app
            .world_mut()
            .get_mut::<JointCommand>(elb)
            .unwrap()
            .value = 5.0 * (tm * 2.0).cos();
        for jn in six_names {
            if let Some(ent) = art.scene.robots["six_dof_arm"].joint_entity(jn) {
                art.scene
                    .app
                    .world_mut()
                    .get_mut::<JointCommand>(ent)
                    .unwrap()
                    .value = 2.0;
            }
        }
        art.scene.app.update();
        if step % 10 == 0 {
            let pp = art
                .scene
                .app
                .world()
                .get::<JointState>(piv)
                .unwrap()
                .position;
            let ss = art
                .scene
                .app
                .world()
                .get::<JointState>(sho)
                .unwrap()
                .position;
            let ee = art
                .scene
                .app
                .world()
                .get::<JointState>(elb)
                .unwrap()
                .position;
            println!("  step {step:2}: pendulum={pp:+5.3}  shoulder={ss:+5.3}  elbow={ee:+5.3}");
        }
        if art.scene.app.world().resource::<Episode>().is_done() {
            break;
        }
    }
    let mut pend_sensor = RobotJointStateSensor::new(RobotId(0), art.layouts["pendulum"].clone());
    let pend_obs = pend_sensor.read(art.scene.app.world_mut());
    println!(
        "\nPendulum state sensor: {} values — pos={:.3} vel={:.3}",
        pend_obs.len(),
        pend_obs[0],
        pend_obs[1]
    );
    let stats = art.scene.app.world().resource::<EpisodeStats>();
    println!(
        "\nEpisodes: {}  Total steps: {}\nMulti-robot scene example PASSED",
        stats.episodes_completed, stats.total_steps
    );
}
