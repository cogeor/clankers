//! End-to-end integration tests for the full Clankers simulation stack.
//!
//! These tests exercise the complete pipeline: scene construction, robot
//! spawning, episode lifecycle, policy integration, domain randomization,
//! and statistics tracking across multiple episodes.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use clankers_actuator::clankers_actuator_core::prelude::MotorType;
    use clankers_actuator::components::{Actuator, JointCommand, JointState};
    use clankers_core::config::SimConfig;
    use clankers_core::types::{RobotGroup, RobotId};
    use clankers_domain_rand::prelude::*;
    use clankers_env::episode::{Episode, EpisodeConfig, EpisodeState};
    use clankers_env::sensors::{RobotJointCommandSensor, RobotJointStateSensor};
    use clankers_policy::prelude::*;

    use crate::builder::SceneBuilder;
    use crate::stats::EpisodeStats;

    // -----------------------------------------------------------------------
    // Shared URDF fixtures
    // -----------------------------------------------------------------------

    const SINGLE_JOINT_URDF: &str = r#"
        <robot name="pendulum">
            <link name="base"/>
            <link name="arm"/>
            <joint name="pivot" type="revolute">
                <parent link="base"/>
                <child link="arm"/>
                <axis xyz="0 0 1"/>
                <limit lower="-3.14" upper="3.14" effort="10" velocity="5"/>
                <dynamics damping="0.1" friction="0.05"/>
            </joint>
        </robot>
    "#;

    const TWO_JOINT_URDF: &str = r#"
        <robot name="arm">
            <link name="base"/>
            <link name="link1"/>
            <link name="link2"/>
            <joint name="shoulder" type="revolute">
                <parent link="base"/>
                <child link="link1"/>
                <axis xyz="0 0 1"/>
                <limit lower="-1.57" upper="1.57" effort="50" velocity="3"/>
                <dynamics damping="0.5" friction="0.1"/>
            </joint>
            <joint name="elbow" type="revolute">
                <parent link="link1"/>
                <child link="link2"/>
                <axis xyz="0 1 0"/>
                <limit lower="-2.0" upper="2.0" effort="30" velocity="5"/>
                <dynamics damping="0.3" friction="0.05"/>
            </joint>
        </robot>
    "#;

    // -----------------------------------------------------------------------
    // Full pipeline: build → run episodes → verify stats
    // -----------------------------------------------------------------------

    #[test]
    fn full_episode_lifecycle() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(10)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        // Start episode
        scene.app.world_mut().resource_mut::<Episode>().reset(None);

        // Run to completion
        for _ in 0..10 {
            scene.app.update();
        }

        let episode = scene.app.world().resource::<Episode>();
        assert!(episode.state.is_terminal());

        let stats = scene.app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 1);
        assert_eq!(stats.total_steps, 10);
    }

    #[test]
    fn multiple_episodes_accumulate_stats() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(5)
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        for _ in 0..4 {
            scene.app.world_mut().resource_mut::<Episode>().reset(None);
            for _ in 0..5 {
                scene.app.update();
            }
        }

        let stats = scene.app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 4);
        assert_eq!(stats.total_steps, 20);
        assert_eq!(stats.step_history.len(), 4);
        assert!(stats.step_history.iter().all(|&s| s == 5));
    }

    // -----------------------------------------------------------------------
    // Custom SimConfig propagation
    // -----------------------------------------------------------------------

    #[test]
    fn custom_sim_config_propagates() {
        let config = SimConfig {
            physics_dt: 0.005,
            control_dt: 0.05,
            seed: 42,
            ..SimConfig::default()
        };
        let scene = SceneBuilder::new().with_sim_config(config).build();

        let cfg = scene.app.world().resource::<SimConfig>();
        assert!((cfg.physics_dt - 0.005).abs() < f64::EPSILON);
        assert!((cfg.control_dt - 0.05).abs() < f64::EPSILON);
        assert_eq!(cfg.seed, 42);
    }

    // -----------------------------------------------------------------------
    // Robot spawning with initial positions
    // -----------------------------------------------------------------------

    #[test]
    fn robot_spawned_with_initial_positions() {
        let mut positions = HashMap::new();
        positions.insert("shoulder".into(), 0.5_f32);
        positions.insert("elbow".into(), -1.0_f32);

        let scene = SceneBuilder::new()
            .with_robot_urdf(TWO_JOINT_URDF, positions)
            .unwrap()
            .build();

        let bot = &scene.robots["arm"];

        let shoulder = bot.joint_entity("shoulder").unwrap();
        let state = scene.app.world().get::<JointState>(shoulder).unwrap();
        assert!((state.position - 0.5).abs() < f32::EPSILON);

        let elbow = bot.joint_entity("elbow").unwrap();
        let state = scene.app.world().get::<JointState>(elbow).unwrap();
        assert!((state.position - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn actuator_configured_from_urdf_limits() {
        let scene = SceneBuilder::new()
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        let bot = &scene.robots["pendulum"];
        let entity = bot.joint_entity("pivot").unwrap();
        let actuator = scene.app.world().get::<Actuator>(entity).unwrap();

        if let MotorType::Ideal(motor) = &actuator.motor {
            assert!((motor.max_torque - 10.0).abs() < f32::EPSILON);
            assert!((motor.max_velocity - 5.0).abs() < f32::EPSILON);
        } else {
            panic!("expected IdealMotor");
        }

        assert!((actuator.friction.viscous - 0.1).abs() < f32::EPSILON);
        assert!((actuator.friction.coulomb - 0.05).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Multi-robot scene
    // -----------------------------------------------------------------------

    #[test]
    fn multi_robot_scene_runs_episode() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(5)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        assert_eq!(scene.robots.len(), 2);
        assert_eq!(scene.robots["pendulum"].joint_count(), 1);
        assert_eq!(scene.robots["arm"].joint_count(), 2);

        scene.app.world_mut().resource_mut::<Episode>().reset(None);
        for _ in 0..5 {
            scene.app.update();
        }

        let stats = scene.app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 1);
    }

    // -----------------------------------------------------------------------
    // Policy integration
    // -----------------------------------------------------------------------

    #[test]
    fn scene_with_zero_policy() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(5)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        // Add policy plugin + runner after build
        scene.app.add_plugins(ClankersPolicyPlugin);
        scene
            .app
            .insert_resource(PolicyRunner::new(Box::new(ZeroPolicy::new(1)), 1));

        scene.app.world_mut().resource_mut::<Episode>().reset(None);
        for _ in 0..5 {
            scene.app.update();
        }

        let runner = scene.app.world().resource::<PolicyRunner>();
        assert_eq!(runner.policy_name(), "ZeroPolicy");

        let stats = scene.app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 1);
    }

    #[test]
    fn scene_with_constant_policy() {
        use clankers_core::types::Action;

        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(3)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        let action = Action::from(vec![5.0]);
        scene.app.add_plugins(ClankersPolicyPlugin);
        scene
            .app
            .insert_resource(PolicyRunner::new(Box::new(ConstantPolicy::new(action)), 1));

        scene.app.world_mut().resource_mut::<Episode>().reset(None);
        for _ in 0..3 {
            scene.app.update();
        }

        let runner = scene.app.world().resource::<PolicyRunner>();
        assert_eq!(runner.action().as_slice(), &[5.0]);
    }

    // -----------------------------------------------------------------------
    // Domain randomization integration
    // -----------------------------------------------------------------------

    #[test]
    fn scene_with_domain_randomization() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(3)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        // Add domain rand plugin
        scene.app.add_plugins(ClankersDomainRandPlugin);

        // Configure motor torque randomization
        let motor_rand = MotorRandomizer {
            max_torque: Some(RandomizationRange::uniform(5.0, 20.0).unwrap()),
            ..Default::default()
        };
        let config = DomainRandConfig::default()
            .with_seed(42)
            .with_actuator(ActuatorRandomizer {
                motor: motor_rand,
                ..Default::default()
            });
        scene.app.insert_resource(config);

        // Run first episode
        scene.app.world_mut().resource_mut::<Episode>().reset(None);
        scene.app.update();

        // Check that actuator was randomized
        let bot = &scene.robots["pendulum"];
        let entity = bot.joint_entity("pivot").unwrap();
        let actuator = scene.app.world().get::<Actuator>(entity).unwrap();

        if let MotorType::Ideal(motor) = &actuator.motor {
            assert!(
                motor.max_torque >= 5.0 && motor.max_torque <= 20.0,
                "expected randomized torque in [5, 20], got {}",
                motor.max_torque
            );
        } else {
            panic!("expected IdealMotor");
        }
    }

    #[test]
    fn domain_rand_changes_across_episodes() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(3)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        scene.app.add_plugins(ClankersDomainRandPlugin);

        let motor_rand = MotorRandomizer {
            max_torque: Some(RandomizationRange::uniform(1.0, 1000.0).unwrap()),
            ..Default::default()
        };
        scene
            .app
            .insert_resource(DomainRandConfig::default().with_seed(0).with_actuator(
                ActuatorRandomizer {
                    motor: motor_rand,
                    ..Default::default()
                },
            ));

        let bot_entity = scene.robots["pendulum"].joint_entity("pivot").unwrap();

        // Episode 1
        scene.app.world_mut().resource_mut::<Episode>().reset(None);
        for _ in 0..3 {
            scene.app.update();
        }
        let torque_1 = match &scene.app.world().get::<Actuator>(bot_entity).unwrap().motor {
            MotorType::Ideal(m) => m.max_torque,
            _ => panic!("expected Ideal"),
        };

        // Episode 2
        scene.app.world_mut().resource_mut::<Episode>().reset(None);
        scene.app.update();
        let torque_2 = match &scene.app.world().get::<Actuator>(bot_entity).unwrap().motor {
            MotorType::Ideal(m) => m.max_torque,
            _ => panic!("expected Ideal"),
        };

        // Different episodes should (very likely) produce different values
        assert!(
            (torque_1 - torque_2).abs() > f32::EPSILON,
            "expected different torques across episodes: {torque_1} vs {torque_2}",
        );
    }

    // -----------------------------------------------------------------------
    // Sequential episode resets
    // -----------------------------------------------------------------------

    #[test]
    fn sequential_episodes_with_manual_reset() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(3)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        // Episode 1
        scene.app.world_mut().resource_mut::<Episode>().reset(None);
        for _ in 0..3 {
            scene.app.update();
        }
        assert!(scene.app.world().resource::<Episode>().state.is_terminal());

        // Episode 2 — reset and run again
        scene
            .app
            .world_mut()
            .resource_mut::<Episode>()
            .reset(Some(42));
        for _ in 0..3 {
            scene.app.update();
        }

        let stats = scene.app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 2);
        assert_eq!(stats.total_steps, 6);
        assert_eq!(stats.step_history, vec![3, 3]);
    }

    // -----------------------------------------------------------------------
    // Joint commands affect state
    // -----------------------------------------------------------------------

    #[test]
    fn joint_commands_applied_during_episode() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(20)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        let entity = scene.robots["pendulum"].joint_entity("pivot").unwrap();

        scene.app.world_mut().resource_mut::<Episode>().reset(None);

        // Apply a torque command
        scene
            .app
            .world_mut()
            .get_mut::<JointCommand>(entity)
            .unwrap()
            .value = 5.0;

        // Step a few times
        for _ in 0..5 {
            // Re-apply command each step (normally policy does this)
            scene
                .app
                .world_mut()
                .get_mut::<JointCommand>(entity)
                .unwrap()
                .value = 5.0;
            scene.app.update();
        }

        // The actuator system should have computed a non-zero torque
        let torque = scene
            .app
            .world()
            .get::<clankers_actuator::components::JointTorque>(entity)
            .unwrap();
        // With a 5.0 Nm command and max_torque of 10.0, torque should be non-zero
        assert!(
            torque.value.abs() > f32::EPSILON,
            "expected non-zero torque, got {}",
            torque.value
        );
    }

    // -----------------------------------------------------------------------
    // Stats helper methods
    // -----------------------------------------------------------------------

    #[test]
    fn stats_mean_episode_length_after_episodes() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(3)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        // Run 2 episodes
        for _ in 0..2 {
            scene.app.world_mut().resource_mut::<Episode>().reset(None);
            for _ in 0..3 {
                scene.app.update();
            }
        }

        let stats = scene.app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 2);
        if let Some(mean) = stats.mean_episode_length() {
            assert!((mean - 3.0).abs() < f32::EPSILON);
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn empty_scene_runs_episode() {
        let mut scene = SceneBuilder::new().with_max_episode_steps(3).build();

        scene.app.world_mut().resource_mut::<Episode>().reset(None);
        for _ in 0..3 {
            scene.app.update();
        }

        let stats = scene.app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 1);
    }

    #[test]
    fn episode_idle_until_reset() {
        let scene = SceneBuilder::new()
            .with_max_episode_steps(10)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        let episode = scene.app.world().resource::<Episode>();
        assert_eq!(episode.state, EpisodeState::Idle);
        assert_eq!(episode.step_count, 0);
    }

    #[test]
    fn scene_builder_is_composable() {
        // Verify the builder can be passed around and built later
        let builder = SceneBuilder::new()
            .with_max_episode_steps(100)
            .with_sim_config(SimConfig {
                seed: 123,
                ..SimConfig::default()
            });

        let builder = builder
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap();

        let scene = builder.build();
        assert_eq!(scene.robots.len(), 1);

        let cfg = scene.app.world().resource::<SimConfig>();
        assert_eq!(cfg.seed, 123);

        let ep_cfg = scene.app.world().resource::<EpisodeConfig>();
        assert_eq!(ep_cfg.max_episode_steps, 100);
    }

    // -----------------------------------------------------------------------
    // Multi-robot identity and robot-scoped sensors
    // -----------------------------------------------------------------------

    #[test]
    fn multi_robot_robot_group_populated() {
        let scene = SceneBuilder::new()
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        let group = scene.app.world().resource::<RobotGroup>();
        assert_eq!(group.len(), 2);

        let info0 = group.get(RobotId(0)).unwrap();
        assert_eq!(info0.name(), "pendulum");
        assert_eq!(info0.joint_count(), 1);

        let info1 = group.get(RobotId(1)).unwrap();
        assert_eq!(info1.name(), "arm");
        assert_eq!(info1.joint_count(), 2);
    }

    #[test]
    fn multi_robot_entities_tagged_with_robot_id() {
        let scene = SceneBuilder::new()
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        // All pendulum joints should have RobotId(0)
        let pendulum = &scene.robots["pendulum"];
        let pivot = pendulum.joint_entity("pivot").unwrap();
        assert_eq!(scene.app.world().get::<RobotId>(pivot).unwrap().index(), 0);

        // All arm joints should have RobotId(1)
        let arm = &scene.robots["arm"];
        for name in ["shoulder", "elbow"] {
            let entity = arm.joint_entity(name).unwrap();
            assert_eq!(scene.app.world().get::<RobotId>(entity).unwrap().index(), 1);
        }
    }

    #[test]
    fn robot_scoped_sensors_read_correct_subset() {
        use clankers_core::traits::Sensor;

        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(5)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        // Set distinctive commands per robot
        let pendulum = &scene.robots["pendulum"];
        let pivot = pendulum.joint_entity("pivot").unwrap();
        scene
            .app
            .world_mut()
            .get_mut::<JointCommand>(pivot)
            .unwrap()
            .value = 7.0;

        let arm = &scene.robots["arm"];
        let shoulder = arm.joint_entity("shoulder").unwrap();
        scene
            .app
            .world_mut()
            .get_mut::<JointCommand>(shoulder)
            .unwrap()
            .value = 3.0;
        let elbow = arm.joint_entity("elbow").unwrap();
        scene
            .app
            .world_mut()
            .get_mut::<JointCommand>(elbow)
            .unwrap()
            .value = 5.0;

        // Robot 0 (pendulum) command sensor should see 1 value: 7.0
        let sensor0 = RobotJointCommandSensor::new(RobotId(0), 1);
        let obs0 = sensor0.read(scene.app.world_mut());
        assert_eq!(obs0.len(), 1);
        assert!((obs0[0] - 7.0).abs() < f32::EPSILON);

        // Robot 1 (arm) command sensor should see 2 values: 3.0 and 5.0
        let sensor1 = RobotJointCommandSensor::new(RobotId(1), 2);
        let obs1 = sensor1.read(scene.app.world_mut());
        assert_eq!(obs1.len(), 2);
        let vals: Vec<f32> = obs1.as_slice().to_vec();
        assert!(vals.contains(&3.0));
        assert!(vals.contains(&5.0));
    }

    #[test]
    fn robot_scoped_state_sensor_isolates_robots() {
        use clankers_core::traits::Sensor;

        let mut positions = HashMap::new();
        positions.insert("shoulder".into(), 1.5_f32);
        positions.insert("elbow".into(), -0.5_f32);

        let mut scene = SceneBuilder::new()
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .with_robot_urdf(TWO_JOINT_URDF, positions)
            .unwrap()
            .build();

        // Robot 0 state sensor: 1 joint → 2 values (pos, vel)
        let sensor0 = RobotJointStateSensor::new(RobotId(0), 1);
        let obs0 = sensor0.read(scene.app.world_mut());
        assert_eq!(obs0.len(), 2);
        // Default initial position is 0.0
        assert!(obs0[0].abs() < f32::EPSILON);

        // Robot 1 state sensor: 2 joints → 4 values
        let sensor1 = RobotJointStateSensor::new(RobotId(1), 2);
        let obs1 = sensor1.read(scene.app.world_mut());
        assert_eq!(obs1.len(), 4);
        let vals: Vec<f32> = obs1.as_slice().to_vec();
        assert!(vals.contains(&1.5));
        assert!(vals.contains(&-0.5));
    }

    #[test]
    fn multi_robot_scene_runs_with_independent_actions() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(5)
            .with_robot_urdf(SINGLE_JOINT_URDF, HashMap::new())
            .unwrap()
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        scene.app.world_mut().resource_mut::<Episode>().reset(None);

        // Apply different commands to each robot each step
        for step in 0..5_u16 {
            let pivot = scene.robots["pendulum"].joint_entity("pivot").unwrap();
            scene
                .app
                .world_mut()
                .get_mut::<JointCommand>(pivot)
                .unwrap()
                .value = f32::from(step) * 2.0;

            let shoulder = scene.robots["arm"].joint_entity("shoulder").unwrap();
            scene
                .app
                .world_mut()
                .get_mut::<JointCommand>(shoulder)
                .unwrap()
                .value = -1.0;

            scene.app.update();
        }

        let stats = scene.app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 1);
        assert_eq!(stats.total_steps, 5);
    }
}
