//! Headless simulation smoke tests.
//!
//! Verifies that the full Clankers stack runs correctly in headless mode
//! (no window, no GPU, pure ECS) across all modules: core, actuator, env,
//! policy, domain-rand, teleop, render, and gym.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
    use clankers_core::traits::ActionApplicator;
    use clankers_core::types::{Action, ActionSpace, ObservationSpace};
    use clankers_domain_rand::prelude::*;
    use clankers_env::prelude::*;
    use clankers_gym::GymEnv;
    use clankers_render::prelude::*;
    use clankers_teleop::prelude::*;

    use crate::builder::SceneBuilder;
    use crate::stats::EpisodeStats;

    const PENDULUM_URDF: &str = r#"
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

    // -------------------------------------------------------------------
    // Full stack headless test
    // -------------------------------------------------------------------

    #[test]
    fn full_stack_headless_simulation() {
        // Build scene with URDF robot
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(10)
            .with_robot_urdf(PENDULUM_URDF, HashMap::new())
            .unwrap()
            .build();

        // Add teleop plugin
        scene.app.add_plugins(ClankersTeleopPlugin);

        // Add render plugin with small frame buffer
        scene
            .app
            .insert_resource(RenderConfig::new(4, 4).with_format(PixelFormat::Rgb8));
        scene.app.add_plugins(ClankersRenderPlugin);

        // Finish app setup
        scene.app.finish();
        scene.app.cleanup();

        // Configure teleop mapping
        let pivot = scene.robots["pendulum"].joint_entity("pivot").unwrap();
        *scene.app.world_mut().resource_mut::<TeleopConfig>() =
            TeleopConfig::new().with_mapping("axis_0", JointMapping::new(pivot).with_scale(2.0));

        // Run a full episode
        scene.app.world_mut().resource_mut::<Episode>().reset(None);

        for i in 0_u8..10 {
            // Drive teleop with a varying signal
            let signal = f32::from(i) * 0.1;
            scene
                .app
                .world_mut()
                .resource_mut::<TeleopCommander>()
                .set("axis_0", signal);

            scene.app.update();
        }

        // Verify episode completed
        let stats = scene.app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 1);

        // Verify frame buffer exists
        let buf = scene.app.world().resource::<FrameBuffer>();
        assert_eq!(buf.width(), 4);
        assert_eq!(buf.height(), 4);

        // Verify teleop resources exist
        assert!(
            scene
                .app
                .world()
                .get_resource::<TeleopCommander>()
                .is_some()
        );
        assert!(scene.app.world().get_resource::<TeleopConfig>().is_some());
    }

    // -------------------------------------------------------------------
    // Gym env headless test
    // -------------------------------------------------------------------

    struct JointCommandApplicator;

    impl ActionApplicator for JointCommandApplicator {
        fn apply(&self, world: &mut bevy::prelude::World, action: &Action) {
            let values = action.as_slice();
            let mut query = world.query::<&mut JointCommand>();
            for (i, mut cmd) in query.iter_mut(world).enumerate() {
                if i < values.len() {
                    cmd.value = values[i];
                }
            }
        }

        #[allow(clippy::unnecessary_literal_bound)]
        fn name(&self) -> &str {
            "JointCommandApplicator"
        }
    }

    #[test]
    fn gym_env_headless_episode() {
        let mut app = bevy::prelude::App::new();
        app.add_plugins(crate::ClankersSimPlugin);

        // Spawn a joint
        app.world_mut().spawn((
            Actuator::default(),
            JointCommand::default(),
            JointState::default(),
            JointTorque::default(),
        ));

        // Register a sensor
        {
            let world = app.world_mut();
            let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
            let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
            registry.register(Box::new(JointStateSensor::new(1)), &mut buffer);
            world.insert_resource(buffer);
            world.insert_resource(registry);
        }

        // Set max steps
        app.world_mut()
            .resource_mut::<EpisodeConfig>()
            .max_episode_steps = 5;

        let obs_space = ObservationSpace::Box {
            low: vec![-10.0, -10.0],
            high: vec![10.0, 10.0],
        };
        let act_space = ActionSpace::Box {
            low: vec![-1.0],
            high: vec![1.0],
        };

        let mut env = GymEnv::new(app, obs_space, act_space, Box::new(JointCommandApplicator));

        // Reset
        let reset = env.reset(Some(42));
        assert_eq!(reset.observation.len(), 2);

        // Step until truncation
        for _ in 0..5 {
            let result = env.step(&Action::Continuous(vec![0.5]));
            if result.truncated {
                assert_eq!(result.info.episode_length, 5);
                return;
            }
        }

        panic!("expected truncation after 5 steps");
    }

    // -------------------------------------------------------------------
    // Multiple plugins coexist headless
    // -------------------------------------------------------------------

    #[test]
    fn all_plugins_coexist_headless() {
        let mut app = bevy::prelude::App::new();

        // Add all auto-init plugins (PolicyPlugin excluded — it requires
        // a manually inserted PolicyRunner resource)
        app.add_plugins(crate::ClankersSimPlugin);
        app.add_plugins(ClankersTeleopPlugin);
        app.add_plugins(ClankersRenderPlugin);
        app.add_plugins(ClankersDomainRandPlugin);

        app.finish();
        app.cleanup();

        // Run a few updates — should not panic
        for _ in 0..5 {
            app.update();
        }

        // Verify all resources exist
        assert!(app.world().get_resource::<Episode>().is_some());
        assert!(app.world().get_resource::<TeleopCommander>().is_some());
        assert!(app.world().get_resource::<RenderConfig>().is_some());
        assert!(app.world().get_resource::<EpisodeStats>().is_some());
    }

    // -------------------------------------------------------------------
    // Render + Gym integration
    // -------------------------------------------------------------------

    #[test]
    fn render_buffer_accessible_during_gym_step() {
        let mut app = bevy::prelude::App::new();
        app.add_plugins(crate::ClankersSimPlugin);
        app.insert_resource(RenderConfig::new(2, 2));
        app.add_plugins(ClankersRenderPlugin);

        app.world_mut().spawn((
            Actuator::default(),
            JointCommand::default(),
            JointState::default(),
            JointTorque::default(),
        ));

        let obs_space = ObservationSpace::Box {
            low: vec![-1.0],
            high: vec![1.0],
        };
        let act_space = ActionSpace::Box {
            low: vec![-1.0],
            high: vec![1.0],
        };

        let mut env = GymEnv::new(app, obs_space, act_space, Box::new(JointCommandApplicator));

        env.reset(None);
        env.step(&Action::Continuous(vec![0.0]));

        // Frame buffer should be accessible through the app
        let buf = env.app().world().resource::<FrameBuffer>();
        assert_eq!(buf.width(), 2);
        assert_eq!(buf.height(), 2);
    }

    // -------------------------------------------------------------------
    // Domain rand + teleop coexist
    // -------------------------------------------------------------------

    #[test]
    fn domain_rand_and_teleop_coexist() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(5)
            .with_robot_urdf(PENDULUM_URDF, HashMap::new())
            .unwrap()
            .build();

        scene.app.add_plugins(ClankersTeleopPlugin);
        scene.app.add_plugins(ClankersDomainRandPlugin);
        scene.app.finish();
        scene.app.cleanup();

        let pivot = scene.robots["pendulum"].joint_entity("pivot").unwrap();
        *scene.app.world_mut().resource_mut::<TeleopConfig>() =
            TeleopConfig::new().with_mapping("axis_0", JointMapping::new(pivot));

        // Run episode with both teleop and domain rand active
        scene.app.world_mut().resource_mut::<Episode>().reset(None);
        for _ in 0..5 {
            scene
                .app
                .world_mut()
                .resource_mut::<TeleopCommander>()
                .set("axis_0", 1.0);
            scene.app.update();
        }

        let stats = scene.app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 1);
    }
}
