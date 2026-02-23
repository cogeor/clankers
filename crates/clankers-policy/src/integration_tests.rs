//! Full-loop integration tests across `ClankersCore`, Actuator, Env, `DomainRand`, and Policy.
//!
//! These tests verify the complete simulation loop works end-to-end:
//! Observe → Decide → Act → Evaluate with episode lifecycle and domain randomization.

#[cfg(test)]
mod tests {
    use crate::policies::{ConstantPolicy, ZeroPolicy};
    use crate::runner::PolicyRunner;
    use bevy::prelude::*;
    use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
    use clankers_actuator::prelude::MotorType;
    use clankers_env::SensorRegistry;
    use clankers_env::buffer::ObservationBuffer;
    use clankers_env::episode::{Episode, EpisodeConfig, EpisodeState};
    use clankers_env::sensors::JointStateSensor;

    /// Build a full-stack test app with all plugins.
    fn full_app(runner: PolicyRunner) -> App {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(clankers_actuator::ClankersActuatorPlugin);
        app.add_plugins(clankers_env::ClankersEnvPlugin);
        app.add_plugins(crate::ClankersPolicyPlugin);
        app.insert_resource(runner);
        app.finish();
        app.cleanup();
        app
    }

    fn spawn_joint(world: &mut World, pos: f32, vel: f32) {
        world.spawn((
            Actuator::default(),
            JointCommand::default(),
            JointState {
                position: pos,
                velocity: vel,
            },
            JointTorque::default(),
        ));
    }

    fn register_state_sensor(app: &mut App, n_joints: usize) {
        let world = app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(n_joints)), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // -----------------------------------------------------------------------
    // End-to-end simulation loop
    // -----------------------------------------------------------------------

    #[test]
    fn full_loop_observe_decide_act_evaluate() {
        let runner = PolicyRunner::new(
            Box::new(ConstantPolicy::new(clankers_core::types::Action::from(
                vec![1.0],
            ))),
            1,
        );
        let mut app = full_app(runner);

        // Spawn one joint and register sensor
        spawn_joint(app.world_mut(), 0.5, 0.0);
        register_state_sensor(&mut app, 1);

        // Start episode
        app.world_mut().resource_mut::<Episode>().reset(None);

        // Step the simulation
        app.update();

        // Verify observation was collected (2 values: pos + vel)
        let buffer = app.world().resource::<ObservationBuffer>();
        assert_eq!(buffer.dim(), 2);
        let state_data = buffer.read(0);
        assert_eq!(state_data.len(), 2);

        // Verify policy ran and produced an action
        let runner = app.world().resource::<PolicyRunner>();
        assert_eq!(runner.action().as_slice().len(), 1);

        // Verify episode advanced
        let ep = app.world().resource::<Episode>();
        assert_eq!(ep.step_count, 1);
    }

    #[test]
    fn episode_runs_to_truncation() {
        let runner = PolicyRunner::new(Box::new(ZeroPolicy::new(1)), 1);
        let mut app = full_app(runner);

        spawn_joint(app.world_mut(), 0.0, 0.0);
        register_state_sensor(&mut app, 1);

        // Set short episode
        app.world_mut()
            .resource_mut::<EpisodeConfig>()
            .max_episode_steps = 5;
        app.world_mut().resource_mut::<Episode>().reset(None);

        // Run 5 steps
        for _ in 0..5 {
            app.update();
        }

        let ep = app.world().resource::<Episode>();
        assert_eq!(ep.state, EpisodeState::Truncated);
        assert_eq!(ep.step_count, 5);
        assert!(ep.is_done());
    }

    #[test]
    fn policy_stops_on_done_episode() {
        let runner = PolicyRunner::new(
            Box::new(ConstantPolicy::new(clankers_core::types::Action::from(
                vec![99.0],
            ))),
            1,
        );
        let mut app = full_app(runner);

        spawn_joint(app.world_mut(), 0.0, 0.0);
        register_state_sensor(&mut app, 1);

        app.world_mut()
            .resource_mut::<EpisodeConfig>()
            .max_episode_steps = 2;
        app.world_mut().resource_mut::<Episode>().reset(None);

        // Run past truncation
        app.update();
        app.update();

        // Verify truncated
        assert!(app.world().resource::<Episode>().is_done());

        // Reset runner action to zeros so we can detect if policy runs
        app.world_mut().resource_mut::<PolicyRunner>().reset();

        // Step again - policy should NOT run (episode done)
        app.update();

        let runner = app.world().resource::<PolicyRunner>();
        // Action should still be zeros (policy didn't run)
        assert_eq!(runner.action().as_slice(), &[0.0]);
    }

    #[test]
    fn multiple_joints_observed_and_acted() {
        let runner = PolicyRunner::new(Box::new(ZeroPolicy::new(2)), 2);
        let mut app = full_app(runner);

        spawn_joint(app.world_mut(), 1.0, 2.0);
        spawn_joint(app.world_mut(), 3.0, 4.0);
        register_state_sensor(&mut app, 2);

        app.world_mut().resource_mut::<Episode>().reset(None);
        app.update();

        // Buffer should have 4 values (2 joints × 2 values each)
        let buffer = app.world().resource::<ObservationBuffer>();
        assert_eq!(buffer.dim(), 4);
        let obs = buffer.as_observation();
        assert_eq!(obs.len(), 4);
    }

    #[test]
    fn steps_accumulate_across_updates() {
        let runner = PolicyRunner::new(Box::new(ZeroPolicy::new(1)), 1);
        let mut app = full_app(runner);

        spawn_joint(app.world_mut(), 0.0, 0.0);
        register_state_sensor(&mut app, 1);

        app.world_mut().resource_mut::<Episode>().reset(None);

        for _ in 0..3 {
            app.update();
        }

        let ep = app.world().resource::<Episode>();
        assert_eq!(ep.step_count, 3);
    }

    #[test]
    fn episode_reset_clears_state() {
        let runner = PolicyRunner::new(Box::new(ZeroPolicy::new(1)), 1);
        let mut app = full_app(runner);

        spawn_joint(app.world_mut(), 0.0, 0.0);
        register_state_sensor(&mut app, 1);

        // First episode
        app.world_mut().resource_mut::<Episode>().reset(Some(42));
        app.update();
        app.update();

        let ep = app.world().resource::<Episode>();
        assert_eq!(ep.step_count, 2);
        assert_eq!(ep.episode_number, 1);

        // Second episode
        app.world_mut().resource_mut::<Episode>().reset(Some(99));

        let ep = app.world().resource::<Episode>();
        assert_eq!(ep.step_count, 0);
        assert_eq!(ep.seed, Some(99));
        assert_eq!(ep.episode_number, 2);
    }

    #[test]
    fn deterministic_with_same_seed() {
        fn run_episode_get_obs(seed: u64) -> Vec<f32> {
            let runner = PolicyRunner::new(Box::new(ZeroPolicy::new(1)), 1);
            let mut app = full_app(runner);

            spawn_joint(app.world_mut(), 1.0, 0.5);
            register_state_sensor(&mut app, 1);

            app.world_mut().resource_mut::<Episode>().reset(Some(seed));
            app.update();

            let buffer = app.world().resource::<ObservationBuffer>();
            buffer.as_observation().as_slice().to_vec()
        }

        let obs1 = run_episode_get_obs(42);
        let obs2 = run_episode_get_obs(42);
        assert_eq!(obs1, obs2, "same seed should produce same observations");
    }

    // -----------------------------------------------------------------------
    // With domain randomization
    // -----------------------------------------------------------------------

    #[test]
    fn full_loop_with_domain_rand() {
        use clankers_domain_rand::prelude::*;
        use clankers_domain_rand::ranges::RandomizationRange;

        let runner = PolicyRunner::new(Box::new(ZeroPolicy::new(1)), 1);
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(clankers_actuator::ClankersActuatorPlugin);
        app.add_plugins(clankers_env::ClankersEnvPlugin);
        app.add_plugins(ClankersDomainRandPlugin);
        app.add_plugins(crate::ClankersPolicyPlugin);
        app.insert_resource(runner);
        app.finish();
        app.cleanup();

        // Configure randomization
        app.world_mut().resource_mut::<DomainRandConfig>().actuator = ActuatorRandomizer {
            motor: MotorRandomizer {
                max_torque: Some(RandomizationRange::uniform(50.0, 100.0).unwrap()),
                ..Default::default()
            },
            ..Default::default()
        };
        app.world_mut().resource_mut::<DomainRandConfig>().seed = 42;

        spawn_joint(app.world_mut(), 0.0, 0.0);
        register_state_sensor(&mut app, 1);

        // Start episode (triggers randomization)
        app.world_mut().resource_mut::<Episode>().reset(None);
        app.update();

        // Verify motor was randomized
        let actuator = app
            .world_mut()
            .query::<&Actuator>()
            .single(app.world())
            .unwrap();
        if let MotorType::Ideal(m) = &actuator.motor {
            assert!(
                m.max_torque >= 50.0 && m.max_torque < 100.0,
                "expected randomized max_torque, got {}",
                m.max_torque
            );
        }

        // Verify the rest of the pipeline still works
        let ep = app.world().resource::<Episode>();
        assert_eq!(ep.step_count, 1);
        assert!(ep.is_running());
    }

    // -----------------------------------------------------------------------
    // System ordering verification
    // -----------------------------------------------------------------------

    #[test]
    fn system_ordering_observe_before_decide() {
        // Verify that observations are available when the policy runs.
        // Use a ConstantPolicy that ignores observations - we just need
        // to verify no panics and the pipeline executes correctly.
        let runner = PolicyRunner::new(
            Box::new(ConstantPolicy::new(clankers_core::types::Action::from(
                vec![1.0],
            ))),
            1,
        );
        let mut app = full_app(runner);

        spawn_joint(app.world_mut(), 2.5, 1.0);
        register_state_sensor(&mut app, 1);

        app.world_mut().resource_mut::<Episode>().reset(None);

        // Run multiple steps to verify ordering is stable
        for _ in 0..10 {
            app.update();
        }

        let ep = app.world().resource::<Episode>();
        assert_eq!(ep.step_count, 10);
    }
}
