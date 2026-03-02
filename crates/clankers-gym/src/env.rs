//! Gymnasium-style environment wrapper around a Bevy App.
//!
//! [`GymEnv`] drives a Bevy simulation step-by-step, exposing the standard
//! `step`/`reset` API that training loops expect.

use bevy::prelude::*;

use clankers_core::traits::ActionApplicator;
use clankers_core::types::{
    Action, ActionSpace, ContactEvent, Observation, ObservationSpace, ResetInfo, ResetResult,
    StepInfo, StepResult,
};
use clankers_env::buffer::ObservationBuffer;
use clankers_env::episode::Episode;

// ---------------------------------------------------------------------------
// GymEnv
// ---------------------------------------------------------------------------

/// Gymnasium-compatible environment wrapping a Bevy [`App`].
///
/// Owns the Bevy App and an [`ActionApplicator`] to bridge between the flat
/// action vector and the ECS world. Each call to [`step`](Self::step) applies
/// an action, runs one ECS update, then reads the observation and episode
/// state.
///
/// # Example
///
/// ```no_run
/// use bevy::prelude::*;
/// use clankers_core::types::{Action, ActionSpace, ObservationSpace};
/// use clankers_gym::env::GymEnv;
///
/// // (In practice, build a real app with plugins and an action applicator)
/// ```
/// Callback invoked on each environment reset.
type ResetFn = dyn Fn(&mut World);

/// Callback evaluated after each step to determine task success.
type SuccessFn = dyn Fn(&World) -> bool;

pub struct GymEnv {
    app: App,
    obs_space: ObservationSpace,
    act_space: ActionSpace,
    applicator: Box<dyn ActionApplicator>,
    reset_fn: Option<Box<ResetFn>>,
    success_fn: Option<Box<SuccessFn>>,
}

impl GymEnv {
    /// Create a new gym environment.
    ///
    /// The `app` must have [`ClankersEnvPlugin`](clankers_env::ClankersEnvPlugin)
    /// (or equivalent resources) already added. The first `app.update()` is
    /// **not** called here — call [`reset`](Self::reset) first.
    #[must_use]
    pub fn new(
        app: App,
        obs_space: ObservationSpace,
        act_space: ActionSpace,
        applicator: Box<dyn ActionApplicator>,
    ) -> Self {
        Self {
            app,
            obs_space,
            act_space,
            applicator,
            reset_fn: None,
            success_fn: None,
        }
    }

    /// Set a callback that resets world state (physics, joints) on episode reset.
    pub fn with_reset_fn(mut self, f: impl Fn(&mut World) + 'static) -> Self {
        self.reset_fn = Some(Box::new(f));
        self
    }

    /// Set a callback that evaluates task success after each step.
    ///
    /// The closure receives a read-only reference to the [`World`] and returns
    /// `true` when the task-specific success condition is met. The result is
    /// written to [`StepInfo::is_success`].
    pub fn with_success_fn(mut self, f: impl Fn(&World) -> bool + 'static) -> Self {
        self.success_fn = Some(Box::new(f));
        self
    }

    /// Observation space descriptor.
    #[must_use]
    pub const fn observation_space(&self) -> &ObservationSpace {
        &self.obs_space
    }

    /// Action space descriptor.
    #[must_use]
    pub const fn action_space(&self) -> &ActionSpace {
        &self.act_space
    }

    /// Reset the environment, optionally with a seed.
    ///
    /// Runs one ECS update to collect the initial observation (the
    /// `observe_system` does not gate on episode state), then transitions
    /// the [`Episode`] to `Running` so subsequent [`step`](Self::step)
    /// calls correctly advance the counter from zero.
    pub fn reset(&mut self, seed: Option<u64>) -> ResetResult {
        // Reset world state (physics, joint positions/velocities) if callback is set.
        if let Some(ref reset_fn) = self.reset_fn {
            (reset_fn)(self.app.world_mut());
        }

        // Collect initial observations without advancing the episode.
        // observe_system runs regardless of episode state, while
        // episode_step_system only advances when Running.
        self.app.update();

        // Now transition to Running so the next step() advances correctly.
        self.app.world_mut().resource_mut::<Episode>().reset(seed);

        let observation = self.current_observation();
        let step_info = Self::collect_step_info(self.app.world());
        ResetResult {
            observation,
            info: ResetInfo {
                seed,
                custom: step_info.custom,
                body_poses: step_info.body_poses,
            },
        }
    }

    /// Take one step with the given action.
    ///
    /// Applies the action via the [`ActionApplicator`], runs one ECS update,
    /// then reads the observation and episode state.
    pub fn step(&mut self, action: &Action) -> StepResult {
        // Apply action to the world
        self.applicator.apply(self.app.world_mut(), action);

        // Advance simulation by one frame
        self.app.update();

        // Read results
        let observation = self.current_observation();
        let episode = self.app.world().resource::<Episode>();
        let terminated = episode.state == clankers_env::episode::EpisodeState::Done;
        let truncated = episode.state == clankers_env::episode::EpisodeState::Truncated;
        let episode_length = episode.step_count;

        let mut info = Self::collect_step_info(self.app.world());
        info.episode_length = episode_length;
        if let Some(ref success_fn) = self.success_fn {
            info.is_success = (success_fn)(self.app.world());
        }

        StepResult {
            observation,
            reward: 0.0,
            terminated,
            truncated,
            info,
        }
    }

    /// Mutable access to the underlying [`App`].
    pub const fn app_mut(&mut self) -> &mut App {
        &mut self.app
    }

    /// Read-only access to the underlying [`App`].
    #[must_use]
    pub const fn app(&self) -> &App {
        &self.app
    }

    /// Collect body poses and contact events from Rapier physics (if present).
    fn collect_step_info(world: &World) -> StepInfo {
        use clankers_physics::rapier::RapierContext;
        use std::collections::HashMap;

        let mut info = StepInfo::default();

        let Some(ctx) = world.get_resource::<RapierContext>() else {
            return info;
        };

        // Body poses: read all named bodies from body_handles
        let mut body_poses = HashMap::new();
        for (name, &handle) in &ctx.body_handles {
            if let Some(body) = ctx.rigid_body_set.get(handle) {
                let t = body.translation();
                let r = body.rotation();
                body_poses.insert(
                    name.clone(),
                    [t.x, t.y, t.z, r.x, r.y, r.z, r.w],
                );
            }
        }
        info.body_poses = body_poses;

        // Contact events: iterate active contact pairs
        let mut contacts = Vec::new();
        // Build reverse map: collider handle -> body name
        let mut collider_to_name: HashMap<rapier3d::prelude::ColliderHandle, String> =
            HashMap::new();
        for (name, &body_handle) in &ctx.body_handles {
            if let Some(body) = ctx.rigid_body_set.get(body_handle) {
                for &collider_handle in body.colliders() {
                    collider_to_name.insert(collider_handle, name.clone());
                }
            }
        }

        let dt = ctx.integration_parameters.dt;
        for contact_pair in ctx.narrow_phase.contact_pairs() {
            if !contact_pair.has_any_active_contact() {
                continue;
            }
            let name_a = collider_to_name
                .get(&contact_pair.collider1)
                .cloned()
                .unwrap_or_default();
            let name_b = collider_to_name
                .get(&contact_pair.collider2)
                .cloned()
                .unwrap_or_default();
            // total_impulse() returns N·s; divide by dt to get force in N
            let impulse = contact_pair.total_impulse();
            let force_magnitude = if dt > 0.0 {
                impulse.length() / dt
            } else {
                impulse.length()
            };
            if !name_a.is_empty() || !name_b.is_empty() {
                contacts.push(ContactEvent {
                    body_a: name_a,
                    body_b: name_b,
                    force_magnitude,
                });
            }
        }
        info.contact_events = contacts;

        info
    }

    fn current_observation(&self) -> Observation {
        self.app
            .world()
            .get_resource::<ObservationBuffer>()
            .map_or_else(|| Observation::zeros(0), ObservationBuffer::as_observation)
    }
}

// ---------------------------------------------------------------------------
// VecEnvInstance impl — allows GymEnv to be used inside VecEnvRunner
// ---------------------------------------------------------------------------

impl clankers_env::vec_runner::VecEnvInstance for GymEnv {
    fn reset(&mut self, seed: Option<u64>) -> ResetResult {
        Self::reset(self, seed)
    }

    fn step(&mut self, action: &Action) -> StepResult {
        Self::step(self, action)
    }

    fn obs_dim(&self) -> usize {
        self.obs_space.shape().iter().product()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
    use clankers_env::prelude::*;

    /// A test action applicator that writes continuous action values to
    /// `JointCommand` components in entity order.
    struct TestApplicator;

    impl ActionApplicator for TestApplicator {
        fn apply(&self, world: &mut World, action: &Action) {
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
            "TestApplicator"
        }
    }

    fn build_test_env(num_joints: usize) -> GymEnv {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(ClankersEnvPlugin);

        // Spawn joints
        for _ in 0..num_joints {
            app.world_mut().spawn((
                Actuator::default(),
                JointCommand::default(),
                JointState::default(),
                JointTorque::default(),
            ));
        }

        // Register a sensor so we have observations
        {
            let world = app.world_mut();
            let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
            let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
            registry.register(Box::new(JointStateSensor::new(num_joints)), &mut buffer);
            world.insert_resource(buffer);
            world.insert_resource(registry);
        }

        let obs_dim = num_joints * 2; // position + velocity per joint
        let obs_space = ObservationSpace::Box {
            low: vec![-10.0; obs_dim],
            high: vec![10.0; obs_dim],
        };
        let act_space = ActionSpace::Box {
            low: vec![-1.0; num_joints],
            high: vec![1.0; num_joints],
        };

        GymEnv::new(app, obs_space, act_space, Box::new(TestApplicator))
    }

    #[test]
    fn reset_returns_observation() {
        let mut env = build_test_env(2);
        let result = env.reset(Some(42));
        assert_eq!(result.observation.len(), 4); // 2 joints * 2 (pos+vel)
        assert_eq!(result.info.seed, Some(42));
    }

    #[test]
    fn step_returns_step_result() {
        let mut env = build_test_env(2);
        env.reset(None);

        let action = Action::Continuous(vec![0.5, -0.3]);
        let result = env.step(&action);

        assert_eq!(result.observation.len(), 4);
        assert!(!result.terminated);
        assert!(!result.truncated);
        assert_eq!(result.info.episode_length, 1);
    }

    #[test]
    fn multiple_steps_accumulate() {
        let mut env = build_test_env(1);
        env.reset(None);

        let action = Action::Continuous(vec![0.1]);
        let r1 = env.step(&action);
        let r2 = env.step(&action);

        assert_eq!(r1.info.episode_length, 1);
        assert_eq!(r2.info.episode_length, 2);
    }

    #[test]
    fn truncation_after_max_steps() {
        let mut env = build_test_env(1);
        // Set max steps to 3
        env.app_mut()
            .world_mut()
            .resource_mut::<EpisodeConfig>()
            .max_episode_steps = 3;

        env.reset(None);

        let action = Action::Continuous(vec![0.0]);
        let r1 = env.step(&action);
        assert!(!r1.truncated);

        let r2 = env.step(&action);
        assert!(!r2.truncated);

        let r3 = env.step(&action);
        assert!(r3.truncated);
    }

    #[test]
    fn reset_after_done() {
        let mut env = build_test_env(1);
        env.app_mut()
            .world_mut()
            .resource_mut::<EpisodeConfig>()
            .max_episode_steps = 1;

        env.reset(None);
        let r = env.step(&Action::Continuous(vec![0.0]));
        assert!(r.truncated);

        // Reset and step again
        let reset = env.reset(Some(99));
        assert_eq!(reset.observation.len(), 2);

        let r2 = env.step(&Action::Continuous(vec![0.0]));
        assert!(r2.truncated);
        assert_eq!(r2.info.episode_length, 1);
    }

    #[test]
    fn spaces_accessible() {
        let env = build_test_env(2);
        assert_eq!(env.observation_space().shape(), vec![4]);
        assert_eq!(env.action_space().shape(), vec![2]);
    }
}
