//! Policy runner resource and Bevy system for the decide phase.
//!
//! [`PolicyRunner`] holds a boxed [`Policy`] and the current action.
//! The [`policy_decide_system`] reads the observation buffer, queries the
//! policy, and stores the resulting action.

use bevy::prelude::*;
use clankers_core::traits::Policy;
use clankers_core::types::{Action, Observation};
use clankers_env::buffer::ObservationBuffer;
use clankers_env::episode::Episode;

// ---------------------------------------------------------------------------
// PolicyRunner
// ---------------------------------------------------------------------------

/// Resource that drives policy inference each step.
///
/// Holds a boxed policy, the latest observation, and the current action.
#[derive(Resource)]
pub struct PolicyRunner {
    policy: Box<dyn Policy>,
    current_action: Action,
    action_dim: usize,
}

impl PolicyRunner {
    /// Create a new policy runner with the given policy and action dimension.
    pub fn new(policy: Box<dyn Policy>, action_dim: usize) -> Self {
        Self {
            policy,
            current_action: Action::zeros(action_dim),
            action_dim,
        }
    }

    /// Get the current action (result of last `decide`).
    pub const fn action(&self) -> &Action {
        &self.current_action
    }

    /// Get the policy name.
    pub fn policy_name(&self) -> &str {
        self.policy.name()
    }

    /// Run the policy on the given observation and update the current action.
    pub fn decide(&mut self, obs: &Observation) {
        self.current_action = self.policy.get_action(obs);
    }

    /// Reset the current action to zeros.
    pub fn reset(&mut self) {
        self.current_action = Action::zeros(self.action_dim);
    }
}

// ---------------------------------------------------------------------------
// policy_decide_system
// ---------------------------------------------------------------------------

/// System that reads the observation buffer, runs the policy, and updates
/// the current action in [`PolicyRunner`].
///
/// Only runs when the episode is active (Running state).
/// Runs in [`ClankersSet::Decide`](clankers_core::ClankersSet::Decide).
#[allow(clippy::needless_pass_by_value)]
pub fn policy_decide_system(
    mut runner: ResMut<PolicyRunner>,
    buffer: Res<ObservationBuffer>,
    episode: Res<Episode>,
) {
    if !episode.is_running() {
        return;
    }
    let obs = buffer.as_observation();
    runner.decide(&obs);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::ZeroPolicy;
    use clankers_core::ClankersSet;

    #[test]
    fn policy_runner_new() {
        let runner = PolicyRunner::new(Box::new(ZeroPolicy::new(3)), 3);
        assert_eq!(runner.action().as_slice(), &[0.0, 0.0, 0.0]);
        assert_eq!(runner.policy_name(), "ZeroPolicy");
    }

    #[test]
    fn policy_runner_decide() {
        let mut runner = PolicyRunner::new(Box::new(ZeroPolicy::new(2)), 2);
        let obs = Observation::new(vec![1.0, 2.0, 3.0]);
        runner.decide(&obs);
        assert_eq!(runner.action().as_slice(), &[0.0, 0.0]);
    }

    #[test]
    fn policy_runner_reset() {
        let mut runner = PolicyRunner::new(
            Box::new(crate::policies::ConstantPolicy::new(Action::from(vec![
                5.0, 10.0,
            ]))),
            2,
        );
        let obs = Observation::new(vec![1.0]);
        runner.decide(&obs);
        assert_eq!(runner.action().as_slice(), &[5.0, 10.0]);

        runner.reset();
        assert_eq!(runner.action().as_slice(), &[0.0, 0.0]);
    }

    fn build_test_app(runner: PolicyRunner) -> App {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(clankers_env::ClankersEnvPlugin);
        app.insert_resource(runner);
        app.add_systems(Update, policy_decide_system.in_set(ClankersSet::Decide));
        app.finish();
        app.cleanup();
        app
    }

    #[test]
    fn system_runs_policy_on_running_episode() {
        let runner = PolicyRunner::new(
            Box::new(crate::policies::ConstantPolicy::new(Action::from(vec![
                7.0,
            ]))),
            1,
        );
        let mut app = build_test_app(runner);

        // Start episode
        app.world_mut().resource_mut::<Episode>().reset(None);
        app.update();

        let runner = app.world().resource::<PolicyRunner>();
        assert_eq!(runner.action().as_slice(), &[7.0]);
    }

    #[test]
    fn system_skips_when_not_running() {
        let runner = PolicyRunner::new(
            Box::new(crate::policies::ConstantPolicy::new(Action::from(vec![
                7.0,
            ]))),
            1,
        );
        let mut app = build_test_app(runner);

        // Don't start episode (Idle state)
        app.update();

        let runner = app.world().resource::<PolicyRunner>();
        assert_eq!(runner.action().as_slice(), &[0.0]); // Still zeros
    }
}
