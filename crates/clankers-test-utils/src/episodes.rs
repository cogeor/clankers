//! Episode lifecycle helpers for tests.
//!
//! Thin wrappers around common episode operations that reduce boilerplate
//! in integration tests.

use bevy::prelude::*;
use clankers_env::episode::{Episode, EpisodeState};
use clankers_env::systems::StepReward;

/// Reset the episode to `Running` with an optional seed.
pub fn reset_episode(app: &mut App, seed: Option<u64>) {
    app.world_mut().resource_mut::<Episode>().reset(seed);
}

/// Set the reward that will be consumed on the next step.
pub fn set_step_reward(app: &mut App, reward: f32) {
    app.world_mut().resource_mut::<StepReward>().0 = reward;
}

/// Run `n` simulation steps (calls `app.update()` `n` times).
pub fn step_n(app: &mut App, n: usize) {
    for _ in 0..n {
        app.update();
    }
}

/// Run until the episode ends or `max_safety_steps` is reached.
///
/// Returns the number of steps actually taken.
pub fn run_until_done(app: &mut App, max_safety_steps: usize) -> usize {
    for i in 0..max_safety_steps {
        if app.world().resource::<Episode>().is_done() {
            return i;
        }
        app.update();
    }
    max_safety_steps
}

/// Query current episode state as a tuple `(step_count, total_reward, state)`.
pub fn episode_snapshot(app: &App) -> (u32, f32, EpisodeState) {
    let ep = app.world().resource::<Episode>();
    (ep.step_count, ep.total_reward, ep.state)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::full_test_app;

    #[test]
    fn reset_episode_starts_running() {
        let mut app = full_test_app();
        reset_episode(&mut app, Some(42));

        let ep = app.world().resource::<Episode>();
        assert!(ep.is_running());
        assert_eq!(ep.seed, Some(42));
    }

    #[test]
    fn step_n_advances_episode() {
        let mut app = full_test_app();
        reset_episode(&mut app, None);
        step_n(&mut app, 5);

        let (count, _, state) = episode_snapshot(&app);
        assert_eq!(count, 5);
        assert_eq!(state, EpisodeState::Running);
    }

    #[test]
    fn set_step_reward_accumulates() {
        let mut app = full_test_app();
        reset_episode(&mut app, None);

        set_step_reward(&mut app, 3.0);
        app.update();
        set_step_reward(&mut app, 7.0);
        app.update();

        let (_, reward, _) = episode_snapshot(&app);
        assert!((reward - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn run_until_done_respects_truncation() {
        let mut app = full_test_app();

        app.world_mut()
            .resource_mut::<clankers_env::episode::EpisodeConfig>()
            .max_episode_steps = 3;
        reset_episode(&mut app, None);

        let steps = run_until_done(&mut app, 100);
        assert_eq!(steps, 3);
        assert!(app.world().resource::<Episode>().is_done());
    }

    #[test]
    fn run_until_done_returns_max_if_never_done() {
        let mut app = full_test_app();
        // max_episode_steps default is 1000, so 10 steps won't trigger truncation
        reset_episode(&mut app, None);
        let steps = run_until_done(&mut app, 10);
        assert_eq!(steps, 10);
    }

    #[test]
    fn episode_snapshot_reflects_state() {
        let app = full_test_app();
        let (count, reward, state) = episode_snapshot(&app);
        assert_eq!(count, 0);
        assert!(reward.abs() < f32::EPSILON);
        assert_eq!(state, EpisodeState::Idle);
    }
}
