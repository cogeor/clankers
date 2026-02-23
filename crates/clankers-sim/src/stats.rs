//! Episode statistics tracking.
//!
//! [`EpisodeStats`] records cumulative statistics across episodes:
//! total episodes, total steps, and per-episode step-count history.

use bevy::prelude::*;
use clankers_env::episode::{Episode, EpisodeState};

// ---------------------------------------------------------------------------
// EpisodeStats
// ---------------------------------------------------------------------------

/// Bevy resource that tracks cumulative statistics across episodes.
#[derive(Resource, Clone, Debug)]
pub struct EpisodeStats {
    /// Total number of completed episodes.
    pub episodes_completed: u32,
    /// Total steps across all episodes.
    pub total_steps: u64,
    /// Step-count history (steps per completed episode).
    pub step_history: Vec<u32>,
    /// Whether we saw the episode running last frame (for edge detection).
    was_running: bool,
}

impl Default for EpisodeStats {
    fn default() -> Self {
        Self::new()
    }
}

impl EpisodeStats {
    /// Create empty stats.
    pub const fn new() -> Self {
        Self {
            episodes_completed: 0,
            total_steps: 0,
            step_history: Vec::new(),
            was_running: false,
        }
    }

    /// Average episode length (steps) across all completed episodes.
    pub fn mean_episode_length(&self) -> Option<f32> {
        if self.step_history.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let sum: f32 = self.step_history.iter().map(|&s| s as f32).sum();
        #[allow(clippy::cast_precision_loss)]
        Some(sum / self.step_history.len() as f32)
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// System that records episode stats when an episode transitions to a
/// terminal state (Done or Truncated).
#[allow(clippy::needless_pass_by_value)]
pub fn episode_stats_system(episode: Res<Episode>, mut stats: ResMut<EpisodeStats>) {
    let is_done = episode.state.is_terminal();
    let just_finished = is_done && stats.was_running;

    if just_finished {
        stats.episodes_completed += 1;
        stats.total_steps += u64::from(episode.step_count);
        stats.step_history.push(episode.step_count);
    }

    stats.was_running = episode.state == EpisodeState::Running;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_default_empty() {
        let stats = EpisodeStats::new();
        assert_eq!(stats.episodes_completed, 0);
        assert_eq!(stats.total_steps, 0);
        assert!(stats.step_history.is_empty());
        assert!(stats.mean_episode_length().is_none());
    }

    #[test]
    fn mean_episode_length_computes() {
        let mut stats = EpisodeStats::new();
        stats.step_history = vec![100, 200, 300];
        assert!((stats.mean_episode_length().unwrap() - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reset_clears_stats() {
        let mut stats = EpisodeStats::new();
        stats.episodes_completed = 5;
        stats.total_steps = 500;
        stats.step_history = vec![10, 20];
        stats.reset();
        assert_eq!(stats.episodes_completed, 0);
        assert!(stats.step_history.is_empty());
    }

    #[test]
    fn stats_system_records_on_episode_end() {
        use clankers_test_utils::{full_test_app, reset_episode, step_n};

        let mut app = full_test_app();
        app.init_resource::<EpisodeStats>();
        app.add_systems(
            Update,
            episode_stats_system.in_set(clankers_core::ClankersSet::Communicate),
        );
        app.world_mut()
            .resource_mut::<clankers_env::episode::EpisodeConfig>()
            .max_episode_steps = 3;

        reset_episode(&mut app, None);
        step_n(&mut app, 3);

        let stats = app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 1);
        assert_eq!(stats.total_steps, 3);
        assert_eq!(stats.step_history, vec![3]);
    }

    #[test]
    fn stats_system_does_not_double_count() {
        use clankers_test_utils::{full_test_app, reset_episode, step_n};

        let mut app = full_test_app();
        app.init_resource::<EpisodeStats>();
        app.add_systems(
            Update,
            episode_stats_system.in_set(clankers_core::ClankersSet::Communicate),
        );
        app.world_mut()
            .resource_mut::<clankers_env::episode::EpisodeConfig>()
            .max_episode_steps = 2;

        reset_episode(&mut app, None);
        step_n(&mut app, 2);

        // Extra updates while episode is done — should not count again
        app.update();
        app.update();

        let stats = app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 1);
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn stats_is_send_sync() {
        assert_send_sync::<EpisodeStats>();
    }
}
