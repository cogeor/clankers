//! Episode state machine and lifecycle management.
//!
//! An episode is a single rollout from reset to termination/truncation.
//! The [`Episode`] resource tracks state, step count, and accumulated reward.

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// EpisodeState
// ---------------------------------------------------------------------------

/// Lifecycle state of an episode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EpisodeState {
    /// Before the first reset.
    #[default]
    Idle,
    /// Actively stepping.
    Running,
    /// Ended due to task success or failure.
    Done,
    /// Ended due to time limit.
    Truncated,
}

impl EpisodeState {
    /// Returns `true` if the episode is finished (Done or Truncated).
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Truncated)
    }

    /// Returns `true` if the episode is active.
    pub const fn is_running(self) -> bool {
        matches!(self, Self::Running)
    }
}

// ---------------------------------------------------------------------------
// EpisodeConfig
// ---------------------------------------------------------------------------

/// Configuration for episode lifecycle.
#[derive(Resource, Clone, Debug)]
pub struct EpisodeConfig {
    /// Maximum steps before forced truncation.  `0` means no limit.
    pub max_episode_steps: u32,
    /// Whether to automatically reset when the episode ends.
    pub auto_reset: bool,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            max_episode_steps: 1000,
            auto_reset: false,
        }
    }
}

impl EpisodeConfig {
    /// Builder: set max episode steps.
    pub const fn with_max_steps(mut self, steps: u32) -> Self {
        self.max_episode_steps = steps;
        self
    }

    /// Builder: enable auto-reset.
    pub const fn with_auto_reset(mut self, auto_reset: bool) -> Self {
        self.auto_reset = auto_reset;
        self
    }
}

// ---------------------------------------------------------------------------
// Episode
// ---------------------------------------------------------------------------

/// Bevy resource tracking the current episode's state.
#[derive(Resource, Clone, Debug)]
pub struct Episode {
    /// Current lifecycle state.
    pub state: EpisodeState,
    /// Number of steps taken this episode.
    pub step_count: u32,
    /// Total accumulated reward this episode.
    pub total_reward: f32,
    /// Seed used for this episode (set on reset).
    pub seed: Option<u64>,
    /// Number of completed episodes since app start.
    pub episode_number: u32,
}

impl Default for Episode {
    fn default() -> Self {
        Self {
            state: EpisodeState::Idle,
            step_count: 0,
            total_reward: 0.0,
            seed: None,
            episode_number: 0,
        }
    }
}

impl Episode {
    /// Reset the episode to `Running` with an optional seed.
    pub const fn reset(&mut self, seed: Option<u64>) {
        self.state = EpisodeState::Running;
        self.step_count = 0;
        self.total_reward = 0.0;
        self.seed = seed;
        self.episode_number += 1;
    }

    /// Advance one step, accumulating reward.  Returns `false` if the
    /// episode is not running.
    pub fn advance(&mut self, reward: f32) -> bool {
        if self.state != EpisodeState::Running {
            return false;
        }
        self.step_count += 1;
        self.total_reward += reward;
        true
    }

    /// Mark the episode as done (task success/failure).
    pub const fn terminate(&mut self) {
        self.state = EpisodeState::Done;
    }

    /// Mark the episode as truncated (time limit).
    pub const fn truncate(&mut self) {
        self.state = EpisodeState::Truncated;
    }

    /// Check if the episode should be truncated based on max steps.
    /// Returns `true` and sets the state if the limit is reached.
    pub fn check_truncation(&mut self, max_steps: u32) -> bool {
        if max_steps > 0 && self.step_count >= max_steps && self.state == EpisodeState::Running {
            self.state = EpisodeState::Truncated;
            return true;
        }
        false
    }

    /// Whether the episode is in a terminal state.
    pub const fn is_done(&self) -> bool {
        self.state.is_terminal()
    }

    /// Whether the episode is actively running.
    pub const fn is_running(&self) -> bool {
        self.state.is_running()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- EpisodeState --

    #[test]
    fn state_default_is_idle() {
        assert_eq!(EpisodeState::default(), EpisodeState::Idle);
    }

    #[test]
    fn state_terminal_detection() {
        assert!(!EpisodeState::Idle.is_terminal());
        assert!(!EpisodeState::Running.is_terminal());
        assert!(EpisodeState::Done.is_terminal());
        assert!(EpisodeState::Truncated.is_terminal());
    }

    #[test]
    fn state_running_detection() {
        assert!(!EpisodeState::Idle.is_running());
        assert!(EpisodeState::Running.is_running());
        assert!(!EpisodeState::Done.is_running());
        assert!(!EpisodeState::Truncated.is_running());
    }

    // -- EpisodeConfig --

    #[test]
    fn config_default() {
        let c = EpisodeConfig::default();
        assert_eq!(c.max_episode_steps, 1000);
        assert!(!c.auto_reset);
    }

    #[test]
    fn config_builder() {
        let c = EpisodeConfig::default()
            .with_max_steps(500)
            .with_auto_reset(true);
        assert_eq!(c.max_episode_steps, 500);
        assert!(c.auto_reset);
    }

    // -- Episode --

    #[test]
    fn episode_default_is_idle() {
        let ep = Episode::default();
        assert_eq!(ep.state, EpisodeState::Idle);
        assert_eq!(ep.step_count, 0);
        assert!(ep.total_reward.abs() < f32::EPSILON);
        assert!(ep.seed.is_none());
        assert_eq!(ep.episode_number, 0);
    }

    #[test]
    fn episode_reset_transitions_to_running() {
        let mut ep = Episode::default();
        ep.reset(Some(42));
        assert_eq!(ep.state, EpisodeState::Running);
        assert_eq!(ep.step_count, 0);
        assert_eq!(ep.seed, Some(42));
        assert_eq!(ep.episode_number, 1);
    }

    #[test]
    fn episode_advance_accumulates() {
        let mut ep = Episode::default();
        ep.reset(None);
        assert!(ep.advance(1.5));
        assert!(ep.advance(2.0));
        assert_eq!(ep.step_count, 2);
        assert!((ep.total_reward - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn episode_advance_fails_when_not_running() {
        let mut ep = Episode::default();
        assert!(!ep.advance(1.0)); // Idle
        ep.reset(None);
        ep.terminate();
        assert!(!ep.advance(1.0)); // Done
    }

    #[test]
    fn episode_terminate() {
        let mut ep = Episode::default();
        ep.reset(None);
        ep.terminate();
        assert_eq!(ep.state, EpisodeState::Done);
        assert!(ep.is_done());
        assert!(!ep.is_running());
    }

    #[test]
    fn episode_truncate() {
        let mut ep = Episode::default();
        ep.reset(None);
        ep.truncate();
        assert_eq!(ep.state, EpisodeState::Truncated);
        assert!(ep.is_done());
    }

    #[test]
    fn episode_check_truncation() {
        let mut ep = Episode::default();
        ep.reset(None);
        for _ in 0..5 {
            ep.advance(1.0);
        }
        assert!(!ep.check_truncation(10)); // not yet
        for _ in 0..5 {
            ep.advance(1.0);
        }
        assert!(ep.check_truncation(10)); // now at 10
        assert_eq!(ep.state, EpisodeState::Truncated);
    }

    #[test]
    fn episode_check_truncation_zero_means_no_limit() {
        let mut ep = Episode::default();
        ep.reset(None);
        for _ in 0..1000 {
            ep.advance(1.0);
        }
        assert!(!ep.check_truncation(0));
        assert!(ep.is_running());
    }

    #[test]
    fn episode_reset_increments_episode_number() {
        let mut ep = Episode::default();
        ep.reset(None);
        assert_eq!(ep.episode_number, 1);
        ep.reset(None);
        assert_eq!(ep.episode_number, 2);
        ep.reset(Some(99));
        assert_eq!(ep.episode_number, 3);
    }

    #[test]
    fn episode_reset_clears_reward() {
        let mut ep = Episode::default();
        ep.reset(None);
        ep.advance(100.0);
        ep.reset(None);
        assert!(ep.total_reward.abs() < f32::EPSILON);
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn types_are_send_sync() {
        assert_send_sync::<EpisodeState>();
        assert_send_sync::<EpisodeConfig>();
        assert_send_sync::<Episode>();
    }
}
