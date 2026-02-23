//! Per-environment episode tracking for `VecEnv`.
//!
//! When running multiple parallel environments, each needs its own episode
//! state. [`EnvEpisodeMap`] maps [`EnvId`] to an independent [`Episode`],
//! and [`AutoResetMode`] controls what happens when an episode terminates.

use std::collections::HashMap;

use crate::episode::{Episode, EpisodeState};
use clankers_core::types::EnvId;

// ---------------------------------------------------------------------------
// AutoResetMode
// ---------------------------------------------------------------------------

/// Controls automatic reset behavior when an episode terminates.
///
/// Used by `VecEnv` to determine how to handle environments that reach a
/// terminal state during a batched step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AutoResetMode {
    /// No automatic reset. The caller must explicitly reset terminated envs.
    #[default]
    Disabled,
    /// Reset immediately in the same step, returning the *new* episode's
    /// initial observation.
    Immediate,
    /// Mark the environment for reset on the *next* step. The current step
    /// returns the terminal observation.
    NextStep,
}

// ---------------------------------------------------------------------------
// EnvEpisodeMap
// ---------------------------------------------------------------------------

/// Per-environment episode state map for `VecEnv`.
///
/// Tracks an independent [`Episode`] for each [`EnvId`]. Provides bulk
/// operations like resetting all environments or querying which envs are done.
///
/// # Example
///
/// ```
/// use clankers_core::types::EnvId;
/// use clankers_env::vec_episode::EnvEpisodeMap;
///
/// let mut map = EnvEpisodeMap::new(4);
/// assert_eq!(map.len(), 4);
/// map.reset(EnvId(0), None);
/// assert!(map.get(EnvId(0)).is_running());
/// ```
#[derive(Debug, Clone)]
pub struct EnvEpisodeMap {
    episodes: HashMap<EnvId, Episode>,
}

impl EnvEpisodeMap {
    /// Create a map with `num_envs` environments, all starting in `Idle`.
    #[must_use]
    pub fn new(num_envs: u16) -> Self {
        let mut episodes = HashMap::with_capacity(usize::from(num_envs));
        for i in 0..num_envs {
            episodes.insert(EnvId(i), Episode::default());
        }
        Self { episodes }
    }

    /// Number of environments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Whether the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Get the episode state for an environment.
    ///
    /// # Panics
    ///
    /// Panics if `env_id` is not in the map.
    #[must_use]
    pub fn get(&self, env_id: EnvId) -> &Episode {
        &self.episodes[&env_id]
    }

    /// Get a mutable reference to the episode for an environment.
    ///
    /// # Panics
    ///
    /// Panics if `env_id` is not in the map.
    pub fn get_mut(&mut self, env_id: EnvId) -> &mut Episode {
        self.episodes.get_mut(&env_id).expect("unknown env_id")
    }

    /// Reset a specific environment's episode.
    pub fn reset(&mut self, env_id: EnvId, seed: Option<u64>) {
        self.get_mut(env_id).reset(seed);
    }

    /// Reset all environments.
    pub fn reset_all(&mut self, seed: Option<u64>) {
        for ep in self.episodes.values_mut() {
            ep.reset(seed);
        }
    }

    /// Collect the `EnvId`s of all environments that are in a terminal state.
    #[must_use]
    pub fn done_envs(&self) -> Vec<EnvId> {
        self.episodes
            .iter()
            .filter(|(_, ep)| ep.state.is_terminal())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Collect the `EnvId`s of all environments that are running.
    #[must_use]
    pub fn running_envs(&self) -> Vec<EnvId> {
        self.episodes
            .iter()
            .filter(|(_, ep)| ep.state == EpisodeState::Running)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Iterator over all `(EnvId, &Episode)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&EnvId, &Episode)> {
        self.episodes.iter()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_idle_episodes() {
        let map = EnvEpisodeMap::new(3);
        assert_eq!(map.len(), 3);
        for i in 0..3 {
            assert_eq!(map.get(EnvId(i)).state, EpisodeState::Idle);
        }
    }

    #[test]
    fn reset_single() {
        let mut map = EnvEpisodeMap::new(2);
        map.reset(EnvId(0), Some(42));
        assert!(map.get(EnvId(0)).is_running());
        assert!(!map.get(EnvId(1)).is_running());
        assert_eq!(map.get(EnvId(0)).seed, Some(42));
    }

    #[test]
    fn reset_all_starts_running() {
        let mut map = EnvEpisodeMap::new(4);
        map.reset_all(None);
        for i in 0..4 {
            assert!(map.get(EnvId(i)).is_running());
        }
    }

    #[test]
    fn done_envs_collects_terminal() {
        let mut map = EnvEpisodeMap::new(3);
        map.reset_all(None);
        map.get_mut(EnvId(1)).terminate();
        map.get_mut(EnvId(2)).truncate();

        let done = map.done_envs();
        assert_eq!(done.len(), 2);
        assert!(done.contains(&EnvId(1)));
        assert!(done.contains(&EnvId(2)));
    }

    #[test]
    fn running_envs_filters_active() {
        let mut map = EnvEpisodeMap::new(3);
        map.reset_all(None);
        map.get_mut(EnvId(0)).terminate();

        let running = map.running_envs();
        assert_eq!(running.len(), 2);
        assert!(!running.contains(&EnvId(0)));
    }

    #[test]
    fn advance_increments_step_count() {
        let mut map = EnvEpisodeMap::new(1);
        map.reset(EnvId(0), None);
        map.get_mut(EnvId(0)).advance();
        map.get_mut(EnvId(0)).advance();
        assert_eq!(map.get(EnvId(0)).step_count, 2);
    }

    #[test]
    fn auto_reset_mode_default() {
        assert_eq!(AutoResetMode::default(), AutoResetMode::Disabled);
    }

    #[test]
    fn empty_map() {
        let map = EnvEpisodeMap::new(0);
        assert!(map.is_empty());
        assert_eq!(map.done_envs().len(), 0);
    }

    #[test]
    fn iter_all_episodes() {
        let map = EnvEpisodeMap::new(3);
        assert_eq!(map.iter().count(), 3);
    }
}
