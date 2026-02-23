//! `VecEnv` configuration for multi-environment simulation.
//!
//! Defines [`VecEnvConfig`] which controls the number of parallel environments,
//! auto-reset behavior, and step limits. Used by `VecEnvRunner` to drive
//! multiple environment instances.

use crate::vec_episode::AutoResetMode;

// ---------------------------------------------------------------------------
// VecEnvConfig
// ---------------------------------------------------------------------------

/// Configuration for a vectorized (parallel) environment.
///
/// # Example
///
/// ```
/// use clankers_env::vec_env::VecEnvConfig;
/// use clankers_env::vec_episode::AutoResetMode;
///
/// let cfg = VecEnvConfig::new(8)
///     .with_auto_reset(AutoResetMode::Immediate)
///     .with_max_steps(500);
/// assert_eq!(cfg.num_envs(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct VecEnvConfig {
    num_envs: u16,
    auto_reset_mode: AutoResetMode,
    max_episode_steps: u32,
}

impl VecEnvConfig {
    /// Create a config for `num_envs` parallel environments.
    #[must_use]
    pub const fn new(num_envs: u16) -> Self {
        Self {
            num_envs,
            auto_reset_mode: AutoResetMode::Disabled,
            max_episode_steps: 1000,
        }
    }

    /// Set the auto-reset mode.
    #[must_use]
    pub const fn with_auto_reset(mut self, mode: AutoResetMode) -> Self {
        self.auto_reset_mode = mode;
        self
    }

    /// Set max steps per episode (0 = no limit).
    #[must_use]
    pub const fn with_max_steps(mut self, steps: u32) -> Self {
        self.max_episode_steps = steps;
        self
    }

    /// Number of parallel environments.
    #[must_use]
    pub const fn num_envs(&self) -> u16 {
        self.num_envs
    }

    /// Auto-reset mode.
    #[must_use]
    pub const fn auto_reset_mode(&self) -> AutoResetMode {
        self.auto_reset_mode
    }

    /// Max steps per episode.
    #[must_use]
    pub const fn max_episode_steps(&self) -> u32 {
        self.max_episode_steps
    }
}

impl Default for VecEnvConfig {
    fn default() -> Self {
        Self::new(1)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = VecEnvConfig::default();
        assert_eq!(cfg.num_envs(), 1);
        assert_eq!(cfg.auto_reset_mode(), AutoResetMode::Disabled);
        assert_eq!(cfg.max_episode_steps(), 1000);
    }

    #[test]
    fn builder_pattern() {
        let cfg = VecEnvConfig::new(16)
            .with_auto_reset(AutoResetMode::Immediate)
            .with_max_steps(200);
        assert_eq!(cfg.num_envs(), 16);
        assert_eq!(cfg.auto_reset_mode(), AutoResetMode::Immediate);
        assert_eq!(cfg.max_episode_steps(), 200);
    }
}
