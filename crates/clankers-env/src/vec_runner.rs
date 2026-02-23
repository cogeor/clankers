//! `VecEnvRunner`: sequential multi-environment step/reset driver.
//!
//! Manages a vector of [`GymEnv`]-like closures (environment factories),
//! stepping and resetting them independently. Observations, rewards, and
//! done flags are stored in [`SoA`](super::vec_buffer) buffers for
//! efficient batched access.

use clankers_core::types::{Action, EnvId, Observation, ResetResult, StepResult};

use crate::vec_buffer::{VecDoneBuffer, VecObsBuffer, VecRewardBuffer};
use crate::vec_env::VecEnvConfig;
use crate::vec_episode::{AutoResetMode, EnvEpisodeMap};

/// Trait abstracting a single environment for `VecEnvRunner`.
///
/// Any type providing `step`, `reset`, and `obs_dim` can be used
/// as an environment in the vectorized runner.
pub trait VecEnvInstance {
    /// Reset the environment.
    fn reset(&mut self, seed: Option<u64>) -> ResetResult;
    /// Step the environment with an action.
    fn step(&mut self, action: &Action) -> StepResult;
    /// Observation dimension.
    fn obs_dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// VecEnvRunner
// ---------------------------------------------------------------------------

/// Sequential multi-environment runner.
///
/// Owns N environment instances and drives them in lockstep. Stores results
/// in batched `SoA` buffers for efficient training access.
///
/// # Example
///
/// ```
/// use clankers_env::vec_runner::{VecEnvInstance, VecEnvRunner};
/// use clankers_env::vec_env::VecEnvConfig;
/// use clankers_core::types::*;
///
/// struct DummyEnv;
/// impl VecEnvInstance for DummyEnv {
///     fn reset(&mut self, _seed: Option<u64>) -> ResetResult {
///         ResetResult { observation: Observation::zeros(2), info: ResetInfo::default() }
///     }
///     fn step(&mut self, _action: &Action) -> StepResult {
///         StepResult {
///             observation: Observation::zeros(2), reward: 1.0,
///             terminated: false, truncated: false, info: StepInfo::default(),
///         }
///     }
///     fn obs_dim(&self) -> usize { 2 }
/// }
///
/// let envs: Vec<Box<dyn VecEnvInstance>> = vec![Box::new(DummyEnv), Box::new(DummyEnv)];
/// let config = VecEnvConfig::new(2);
/// let mut runner = VecEnvRunner::new(envs, config);
/// runner.reset_all(None);
/// let _obs = runner.obs_buffer();
/// ```
pub struct VecEnvRunner {
    envs: Vec<Box<dyn VecEnvInstance>>,
    config: VecEnvConfig,
    episodes: EnvEpisodeMap,
    obs_buf: VecObsBuffer,
    reward_buf: VecRewardBuffer,
    done_buf: VecDoneBuffer,
}

impl VecEnvRunner {
    /// Create a runner from a vec of environment instances and config.
    ///
    /// # Panics
    ///
    /// Panics if `envs.len() != config.num_envs()` or if environments
    /// have different observation dimensions.
    #[must_use]
    pub fn new(envs: Vec<Box<dyn VecEnvInstance>>, config: VecEnvConfig) -> Self {
        assert_eq!(
            envs.len(),
            usize::from(config.num_envs()),
            "env count mismatch"
        );
        assert!(!envs.is_empty(), "need at least one environment");

        let obs_dim = envs[0].obs_dim();
        for (i, env) in envs.iter().enumerate() {
            assert_eq!(
                env.obs_dim(),
                obs_dim,
                "obs_dim mismatch for env {i}: expected {obs_dim}, got {}",
                env.obs_dim()
            );
        }

        let n = envs.len();
        let num_envs = config.num_envs();
        Self {
            envs,
            config,
            episodes: EnvEpisodeMap::new(num_envs),
            obs_buf: VecObsBuffer::new(n, obs_dim),
            reward_buf: VecRewardBuffer::new(n),
            done_buf: VecDoneBuffer::new(n),
        }
    }

    /// Number of environments.
    #[must_use]
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Observation buffer (read-only).
    #[must_use]
    pub const fn obs_buffer(&self) -> &VecObsBuffer {
        &self.obs_buf
    }

    /// Reward buffer (read-only).
    #[must_use]
    pub const fn reward_buffer(&self) -> &VecRewardBuffer {
        &self.reward_buf
    }

    /// Done buffer (read-only).
    #[must_use]
    pub const fn done_buffer(&self) -> &VecDoneBuffer {
        &self.done_buf
    }

    /// Episode map (read-only).
    #[must_use]
    pub const fn episodes(&self) -> &EnvEpisodeMap {
        &self.episodes
    }

    /// Reset all environments.
    pub fn reset_all(&mut self, seed: Option<u64>) {
        for (i, env) in self.envs.iter_mut().enumerate() {
            let result = env.reset(seed);
            self.obs_buf.set(i, &result.observation);
            self.episodes.reset(EnvId(u16::try_from(i).expect("env index overflow")), seed);
        }
        self.reward_buf.clear();
        self.done_buf.clear();
    }

    /// Reset specific environments by index.
    pub fn reset_envs(&mut self, env_ids: &[EnvId], seed: Option<u64>) {
        for &env_id in env_ids {
            let idx = usize::from(env_id.index());
            let result = self.envs[idx].reset(seed);
            self.obs_buf.set(idx, &result.observation);
            self.episodes.reset(env_id, seed);
            self.reward_buf.set(idx, 0.0);
            self.done_buf.set(idx, false, false);
        }
    }

    /// Step all environments with the given actions.
    ///
    /// `actions` must have length equal to `num_envs`. After stepping,
    /// environments in a terminal state are handled according to the
    /// configured [`AutoResetMode`].
    ///
    /// # Panics
    ///
    /// Panics if `actions.len() != num_envs`.
    pub fn step_all(&mut self, actions: &[Action]) {
        assert_eq!(actions.len(), self.envs.len(), "action count mismatch");

        for (i, (env, action)) in self.envs.iter_mut().zip(actions.iter()).enumerate() {
            let result = env.step(action);
            self.obs_buf.set(i, &result.observation);
            self.reward_buf.set(i, result.reward);
            self.done_buf.set(i, result.terminated, result.truncated);

            let env_id = EnvId(u16::try_from(i).expect("env index overflow"));
            let ep = self.episodes.get_mut(env_id);
            ep.advance(result.reward);

            if result.terminated {
                ep.terminate();
            } else if result.truncated {
                ep.truncate();
            }

            // Check truncation based on max steps
            if !result.terminated
                && !result.truncated
                && ep.check_truncation(self.config.max_episode_steps())
            {
                self.done_buf.set(i, false, true);
            }
        }

        // Auto-reset handling
        if self.config.auto_reset_mode() == AutoResetMode::Immediate {
            let done_envs = self.episodes.done_envs();
            for env_id in done_envs {
                let idx = usize::from(env_id.index());
                let result = self.envs[idx].reset(None);
                self.obs_buf.set(idx, &result.observation);
                self.episodes.reset(env_id, None);
            }
        }
    }

    /// Get observation for a specific env.
    #[must_use]
    pub fn get_obs(&self, env_idx: usize) -> Observation {
        self.obs_buf.get(env_idx)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clankers_core::types::{ResetInfo, StepInfo};

    struct ConstEnv {
        obs_dim: usize,
        step_count: u32,
        terminated_at: Option<u32>,
    }

    impl ConstEnv {
        fn new(obs_dim: usize) -> Self {
            Self {
                obs_dim,
                step_count: 0,
                terminated_at: None,
            }
        }

        fn terminating_at(obs_dim: usize, step: u32) -> Self {
            Self {
                obs_dim,
                step_count: 0,
                terminated_at: Some(step),
            }
        }
    }

    impl VecEnvInstance for ConstEnv {
        fn reset(&mut self, _seed: Option<u64>) -> ResetResult {
            self.step_count = 0;
            ResetResult {
                observation: Observation::zeros(self.obs_dim),
                info: ResetInfo::default(),
            }
        }

        fn step(&mut self, _action: &Action) -> StepResult {
            self.step_count += 1;
            let terminated = self.terminated_at.is_some_and(|t| self.step_count >= t);
            StepResult {
                #[allow(clippy::cast_precision_loss)]
                observation: Observation::new(vec![self.step_count as f32; self.obs_dim]),
                reward: 1.0,
                terminated,
                truncated: false,
                info: StepInfo::default(),
            }
        }

        fn obs_dim(&self) -> usize {
            self.obs_dim
        }
    }

    fn make_envs(n: usize, obs_dim: usize) -> Vec<Box<dyn VecEnvInstance>> {
        (0..n)
            .map(|_| Box::new(ConstEnv::new(obs_dim)) as Box<dyn VecEnvInstance>)
            .collect()
    }

    #[test]
    fn reset_all_populates_buffers() {
        let envs = make_envs(3, 2);
        let config = VecEnvConfig::new(3);
        let mut runner = VecEnvRunner::new(envs, config);
        runner.reset_all(None);

        assert_eq!(runner.num_envs(), 3);
        for i in 0..3 {
            assert_eq!(runner.get_obs(i).len(), 2);
        }
    }

    #[test]
    fn step_all_updates_buffers() {
        let envs = make_envs(2, 2);
        let config = VecEnvConfig::new(2);
        let mut runner = VecEnvRunner::new(envs, config);
        runner.reset_all(None);

        let actions = vec![Action::zeros(2), Action::zeros(2)];
        runner.step_all(&actions);

        // After one step, obs should be [1.0, 1.0] (step_count=1)
        assert_eq!(runner.get_obs(0).as_slice(), &[1.0, 1.0]);
        assert!((runner.reward_buffer().get(0) - 1.0).abs() < f32::EPSILON);
        assert!(!runner.done_buffer().is_done(0));
    }

    #[test]
    fn step_detects_termination() {
        let envs: Vec<Box<dyn VecEnvInstance>> = vec![
            Box::new(ConstEnv::terminating_at(2, 2)),
            Box::new(ConstEnv::new(2)),
        ];
        let config = VecEnvConfig::new(2);
        let mut runner = VecEnvRunner::new(envs, config);
        runner.reset_all(None);

        let actions = vec![Action::zeros(2), Action::zeros(2)];
        runner.step_all(&actions);
        assert!(!runner.done_buffer().terminated(0));

        runner.step_all(&actions);
        assert!(runner.done_buffer().terminated(0));
        assert!(!runner.done_buffer().terminated(1));
    }

    #[test]
    fn auto_reset_immediate() {
        let envs: Vec<Box<dyn VecEnvInstance>> = vec![
            Box::new(ConstEnv::terminating_at(2, 1)),
            Box::new(ConstEnv::new(2)),
        ];
        let config = VecEnvConfig::new(2).with_auto_reset(AutoResetMode::Immediate);
        let mut runner = VecEnvRunner::new(envs, config);
        runner.reset_all(None);

        let actions = vec![Action::zeros(2), Action::zeros(2)];
        runner.step_all(&actions);

        // Env 0 terminated at step 1, then immediately reset
        // So observation should be zeros (from reset), not [1.0, 1.0]
        assert_eq!(runner.get_obs(0).as_slice(), &[0.0, 0.0]);
        assert!(runner.episodes().get(EnvId(0)).is_running());
    }

    #[test]
    fn reset_specific_envs() {
        let envs = make_envs(3, 2);
        let config = VecEnvConfig::new(3);
        let mut runner = VecEnvRunner::new(envs, config);
        runner.reset_all(None);

        let actions = vec![Action::zeros(2), Action::zeros(2), Action::zeros(2)];
        runner.step_all(&actions);
        // All envs have stepped once, obs = [1.0, 1.0]

        runner.reset_envs(&[EnvId(1)], Some(99));
        // Env 1 reset, obs = [0.0, 0.0]; others unchanged
        assert_eq!(runner.get_obs(0).as_slice(), &[1.0, 1.0]);
        assert_eq!(runner.get_obs(1).as_slice(), &[0.0, 0.0]);
        assert_eq!(runner.get_obs(2).as_slice(), &[1.0, 1.0]);
    }

    #[test]
    fn max_steps_truncation() {
        let envs = make_envs(1, 2);
        let config = VecEnvConfig::new(1).with_max_steps(3);
        let mut runner = VecEnvRunner::new(envs, config);
        runner.reset_all(None);

        let actions = vec![Action::zeros(2)];
        runner.step_all(&actions);
        assert!(!runner.done_buffer().truncated(0));

        runner.step_all(&actions);
        assert!(!runner.done_buffer().truncated(0));

        runner.step_all(&actions);
        assert!(runner.done_buffer().truncated(0));
    }

    #[test]
    fn episodes_tracked_per_env() {
        let envs = make_envs(2, 2);
        let config = VecEnvConfig::new(2);
        let mut runner = VecEnvRunner::new(envs, config);
        runner.reset_all(Some(42));

        assert!(runner.episodes().get(EnvId(0)).is_running());
        assert!(runner.episodes().get(EnvId(1)).is_running());
        assert_eq!(runner.episodes().get(EnvId(0)).seed, Some(42));
    }
}
