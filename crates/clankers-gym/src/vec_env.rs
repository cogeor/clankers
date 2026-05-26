//! Vectorized environment wrapper for batched training over TCP.
//!
//! [`GymVecEnv`] wraps a [`VecEnvRunner`] and exposes batch `step`/`reset`
//! methods returning [`BatchStepResult`] and [`BatchResetResult`]. This is
//! the server-side counterpart of a Python `VecEnv` client.

use clankers_core::types::{
    Action, ActionSpace, BatchResetResult, BatchStepResult, EnvId, ObservationSpace, ResetInfo,
};
use clankers_env::vec_env::VecEnvConfig;
use clankers_env::vec_runner::{VecEnvInstance, VecEnvRunner, VecRunnerLike, runner_for};

// Re-export so downstream callers can name the parallel runner without
// reaching into `clankers-env` directly.
pub use clankers_env::parallel_runner::ParallelVecEnvRunner;

use crate::protocol::EnvInfo;

// ---------------------------------------------------------------------------
// GymVecEnv
// ---------------------------------------------------------------------------

/// Vectorized environment for batched training.
///
/// Owns a [`VecEnvRunner`] and provides batch step/reset methods that return
/// protocol-ready [`BatchStepResult`] and [`BatchResetResult`] types.
///
/// # Example
///
/// ```
/// use clankers_gym::vec_env::GymVecEnv;
/// use clankers_env::vec_env::VecEnvConfig;
/// use clankers_env::vec_runner::VecEnvInstance;
/// use clankers_core::types::*;
///
/// struct DummyEnv;
/// impl VecEnvInstance for DummyEnv {
///     fn reset(&mut self, _seed: Option<u64>) -> ResetResult {
///         ResetResult { observation: Observation::zeros(2), info: ResetInfo::default() }
///     }
///     fn step(&mut self, _action: &Action) -> StepResult {
///         StepResult {
///             observation: Observation::zeros(2), reward: 0.0,
///             terminated: false, truncated: false, info: StepInfo::default(),
///         }
///     }
///     fn obs_dim(&self) -> usize { 2 }
/// }
///
/// let envs: Vec<Box<dyn VecEnvInstance>> =
///     vec![Box::new(DummyEnv), Box::new(DummyEnv)];
/// let config = VecEnvConfig::new(2);
/// let obs_space = ObservationSpace::Box { low: vec![-1.0; 2], high: vec![1.0; 2] };
/// let act_space = ActionSpace::Box { low: vec![-1.0; 2], high: vec![1.0; 2] };
/// let mut vec_env = GymVecEnv::new(envs, config, obs_space, act_space);
/// vec_env.reset_all(None);
/// ```
///
/// # Parallel runner selection
///
/// `GymVecEnv` holds a `Box<dyn VecRunnerLike>` so the concrete
/// runner type is chosen by [`VecEnvConfig::is_parallel`]:
/// - `false` (default) → [`VecEnvRunner`] (sequential, back-compat).
/// - `true` → [`ParallelVecEnvRunner`] (Rayon-backed,
///   deterministically-seeded).
pub struct GymVecEnv {
    runner: Box<dyn VecRunnerLike>,
    obs_space: ObservationSpace,
    act_space: ActionSpace,
}

impl GymVecEnv {
    /// Create a new vectorized environment using the sequential
    /// [`VecEnvRunner`] regardless of [`VecEnvConfig::is_parallel`].
    ///
    /// Back-compat constructor: accepts `Box<dyn VecEnvInstance>`
    /// (without the `Send` bound). [`GymEnv`](crate::env::GymEnv)
    /// wraps a Bevy `App` whose `runner` field holds a
    /// `Box<dyn FnOnce(App) -> AppExit>` (`!Send`), so the parallel
    /// runner is unreachable from `GymEnv`-backed callers. If
    /// `config.is_parallel()` is `true`, this constructor still picks
    /// the sequential path and surfaces a debug-assertion in tests.
    ///
    /// # Panics
    ///
    /// Panics if `envs.len() != config.num_envs()`.
    #[must_use]
    pub fn new(
        envs: Vec<Box<dyn VecEnvInstance>>,
        config: VecEnvConfig,
        obs_space: ObservationSpace,
        act_space: ActionSpace,
    ) -> Self {
        debug_assert!(
            !config.is_parallel(),
            "GymVecEnv::new always selects the sequential runner; \
             use GymVecEnv::new_with_send_envs to enable parallel"
        );
        let runner: Box<dyn VecRunnerLike> = Box::new(VecEnvRunner::new(envs, config));
        Self {
            runner,
            obs_space,
            act_space,
        }
    }

    /// Create a new vectorized environment that may use the parallel
    /// runner.
    ///
    /// Requires `Box<dyn VecEnvInstance + Send>` env instances so the
    /// parallel path is reachable when [`VecEnvConfig::is_parallel`]
    /// is `true`. Sequential fallback is used otherwise (no extra
    /// cost; the cast widens to the non-`Send` trait object).
    ///
    /// # Panics
    ///
    /// Panics if `envs.len() != config.num_envs()`.
    #[must_use]
    pub fn new_with_send_envs(
        envs: Vec<Box<dyn VecEnvInstance + Send>>,
        config: VecEnvConfig,
        obs_space: ObservationSpace,
        act_space: ActionSpace,
    ) -> Self {
        let runner = runner_for(envs, config);
        Self {
            runner,
            obs_space,
            act_space,
        }
    }

    /// Number of environments.
    #[must_use]
    pub fn num_envs(&self) -> usize {
        self.runner.num_envs()
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

    /// Environment info for protocol handshake.
    #[must_use]
    pub fn env_info(&self) -> EnvInfo {
        EnvInfo {
            n_agents: self.runner.num_envs(),
            observation_space: self.obs_space.clone(),
            action_space: self.act_space.clone(),
        }
    }

    /// Reset all environments.
    pub fn reset_all(&mut self, seed: Option<u64>) -> BatchResetResult {
        self.runner.reset_all(seed);
        self.collect_reset_results()
    }

    /// Reset specific environments by index.
    pub fn reset_envs(
        &mut self,
        env_ids: &[u16],
        seeds: Option<&[Option<u64>]>,
    ) -> BatchResetResult {
        let ids: Vec<EnvId> = env_ids.iter().map(|&id| EnvId(id)).collect();

        if let Some(per_env_seeds) = seeds {
            // Reset each env with its individual seed (one trait call per
            // env so each receives its own seed).
            for (&env_id, seed) in env_ids.iter().zip(per_env_seeds.iter()) {
                self.runner.reset_envs(&[EnvId(env_id)], *seed);
            }
        } else {
            self.runner.reset_envs(&ids, None);
        }

        // Collect results only for the reset envs
        let mut observations = Vec::with_capacity(env_ids.len());
        let mut infos = Vec::with_capacity(env_ids.len());

        for &env_id in env_ids {
            let idx = usize::from(env_id);
            observations.push(self.runner.get_obs(idx));
            infos.push(ResetInfo::default());
        }

        BatchResetResult {
            observations,
            infos,
        }
    }

    /// Step all environments with the given actions.
    ///
    /// # Panics
    ///
    /// Panics if `actions.len() != num_envs()`.
    pub fn step_all(&mut self, actions: &[Action]) -> BatchStepResult {
        self.runner.step_all(actions);
        self.collect_step_results()
    }

    /// Read-only access to the underlying runner via the [`VecRunnerLike`]
    /// trait surface.
    #[must_use]
    pub fn runner(&self) -> &dyn VecRunnerLike {
        &*self.runner
    }

    /// Mutable access to the underlying runner via the [`VecRunnerLike`]
    /// trait surface.
    pub fn runner_mut(&mut self) -> &mut dyn VecRunnerLike {
        &mut *self.runner
    }

    fn collect_reset_results(&self) -> BatchResetResult {
        let n = self.runner.num_envs();
        let mut observations = Vec::with_capacity(n);
        let mut infos = Vec::with_capacity(n);

        for i in 0..n {
            observations.push(self.runner.get_obs(i));
            infos.push(ResetInfo::default());
        }

        BatchResetResult {
            observations,
            infos,
        }
    }

    fn collect_step_results(&self) -> BatchStepResult {
        let n = self.runner.num_envs();
        let obs_buf = self.runner.obs_buffer();
        let done_buf = self.runner.done_buffer();
        let episodes = self.runner.episodes();

        let mut observations = Vec::with_capacity(n);
        let mut terminated = Vec::with_capacity(n);
        let mut truncated = Vec::with_capacity(n);
        let mut infos = Vec::with_capacity(n);

        for i in 0..n {
            observations.push(obs_buf.get(i));
            terminated.push(done_buf.terminated(i));
            truncated.push(done_buf.truncated(i));

            let env_id = EnvId(u16::try_from(i).expect("env index overflow"));
            let ep = episodes.get(env_id);
            infos.push(clankers_core::types::StepInfo {
                episode_length: ep.step_count,
                ..Default::default()
            });
        }

        BatchStepResult {
            observations,
            rewards: vec![0.0; n],
            terminated,
            truncated,
            infos,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clankers_core::types::{Observation, ResetResult, StepInfo, StepResult};

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
                info: clankers_core::types::ResetInfo::default(),
            }
        }

        fn step(&mut self, _action: &Action) -> StepResult {
            self.step_count += 1;
            let terminated = self.terminated_at.is_some_and(|t| self.step_count >= t);
            StepResult {
                #[allow(clippy::cast_precision_loss)]
                observation: Observation::new(vec![self.step_count as f32; self.obs_dim]),
                reward: 0.0,
                terminated,
                truncated: false,
                info: StepInfo::default(),
            }
        }

        fn obs_dim(&self) -> usize {
            self.obs_dim
        }
    }

    fn make_vec_env(n: usize, obs_dim: usize) -> GymVecEnv {
        let envs: Vec<Box<dyn VecEnvInstance>> = (0..n)
            .map(|_| Box::new(ConstEnv::new(obs_dim)) as Box<dyn VecEnvInstance>)
            .collect();
        let config = VecEnvConfig::new(u16::try_from(n).unwrap());
        let obs_space = ObservationSpace::Box {
            low: vec![-10.0; obs_dim],
            high: vec![10.0; obs_dim],
        };
        let act_space = ActionSpace::Box {
            low: vec![-1.0; obs_dim],
            high: vec![1.0; obs_dim],
        };
        GymVecEnv::new(envs, config, obs_space, act_space)
    }

    #[test]
    fn reset_all_returns_batch() {
        let mut env = make_vec_env(3, 2);
        let result = env.reset_all(None);
        assert_eq!(result.num_envs(), 3);
        for i in 0..3 {
            assert_eq!(result.observations[i].len(), 2);
        }
    }

    #[test]
    fn step_all_returns_batch() {
        let mut env = make_vec_env(2, 2);
        env.reset_all(None);

        let actions = vec![Action::zeros(2), Action::zeros(2)];
        let result = env.step_all(&actions);
        assert_eq!(result.num_envs(), 2);
        assert_eq!(result.observations[0].as_slice(), &[1.0, 1.0]);
        assert!(!result.terminated[0]);
    }

    #[test]
    fn reset_envs_selective() {
        let mut env = make_vec_env(3, 2);
        env.reset_all(None);

        let actions = vec![Action::zeros(2), Action::zeros(2), Action::zeros(2)];
        env.step_all(&actions);

        let result = env.reset_envs(&[1], None);
        assert_eq!(result.num_envs(), 1);
        assert_eq!(result.observations[0].as_slice(), &[0.0, 0.0]);

        // Env 0 and 2 should still have step 1 observations
        assert_eq!(env.runner().get_obs(0).as_slice(), &[1.0, 1.0]);
        assert_eq!(env.runner().get_obs(2).as_slice(), &[1.0, 1.0]);
    }

    #[test]
    fn reset_envs_with_seeds() {
        let mut env = make_vec_env(2, 2);
        env.reset_all(None);

        let result = env.reset_envs(&[0, 1], Some(&[Some(42), Some(99)]));
        assert_eq!(result.num_envs(), 2);
    }

    #[test]
    fn step_detects_termination_in_batch() {
        let envs: Vec<Box<dyn VecEnvInstance>> = vec![
            Box::new(ConstEnv::terminating_at(2, 2)),
            Box::new(ConstEnv::new(2)),
        ];
        let config = VecEnvConfig::new(2);
        let obs_space = ObservationSpace::Box {
            low: vec![-10.0; 2],
            high: vec![10.0; 2],
        };
        let act_space = ActionSpace::Box {
            low: vec![-1.0; 2],
            high: vec![1.0; 2],
        };
        let mut env = GymVecEnv::new(envs, config, obs_space, act_space);
        env.reset_all(None);

        let actions = vec![Action::zeros(2), Action::zeros(2)];
        let r1 = env.step_all(&actions);
        assert!(!r1.terminated[0]);

        let r2 = env.step_all(&actions);
        assert!(r2.terminated[0]);
        assert!(!r2.terminated[1]);
    }

    #[test]
    fn env_info_reflects_spaces() {
        let env = make_vec_env(4, 3);
        let info = env.env_info();
        assert_eq!(info.n_agents, 4);
        assert_eq!(info.observation_space.shape(), vec![3]);
        assert_eq!(info.action_space.shape(), vec![3]);
    }

    #[test]
    fn num_envs_matches() {
        let env = make_vec_env(5, 2);
        assert_eq!(env.num_envs(), 5);
    }

    // ------------------------------------------------------------------
    // PR1 compile-time / dispatch checks
    // ------------------------------------------------------------------

    // `GymEnv` wraps a Bevy `App` whose `runner` field holds a
    // `Box<dyn FnOnce(App) -> AppExit + 'static>` (not `Send`), so we
    // cannot pin `GymEnv: Send` directly. Pin the bound on the test
    // fixture instead — the fixture is the type that actually rides
    // the parallel runner in the determinism integration test, so
    // anchoring `Send` here is the load-bearing compile-time check
    // for PR1. See `crates/clankers-env/tests/parallel_determinism.rs`.
    static_assertions::assert_impl_all!(ConstEnv: Send);

    #[test]
    fn parallel_runner_factory_dispatches_on_config() {
        // Sequential default
        let seq_envs: Vec<Box<dyn VecEnvInstance + Send>> = (0..2)
            .map(|_| Box::new(ConstEnv::new(2)) as Box<dyn VecEnvInstance + Send>)
            .collect();
        let obs_space = ObservationSpace::Box {
            low: vec![-1.0; 2],
            high: vec![1.0; 2],
        };
        let act_space = ActionSpace::Box {
            low: vec![-1.0; 2],
            high: vec![1.0; 2],
        };
        let seq = GymVecEnv::new_with_send_envs(
            seq_envs,
            VecEnvConfig::new(2),
            obs_space.clone(),
            act_space.clone(),
        );
        assert_eq!(seq.runner().num_envs(), 2);

        // Parallel
        let par_envs: Vec<Box<dyn VecEnvInstance + Send>> = (0..2)
            .map(|_| Box::new(ConstEnv::new(2)) as Box<dyn VecEnvInstance + Send>)
            .collect();
        let par = GymVecEnv::new_with_send_envs(
            par_envs,
            VecEnvConfig::new(2).with_parallel(true),
            obs_space,
            act_space,
        );
        assert_eq!(par.runner().num_envs(), 2);

        // Touch the trait re-export to keep the public API alive.
        let _ = std::any::type_name::<ParallelVecEnvRunner>();
    }
}
