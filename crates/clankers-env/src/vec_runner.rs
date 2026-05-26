//! Sequential multi-environment driver and runner-abstraction trait.
//!
//! Exports the sequential [`VecEnvRunner`] plus the [`VecRunnerLike`]
//! trait that lets sequential and parallel runners plug into
//! `GymVecEnv` interchangeably.
//!
//! Manages a vector of `GymEnv`-like closures (environment factories),
//! stepping and resetting them independently. Observations and done flags
//! are stored in [`SoA`](super::vec_buffer) buffers for efficient batched
//! access.

use clankers_core::seed::SeedHierarchy;
use clankers_core::types::{Action, EnvId, Observation, ResetResult, StepResult};

use crate::vec_buffer::{VecDoneBuffer, VecObsBuffer};
use crate::vec_env::VecEnvConfig;
use crate::vec_episode::{AutoResetMode, EnvEpisodeMap};

/// Trait abstracting a single environment for `VecEnvRunner`.
///
/// Any type providing `step`, `reset`, and `obs_dim` can be used
/// as an environment in the vectorized runner.
///
/// The trait itself does not require `Send`; the *parallel* runner
/// stores `Box<dyn VecEnvInstance>` at the use site so a
/// thread-unsafe env can still ride the sequential path.
pub trait VecEnvInstance {
    /// Reset the environment.
    fn reset(&mut self, seed: Option<u64>) -> ResetResult;
    /// Step the environment with an action.
    fn step(&mut self, action: &Action) -> StepResult;
    /// Observation dimension.
    fn obs_dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// VecRunnerLike
// ---------------------------------------------------------------------------

/// Object-safe trait abstracting a vectorized runner.
///
/// Both the sequential [`VecEnvRunner`] and the Rayon-backed
/// [`ParallelVecEnvRunner`](crate::parallel_runner::ParallelVecEnvRunner)
/// implement this trait, so [`GymVecEnv`](../../clankers-gym/struct.GymVecEnv.html)
/// can hold a `Box<dyn VecRunnerLike>` and dispatch at runtime via
/// [`runner_for`].
///
/// # No `Send` super-bound
///
/// Sequential runs need to wrap `GymEnv` (which holds a Bevy `App`
/// containing a `Box<dyn FnOnce(App) -> AppExit>` runner field that
/// is `!Send`). Forcing `VecRunnerLike: Send` would gate the
/// sequential path on a stricter bound than `GymEnv` can satisfy.
/// Instead, `Send`-ness is enforced only at the *parallel* runner's
/// `Box<dyn VecEnvInstance>` use sites — sequential consumers
/// keep using `Box<dyn VecEnvInstance>`.
///
/// The [`runner_for`] factory therefore takes two parameters: one
/// path per bound. See its docs for the dispatch table.
///
/// # No globally-shared Bevy resources
///
/// Each implementor's environments **must not** share process-wide
/// Bevy resources (asset server, render device, task pool with shared
/// state). Each env owns a fully independent `App` or pure-data
/// fixture so that the parallel runner can step them concurrently
/// without cross-env corruption.
pub trait VecRunnerLike {
    /// Step every environment with the given actions.
    ///
    /// Writes observations into [`obs_buffer`](Self::obs_buffer) and
    /// done flags into [`done_buffer`](Self::done_buffer). The
    /// sequential and parallel paths share an identical book-keeping
    /// post-pass via the crate-private `finalize_step` helper so the
    /// only observable difference is throughput.
    fn step_all(&mut self, actions: &[Action]);
    /// Reset every environment, optionally with a base seed.
    fn reset_all(&mut self, seed: Option<u64>);
    /// Reset a specific subset of environments, with a shared seed.
    ///
    /// Single-env granular reset — used by the gym server to honour
    /// client-driven selective reset. Default impl loops over
    /// `env_ids` and falls back to a per-call reset; the sequential
    /// runner overrides with its inherent `reset_envs` for
    /// byte-equality with pre-W7 behaviour.
    fn reset_envs(&mut self, env_ids: &[EnvId], seed: Option<u64>);
    /// Number of environments.
    fn num_envs(&self) -> usize;
    /// Observation buffer (read-only).
    fn obs_buffer(&self) -> &VecObsBuffer;
    /// Done buffer (read-only).
    fn done_buffer(&self) -> &VecDoneBuffer;
    /// Episode map (read-only).
    fn episodes(&self) -> &EnvEpisodeMap;
    /// Get observation for a specific env.
    fn get_obs(&self, env_idx: usize) -> Observation;
}

// ---------------------------------------------------------------------------
// runner_for factory
// ---------------------------------------------------------------------------

/// Build a [`VecRunnerLike`] trait object whose concrete type is chosen by
/// [`VecEnvConfig::is_parallel`].
///
/// Accepts `Box<dyn VecEnvInstance>` since the parallel runner
/// always requires `Send` envs; the sequential runner happily widens
/// to that bound at zero cost.
///
/// - `config.is_parallel() == false` (default) → [`VecEnvRunner`].
/// - `config.is_parallel() == true` →
///   [`ParallelVecEnvRunner`](crate::parallel_runner::ParallelVecEnvRunner).
///
/// If your env type is `!Send` (e.g. `GymEnv` wrapping a Bevy `App`,
/// which holds a `!Send` runner closure), construct a sequential
/// [`VecEnvRunner`] directly with `Vec<Box<dyn VecEnvInstance>>` —
/// the parallel path is unreachable in that case anyway.
///
/// # Panics
///
/// Panics if `envs.len() != config.num_envs()` or if environments have
/// different observation dimensions (delegated to each runner's `new`).
#[must_use]
pub fn runner_for(
    envs: Vec<Box<dyn VecEnvInstance + Send>>,
    config: VecEnvConfig,
) -> Box<dyn VecRunnerLike> {
    if config.is_parallel() {
        Box::new(crate::parallel_runner::ParallelVecEnvRunner::new(
            envs, config,
        ))
    } else {
        // Widen the Send-bounded element to the plain trait object
        // accepted by the sequential runner — `Send` is a strictly
        // stronger bound, the cast is a no-op.
        let envs: Vec<Box<dyn VecEnvInstance>> = envs
            .into_iter()
            .map(|b| b as Box<dyn VecEnvInstance>)
            .collect();
        Box::new(VecEnvRunner::new(envs, config))
    }
}

// ---------------------------------------------------------------------------
// finalize_step helper (shared between sequential and parallel runners)
// ---------------------------------------------------------------------------

/// Per-step book-keeping shared by [`VecEnvRunner::step_all`] and
/// [`ParallelVecEnvRunner::step_all`](crate::parallel_runner::ParallelVecEnvRunner::step_all).
///
/// Given the index, freshly-stepped [`StepResult`], the episode map,
/// done buffer, and obs buffer, this advances the episode counter,
/// applies terminate / truncate / max-steps truncation, and writes the
/// fresh observation into [`VecObsBuffer`]. Extracted so both runners
/// stay byte-identical for non-rayon book-keeping.
pub(crate) fn finalize_step(
    i: usize,
    result: &StepResult,
    config: &VecEnvConfig,
    episodes: &mut EnvEpisodeMap,
    obs_buf: &mut VecObsBuffer,
    done_buf: &mut VecDoneBuffer,
) {
    obs_buf.set(i, &result.observation);
    done_buf.set(i, result.terminated, result.truncated);

    let env_id = EnvId(u16::try_from(i).expect("env index overflow"));
    let ep = episodes.get_mut(env_id);
    ep.advance();

    if result.terminated {
        ep.terminate();
    } else if result.truncated {
        ep.truncate();
    }

    // Check truncation based on max steps
    if !result.terminated && !result.truncated && ep.check_truncation(config.max_episode_steps()) {
        done_buf.set(i, false, true);
    }
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
/// let mut runner = VecEnvRunner::new(envs, config);
/// runner.reset_all(None);
/// let _obs = runner.obs_buffer();
/// ```
pub struct VecEnvRunner {
    envs: Vec<Box<dyn VecEnvInstance>>,
    config: VecEnvConfig,
    episodes: EnvEpisodeMap,
    obs_buf: VecObsBuffer,
    done_buf: VecDoneBuffer,
    /// Envs that terminated last step and need resetting (for `AutoResetMode::NextStep`).
    pending_resets: Vec<EnvId>,
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
            done_buf: VecDoneBuffer::new(n),
            pending_resets: Vec::new(),
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
            self.episodes
                .reset(EnvId(u16::try_from(i).expect("env index overflow")), seed);
        }
        self.done_buf.clear();
    }

    /// Reset all environments with per-env seeds derived from a [`SeedHierarchy`].
    ///
    /// Each environment receives a unique seed:
    /// `hierarchy.env_seed(env_index)`, ensuring independent but reproducible
    /// random streams.
    pub fn reset_all_from_hierarchy(&mut self, hierarchy: &SeedHierarchy) {
        for (i, env) in self.envs.iter_mut().enumerate() {
            let env_id = EnvId(u16::try_from(i).expect("env index overflow"));
            let seed = hierarchy.env_seed(env_id.index());
            let result = env.reset(Some(seed));
            self.obs_buf.set(i, &result.observation);
            self.episodes.reset(env_id, Some(seed));
        }
        self.done_buf.clear();
    }

    /// Reset specific environments by index.
    pub fn reset_envs(&mut self, env_ids: &[EnvId], seed: Option<u64>) {
        for &env_id in env_ids {
            let idx = usize::from(env_id.index());
            let result = self.envs[idx].reset(seed);
            self.obs_buf.set(idx, &result.observation);
            self.episodes.reset(env_id, seed);
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

        // NextStep: reset envs that terminated on the previous step.
        if self.config.auto_reset_mode() == AutoResetMode::NextStep {
            for env_id in self.pending_resets.drain(..) {
                let idx = usize::from(env_id.index());
                let result = self.envs[idx].reset(None);
                self.obs_buf.set(idx, &result.observation);
                self.episodes.reset(env_id, None);
                self.done_buf.set(idx, false, false);
            }
        }

        for (i, (env, action)) in self.envs.iter_mut().zip(actions.iter()).enumerate() {
            let result = env.step(action);
            finalize_step(
                i,
                &result,
                &self.config,
                &mut self.episodes,
                &mut self.obs_buf,
                &mut self.done_buf,
            );
        }

        // Auto-reset handling
        match self.config.auto_reset_mode() {
            AutoResetMode::Immediate => {
                let done_envs = self.episodes.done_envs();
                for env_id in done_envs {
                    let idx = usize::from(env_id.index());
                    let result = self.envs[idx].reset(None);
                    self.obs_buf.set(idx, &result.observation);
                    self.episodes.reset(env_id, None);
                }
            }
            AutoResetMode::NextStep => {
                self.pending_resets = self.episodes.done_envs();
            }
            AutoResetMode::Disabled => {}
        }
    }

    /// Get observation for a specific env.
    #[must_use]
    pub fn get_obs(&self, env_idx: usize) -> Observation {
        self.obs_buf.get(env_idx)
    }
}

// ---------------------------------------------------------------------------
// VecRunnerLike impl for VecEnvRunner
// ---------------------------------------------------------------------------

impl VecRunnerLike for VecEnvRunner {
    fn step_all(&mut self, actions: &[Action]) {
        Self::step_all(self, actions);
    }

    fn reset_all(&mut self, seed: Option<u64>) {
        Self::reset_all(self, seed);
    }

    fn reset_envs(&mut self, env_ids: &[EnvId], seed: Option<u64>) {
        Self::reset_envs(self, env_ids, seed);
    }

    fn num_envs(&self) -> usize {
        Self::num_envs(self)
    }

    fn obs_buffer(&self) -> &VecObsBuffer {
        Self::obs_buffer(self)
    }

    fn done_buffer(&self) -> &VecDoneBuffer {
        Self::done_buffer(self)
    }

    fn episodes(&self) -> &EnvEpisodeMap {
        Self::episodes(self)
    }

    fn get_obs(&self, env_idx: usize) -> Observation {
        Self::get_obs(self, env_idx)
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
    fn auto_reset_next_step() {
        // Env 0 terminates at step_count >= 2 (i.e. after 2 steps).
        let envs: Vec<Box<dyn VecEnvInstance>> = vec![
            Box::new(ConstEnv::terminating_at(2, 2)),
            Box::new(ConstEnv::new(2)),
        ];
        let config = VecEnvConfig::new(2).with_auto_reset(AutoResetMode::NextStep);
        let mut runner = VecEnvRunner::new(envs, config);
        runner.reset_all(None);

        let actions = vec![Action::zeros(2), Action::zeros(2)];

        // Step 1: env 0 step_count=1, not terminated yet.
        runner.step_all(&actions);
        assert!(!runner.done_buffer().terminated(0));

        // Step 2: env 0 step_count=2, terminates. Terminal obs preserved.
        runner.step_all(&actions);
        assert_eq!(runner.get_obs(0).as_slice(), &[2.0, 2.0]); // terminal obs kept
        assert!(runner.done_buffer().terminated(0));

        // Step 3: env 0 is auto-reset at the START, then stepped.
        // After reset step_count=0, step increments to 1 → obs=[1.0, 1.0].
        runner.step_all(&actions);
        assert_eq!(runner.get_obs(0).as_slice(), &[1.0, 1.0]); // fresh episode, step 1
        assert!(!runner.done_buffer().terminated(0));
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
    fn reset_all_from_hierarchy_assigns_per_env_seeds() {
        let envs = make_envs(3, 2);
        let config = VecEnvConfig::new(3);
        let mut runner = VecEnvRunner::new(envs, config);

        let hierarchy = SeedHierarchy::new(42);
        runner.reset_all_from_hierarchy(&hierarchy);

        let s0 = runner.episodes().get(EnvId(0)).seed;
        let s1 = runner.episodes().get(EnvId(1)).seed;
        let s2 = runner.episodes().get(EnvId(2)).seed;

        // All seeds present
        assert!(s0.is_some());
        assert!(s1.is_some());
        assert!(s2.is_some());

        // All seeds distinct
        assert_ne!(s0, s1);
        assert_ne!(s1, s2);
        assert_ne!(s0, s2);

        // Seeds are deterministic
        let envs2 = make_envs(3, 2);
        let mut runner2 = VecEnvRunner::new(envs2, VecEnvConfig::new(3));
        runner2.reset_all_from_hierarchy(&hierarchy);
        assert_eq!(
            runner.episodes().get(EnvId(0)).seed,
            runner2.episodes().get(EnvId(0)).seed
        );
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

    fn make_send_envs(n: usize, obs_dim: usize) -> Vec<Box<dyn VecEnvInstance + Send>> {
        (0..n)
            .map(|_| Box::new(ConstEnv::new(obs_dim)) as Box<dyn VecEnvInstance + Send>)
            .collect()
    }

    #[test]
    fn runner_for_dispatches_sequential_by_default() {
        let envs = make_send_envs(2, 2);
        let config = VecEnvConfig::new(2);
        let runner = runner_for(envs, config);
        assert_eq!(runner.num_envs(), 2);
    }

    #[test]
    fn runner_for_dispatches_parallel_when_flag_set() {
        let envs = make_send_envs(2, 2);
        let config = VecEnvConfig::new(2).with_parallel(true);
        let runner = runner_for(envs, config);
        assert_eq!(runner.num_envs(), 2);
    }
}
