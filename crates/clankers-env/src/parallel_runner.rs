//! `ParallelVecEnvRunner`: a Rayon-backed multi-environment runner with
//! deterministic per-env seed derivation.
//!
//! # Why `Mutex<Box<dyn VecEnvInstance + Send>>`?
//!
//! Each env sits behind a `Mutex` per WS7-plan § 8 risk 6. Inside
//! the parallel iterator we hold an exclusive `&mut Mutex<...>` so
//! `Mutex::get_mut` bypasses the lock entirely — the borrow checker
//! has already proven nothing else can touch the cell. The Mutex
//! wrapper is retained because:
//!
//! 1. It is the natural home for a future `clone`-friendly access
//!    pattern when the trait grows methods that don't take `&mut self`
//!    (e.g. read-only obs accessors during step-fuse).
//! 2. It documents at the type level that env access is exclusive
//!    per worker — reviewers immediately see "this is the boundary".
//! 3. Wrapping in `Mutex` only widens `Sync`, never narrows `Send`,
//!    so a future `par_iter` (read-only) path can lift to
//!    `T: Sync` cheaply via `Mutex::lock` without re-architecting.
//!
//! Contention is **zero** in the current `par_iter_mut` shape because
//! each env is touched by exactly one worker for the entire step.
//!
//! # Determinism
//!
//! Per-env seeds derive from a single base seed via a pure splitmix64-
//! style hash of `(base, idx)`. Because the hash is a pure function of
//! `(base, idx)`, the seed assigned to env `i` is **independent of
//! thread interleaving, Rayon work-stealing order, and pool size**.
//! See the crate-private `derive_seed` for the formula and the
//! `parallel_vec_env_seed_assignment_is_deterministic` integration
//! test for the byte-equal invariant.

use std::sync::Mutex;

use rayon::prelude::*;

use clankers_core::types::{Action, EnvId, Observation};

use crate::vec_buffer::{VecDoneBuffer, VecObsBuffer};
use crate::vec_env::VecEnvConfig;
use crate::vec_episode::{AutoResetMode, EnvEpisodeMap};
use crate::vec_runner::{VecEnvInstance, VecRunnerLike, finalize_step};

// ---------------------------------------------------------------------------
// derive_seed
// ---------------------------------------------------------------------------

/// Splitmix64-style per-env seed: `seed(base, i) = base * golden_64 + i`.
///
/// The constant `0x9E37_79B9_7F4A_7C15` is the 64-bit fractional part of
/// the golden ratio (same value used by splitmix64, xoroshiro, and
/// Knuth's multiplicative hash). Independent of thread interleaving —
/// every env at index `i` always resolves to the same seed regardless
/// of which Rayon worker services it.
///
/// # Why not [`SeedHierarchy::env_seed`](clankers_core::seed::SeedHierarchy::env_seed)?
///
/// `SeedHierarchy` exists and is used by the sequential
/// [`VecEnvRunner::reset_all_from_hierarchy`](crate::vec_runner::VecEnvRunner::reset_all_from_hierarchy)
/// path, but the WS7 plan explicitly specifies the splitmix64 formula
/// because the
/// `parallel_vec_env_seed_assignment_is_deterministic` integration
/// test pins it as a byte-equal invariant. Reaching for `env_seed`
/// here would couple the parallel runner's determinism contract to a
/// distant module's implementation choices.
#[inline]
#[must_use]
pub(crate) const fn derive_seed(base: u64, idx: usize) -> u64 {
    base.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(idx as u64)
}

// ---------------------------------------------------------------------------
// ParallelVecEnvRunner
// ---------------------------------------------------------------------------

/// Rayon-backed parallel multi-environment runner.
///
/// Drives `step_all` through `rayon::par_iter_mut`. Per-env seed
/// derivation is deterministic (see the crate-private `derive_seed`).
/// Results from parallel workers are collected into a
/// `Vec<(usize, StepResult)>` then merged into the buffer / episode
/// book-keeping on a single thread via the crate-private
/// `finalize_step` so the post-pass is byte-identical to the
/// sequential [`VecEnvRunner`](crate::vec_runner::VecEnvRunner).
///
/// # Example
///
/// ```
/// use clankers_env::parallel_runner::ParallelVecEnvRunner;
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
/// let envs: Vec<Box<dyn VecEnvInstance + Send>> =
///     vec![Box::new(DummyEnv), Box::new(DummyEnv)];
/// let config = VecEnvConfig::new(2).with_parallel(true);
/// let mut runner = ParallelVecEnvRunner::new(envs, config);
/// runner.reset_all(Some(0xDEAD_BEEF));
/// ```
pub struct ParallelVecEnvRunner {
    envs: Vec<Mutex<Box<dyn VecEnvInstance + Send>>>,
    config: VecEnvConfig,
    episodes: EnvEpisodeMap,
    obs_buf: VecObsBuffer,
    done_buf: VecDoneBuffer,
    /// Envs that terminated last step and need resetting (for `AutoResetMode::NextStep`).
    pending_resets: Vec<EnvId>,
    /// Last base seed used by [`Self::reset_all`]. Held so any
    /// auto-reset on terminal envs can re-derive a per-env seed
    /// consistent with the initial assignment.
    base_seed: Option<u64>,
}

impl ParallelVecEnvRunner {
    /// Create a parallel runner from a vec of `Send`-bounded environment
    /// instances and config.
    ///
    /// # Panics
    ///
    /// Panics if `envs.len() != config.num_envs()` or if environments
    /// have different observation dimensions.
    #[must_use]
    pub fn new(envs: Vec<Box<dyn VecEnvInstance + Send>>, config: VecEnvConfig) -> Self {
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
            envs: envs.into_iter().map(Mutex::new).collect(),
            config,
            episodes: EnvEpisodeMap::new(num_envs),
            obs_buf: VecObsBuffer::new(n, obs_dim),
            done_buf: VecDoneBuffer::new(n),
            pending_resets: Vec::new(),
            base_seed: None,
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

    /// Get observation for a specific env (clones out of the buffer).
    #[must_use]
    pub fn get_obs(&self, env_idx: usize) -> Observation {
        self.obs_buf.get(env_idx)
    }

    /// Reset specific environments by index.
    ///
    /// Mirrors the sequential
    /// [`VecEnvRunner::reset_envs`](crate::vec_runner::VecEnvRunner::reset_envs)
    /// surface: when `seed` is provided it is passed verbatim to each
    /// env (matching the sequential contract — the caller wants the
    /// *same* seed for every selected env). Per-env splitmix64
    /// derivation only applies to [`Self::reset_all`].
    pub fn reset_envs(&mut self, env_ids: &[EnvId], seed: Option<u64>) {
        for &env_id in env_ids {
            let idx = usize::from(env_id.index());
            let result = self.envs[idx]
                .get_mut()
                .expect("env mutex poisoned")
                .reset(seed);
            self.obs_buf.set(idx, &result.observation);
            self.episodes.reset(env_id, seed);
            self.done_buf.set(idx, false, false);
        }
    }

    /// The base seed last passed to [`Self::reset_all`].
    ///
    /// `None` before the first reset or when reset was called with
    /// `None`.
    #[must_use]
    pub const fn base_seed(&self) -> Option<u64> {
        self.base_seed
    }

    /// Reset all environments, optionally with a base seed.
    ///
    /// When `seed = Some(base)`, env `i` receives `derive_seed(base, i)`
    /// so the per-env seed stream is deterministic across runs **and**
    /// across thread interleavings. When `seed = None`, each env is
    /// reset with `None` (matches the sequential runner).
    pub fn reset_all(&mut self, seed: Option<u64>) {
        self.base_seed = seed;
        // Reset is itself a parallel operation in the worker count
        // sense, but we do not need to collect outputs — observations
        // are written directly into local results, then merged.
        let results: Vec<(usize, Observation, Option<u64>)> = self
            .envs
            .par_iter_mut()
            .enumerate()
            .map(|(i, env_lock)| {
                let env_seed = seed.map(|s| derive_seed(s, i));
                // `&mut Mutex<T>` is an exclusive reference; `get_mut`
                // bypasses the lock since the borrow checker has
                // already proven nothing else can touch the cell.
                let r = env_lock
                    .get_mut()
                    .expect("env mutex poisoned")
                    .reset(env_seed);
                (i, r.observation, env_seed)
            })
            .collect();
        for (i, obs, env_seed) in results {
            self.obs_buf.set(i, &obs);
            self.episodes.reset(
                EnvId(u16::try_from(i).expect("env index overflow")),
                env_seed,
            );
        }
        self.done_buf.clear();
        self.pending_resets.clear();
    }

    /// Step all environments with the given actions.
    ///
    /// # Panics
    ///
    /// Panics if `actions.len() != num_envs`.
    pub fn step_all(&mut self, actions: &[Action]) {
        assert_eq!(actions.len(), self.envs.len(), "action count mismatch");

        // NextStep: reset envs that terminated on the previous step,
        // re-deriving per-env seed from the stored base seed (matches
        // sequential auto-reset behaviour at parity-of-seed).
        if self.config.auto_reset_mode() == AutoResetMode::NextStep {
            let base = self.base_seed;
            for env_id in self.pending_resets.drain(..) {
                let idx = usize::from(env_id.index());
                let env_seed = base.map(|b| derive_seed(b, idx));
                let result = self.envs[idx]
                    .get_mut()
                    .expect("env mutex poisoned")
                    .reset(env_seed);
                self.obs_buf.set(idx, &result.observation);
                self.episodes.reset(env_id, env_seed);
                self.done_buf.set(idx, false, false);
            }
        }

        // Parallel step: collect (i, result) tuples, then merge in a
        // single-threaded post-pass via finalize_step. Ordering is
        // preserved by Rayon's enumerate-then-collect contract.
        let results: Vec<(usize, clankers_core::types::StepResult)> = self
            .envs
            .par_iter_mut()
            .enumerate()
            .map(|(i, env_lock)| {
                // `&mut Mutex<T>` → `get_mut` (no lock needed; see
                // the reset_all comment).
                let r = env_lock
                    .get_mut()
                    .expect("env mutex poisoned")
                    .step(&actions[i]);
                (i, r)
            })
            .collect();

        for (i, result) in &results {
            finalize_step(
                *i,
                result,
                &self.config,
                &mut self.episodes,
                &mut self.obs_buf,
                &mut self.done_buf,
            );
        }

        // Auto-reset handling (single-threaded; mirrors sequential).
        match self.config.auto_reset_mode() {
            AutoResetMode::Immediate => {
                let done_envs = self.episodes.done_envs();
                let base = self.base_seed;
                for env_id in done_envs {
                    let idx = usize::from(env_id.index());
                    let env_seed = base.map(|b| derive_seed(b, idx));
                    let result = self.envs[idx]
                        .get_mut()
                        .expect("env mutex poisoned")
                        .reset(env_seed);
                    self.obs_buf.set(idx, &result.observation);
                    self.episodes.reset(env_id, env_seed);
                }
            }
            AutoResetMode::NextStep => {
                self.pending_resets = self.episodes.done_envs();
            }
            AutoResetMode::Disabled => {}
        }
    }
}

// ---------------------------------------------------------------------------
// VecRunnerLike impl for ParallelVecEnvRunner
// ---------------------------------------------------------------------------

impl VecRunnerLike for ParallelVecEnvRunner {
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

    #[test]
    fn seed_derivation_is_deterministic() {
        // Same (base, idx) → same seed across calls.
        assert_eq!(derive_seed(0xDEAD, 0), derive_seed(0xDEAD, 0));
        assert_eq!(derive_seed(0xDEAD, 7), derive_seed(0xDEAD, 7));

        // Different idx → different seed for non-zero base.
        assert_ne!(derive_seed(0xDEAD, 0), derive_seed(0xDEAD, 1));
        assert_ne!(derive_seed(0xDEAD, 1), derive_seed(0xDEAD, 2));

        // The multiplicative-hash cosmetic property:
        // when base == 0, derive_seed(0, i) == i.
        for i in 0..16_usize {
            assert_eq!(derive_seed(0, i), i as u64);
        }
    }

    #[test]
    fn seed_derivation_changes_with_base() {
        // Different base → different seed for the same idx (non-zero idx).
        assert_ne!(derive_seed(0xDEAD_BEEF, 3), derive_seed(0xC0FF_EE00, 3));
    }
}
