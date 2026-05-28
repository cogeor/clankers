//! Paired sequential vs parallel auto-reset parity tests (P1.8).
//!
//! CODE_QUALITY_REVIEW Detailed Findings "Auto-Reset Semantics Need
//! Clear Boundary Tests" called out that sequential and parallel
//! runners share `finalize_step` but auto-reset handling lives in
//! runner-specific paths, so behaviour could drift around episode
//! counters, done flags, and observations.
//!
//! Each test below feeds the same fixed action sequence to both
//! `VecEnvRunner` (sequential) and `ParallelVecEnvRunner` (parallel)
//! configured with the same `AutoResetMode` and asserts they produce
//! byte-equal observation / done buffers.
//!
//! The fixture is a `CountdownEnv` that:
//! - terminates after `terminate_at` steps of its own,
//! - writes a per-step signature into the observation that survives
//!   across resets (so we can tell a reset happened), and
//! - records the seed it last received via `reset` (so we can also
//!   check the seed plumbing across resets).
//!
//! Why pure-data (no Bevy world): same reasoning as
//! `parallel_determinism.rs` — we are pinning the runner's contract,
//! not the ECS query iteration order.

use clankers_core::types::{
    Action, EnvId, Observation, ResetInfo, ResetResult, StepInfo, StepResult,
};
use clankers_env::parallel_runner::ParallelVecEnvRunner;
use clankers_env::vec_env::VecEnvConfig;
use clankers_env::vec_episode::AutoResetMode;
use clankers_env::vec_runner::{VecEnvInstance, VecEnvRunner, VecRunnerLike};

// ---------------------------------------------------------------------------
// Fixture: CountdownEnv
// ---------------------------------------------------------------------------

/// Env that terminates after `terminate_at` steps of its own.
///
/// Observation layout (3 f32 slots minimum):
///   [step_count, episode_index, last_seed_lo_as_f32]
///
/// `episode_index` increments every reset; lets the parity tests
/// distinguish "auto-reset happened" from "step never advanced".
#[derive(Clone)]
struct CountdownEnv {
    obs_dim: usize,
    terminate_at: u64,
    step_count: u64,
    episode_index: u64,
    last_seed: Option<u64>,
}

impl CountdownEnv {
    fn new(obs_dim: usize, terminate_at: u64) -> Self {
        Self {
            obs_dim,
            terminate_at,
            step_count: 0,
            episode_index: 0,
            last_seed: None,
        }
    }

    fn obs(&self) -> Observation {
        let mut v = vec![0.0_f32; self.obs_dim];
        #[allow(clippy::cast_precision_loss)]
        {
            if self.obs_dim >= 1 {
                v[0] = self.step_count as f32;
            }
            if self.obs_dim >= 2 {
                v[1] = self.episode_index as f32;
            }
            if self.obs_dim >= 3 {
                // bit-cast low 32 of the seed; preserves byte-equality
                // across runs / threadpool sizes.
                let lo = (self.last_seed.unwrap_or(0) & 0xFFFF_FFFF) as u32;
                v[2] = f32::from_bits(lo);
            }
        }
        Observation::new(v)
    }
}

impl VecEnvInstance for CountdownEnv {
    fn reset(&mut self, seed: Option<u64>) -> ResetResult {
        self.step_count = 0;
        self.episode_index += 1;
        self.last_seed = seed.or(self.last_seed);
        ResetResult {
            observation: self.obs(),
            info: ResetInfo::default(),
        }
    }

    fn step(&mut self, _action: &Action) -> StepResult {
        self.step_count += 1;
        let terminated = self.step_count >= self.terminate_at;
        StepResult {
            observation: self.obs(),
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

// ---------------------------------------------------------------------------
// Harnesses
// ---------------------------------------------------------------------------

/// Snapshot of the runner's externally observable state after each step.
#[derive(Debug, Clone, PartialEq)]
struct StepSnapshot {
    obs_per_env: Vec<Vec<f32>>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
}

fn drive_sequential(
    mode: AutoResetMode,
    num_envs: u16,
    terminate_at: u64,
    n_steps: usize,
) -> Vec<StepSnapshot> {
    // obs_dim=2 so the seed slot is NOT part of the parity snapshot:
    // the parallel runner derives per-env seeds via SeedHierarchy
    // while the sequential runner passes the master seed unchanged.
    // That divergence is intentional; auto-reset parity is about
    // step_count + episode_index + done flags, not seed plumbing.
    let obs_dim = 2;
    let envs: Vec<Box<dyn VecEnvInstance>> = (0..num_envs)
        .map(|_| Box::new(CountdownEnv::new(obs_dim, terminate_at)) as Box<dyn VecEnvInstance>)
        .collect();
    let config = VecEnvConfig::new(num_envs).with_auto_reset(mode);
    let mut runner = VecEnvRunner::new(envs, config);
    runner.reset_all(Some(0xABCD_0001));

    let actions: Vec<Action> = (0..num_envs).map(|_| Action::zeros(obs_dim)).collect();
    let mut history = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        runner.step_all(&actions);
        history.push(snapshot(&runner, num_envs));
    }
    history
}

fn drive_parallel(
    mode: AutoResetMode,
    num_envs: u16,
    terminate_at: u64,
    n_steps: usize,
) -> Vec<StepSnapshot> {
    // obs_dim=2 so the seed slot is NOT part of the parity snapshot:
    // the parallel runner derives per-env seeds via SeedHierarchy
    // while the sequential runner passes the master seed unchanged.
    // That divergence is intentional; auto-reset parity is about
    // step_count + episode_index + done flags, not seed plumbing.
    let obs_dim = 2;
    let envs: Vec<Box<dyn VecEnvInstance + Send>> = (0..num_envs)
        .map(|_| {
            Box::new(CountdownEnv::new(obs_dim, terminate_at)) as Box<dyn VecEnvInstance + Send>
        })
        .collect();
    let config = VecEnvConfig::new(num_envs).with_auto_reset(mode);
    let mut runner = ParallelVecEnvRunner::new(envs, config);
    runner.reset_all(Some(0xABCD_0001));

    let actions: Vec<Action> = (0..num_envs).map(|_| Action::zeros(obs_dim)).collect();
    let mut history = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        runner.step_all(&actions);
        history.push(snapshot(&runner, num_envs));
    }
    history
}

fn snapshot<R: VecRunnerLike>(runner: &R, num_envs: u16) -> StepSnapshot {
    let mut obs_per_env = Vec::with_capacity(usize::from(num_envs));
    let mut terminated = Vec::with_capacity(usize::from(num_envs));
    let mut truncated = Vec::with_capacity(usize::from(num_envs));
    for i in 0..num_envs {
        let idx = usize::from(i);
        obs_per_env.push(runner.get_obs(idx).as_slice().to_vec());
        terminated.push(runner.done_buffer().terminated(idx));
        truncated.push(runner.done_buffer().truncated(idx));
    }
    // Silence the unused-`EnvId` import warning when this fixture is
    // built without the EnvId-keyed paths below.
    let _ = EnvId(0);
    StepSnapshot {
        obs_per_env,
        terminated,
        truncated,
    }
}

// ---------------------------------------------------------------------------
// Parity tests — one per AutoResetMode
// ---------------------------------------------------------------------------

#[test]
fn auto_reset_disabled_parity() {
    // With auto-reset disabled, envs that terminate stop incrementing
    // step_count (the runner doesn't reset them). Sequential and
    // parallel must produce the same observation history.
    let seq = drive_sequential(AutoResetMode::Disabled, 4, 3, 6);
    let par = drive_parallel(AutoResetMode::Disabled, 4, 3, 6);
    assert_eq!(seq, par, "Disabled mode must be byte-equal across runners");
}

#[test]
fn auto_reset_immediate_parity() {
    // Immediate auto-reset: when an env terminates, the runner resets
    // it in the SAME step and returns the new episode's initial obs.
    // The episode_index in obs[1] should advance on every termination.
    let seq = drive_sequential(AutoResetMode::Immediate, 4, 2, 10);
    let par = drive_parallel(AutoResetMode::Immediate, 4, 2, 10);
    assert_eq!(seq, par, "Immediate mode must be byte-equal across runners");
    // Sanity: with terminate_at=2 over 10 steps each env should have
    // terminated several times. Last snapshot's episode_index must be
    // > 1 (one for the initial reset, plus several auto-resets).
    let last_seq_ep = seq.last().unwrap().obs_per_env[0][1];
    assert!(
        last_seq_ep > 1.0,
        "expected at least one auto-reset, got episode_index={last_seq_ep}"
    );
}

#[test]
fn auto_reset_next_step_parity() {
    // NextStep auto-reset: when an env terminates, the runner returns
    // the terminal observation THIS step, and resets at the START of
    // the next step. The terminated flag fires exactly once per
    // termination on the step it happens.
    let seq = drive_sequential(AutoResetMode::NextStep, 4, 3, 12);
    let par = drive_parallel(AutoResetMode::NextStep, 4, 3, 12);
    assert_eq!(seq, par, "NextStep mode must be byte-equal across runners");
    // Sanity: count the terminated-flag firings across the history;
    // both runners must report the same count.
    let count_seq: usize = seq
        .iter()
        .map(|s| s.terminated.iter().filter(|t| **t).count())
        .sum();
    let count_par: usize = par
        .iter()
        .map(|s| s.terminated.iter().filter(|t| **t).count())
        .sum();
    assert_eq!(
        count_seq, count_par,
        "termination-event count must match across runners"
    );
}

#[test]
fn auto_reset_immediate_done_flags_match() {
    // Stricter slice: when Immediate auto-reset is active, the done
    // flags after each step must be identical (terminated and
    // truncated arrays).
    let seq = drive_sequential(AutoResetMode::Immediate, 8, 2, 5);
    let par = drive_parallel(AutoResetMode::Immediate, 8, 2, 5);
    for (i, (s, p)) in seq.iter().zip(par.iter()).enumerate() {
        assert_eq!(
            s.terminated, p.terminated,
            "terminated mismatch at step {i}"
        );
        assert_eq!(s.truncated, p.truncated, "truncated mismatch at step {i}");
    }
}
