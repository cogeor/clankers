//! Integration tests pinning the `ParallelVecEnvRunner` byte-equal
//! determinism contract (WS7 PR1 § 6).
//!
//! These tests deliberately use a **pure-data** [`SeedEchoEnv`]
//! fixture (no Bevy `World`, no rapier physics) so the assertions are
//! about the parallel runner's per-env seed derivation and result
//! ordering, **not** about Bevy ECS query iteration order. Per
//! WS7-plan § 8 risk 5 and MEMORY.md, do **not** port these
//! assertions to a Bevy-backed env without first checking the
//! layout-bound observation contract from W2 PR1.
//!
//! # Thread-pool size determinism
//!
//! The gate runs these tests under both the default Rayon pool
//! (logical-core count) and `RAYON_NUM_THREADS=4`. The seed
//! derivation is a pure function of `(base, idx)` so the snapshots
//! are byte-equal across both invocations regardless of work-stealing
//! order.

use clankers_core::types::{
    Action, EnvId, Observation, ResetInfo, ResetResult, StepInfo, StepResult,
};
use clankers_env::parallel_runner::ParallelVecEnvRunner;
use clankers_env::vec_env::VecEnvConfig;
use clankers_env::vec_runner::VecEnvInstance;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Pure-data env: echoes the per-env seed into observations on every
/// step. Used to assert the parallel runner assigns deterministic
/// per-env seeds (the byte-equal invariant the determinism test
/// pins).
struct SeedEchoEnv {
    obs_dim: usize,
    /// Last seed received by `reset` (writes into obs on every step).
    received_seed: Option<u64>,
    step_count: u64,
}

impl SeedEchoEnv {
    const fn new(obs_dim: usize) -> Self {
        Self {
            obs_dim,
            received_seed: None,
            step_count: 0,
        }
    }

    /// Build a deterministic observation from `(received_seed, step_count)`.
    /// Bit-pattern of the seed is bit-cast into the first two f32 slots
    /// (split into hi/lo 32-bit halves), `step_count` into the third, and
    /// zeros elsewhere — gives a byte-equal-friendly signature.
    fn obs(&self) -> Observation {
        let s = self.received_seed.unwrap_or(0);
        let hi = (s >> 32) as u32;
        let lo = (s & 0xFFFF_FFFF) as u32;
        let mut v = vec![0.0_f32; self.obs_dim];
        if self.obs_dim >= 1 {
            v[0] = f32::from_bits(hi);
        }
        if self.obs_dim >= 2 {
            v[1] = f32::from_bits(lo);
        }
        if self.obs_dim >= 3 {
            // Cast is intentional: small step counts compare byte-equal
            // across runs because we use the same value path on every
            // run.
            #[allow(clippy::cast_precision_loss)]
            {
                v[2] = self.step_count as f32;
            }
        }
        Observation::new(v)
    }
}

impl VecEnvInstance for SeedEchoEnv {
    fn reset(&mut self, seed: Option<u64>) -> ResetResult {
        self.received_seed = seed;
        self.step_count = 0;
        ResetResult {
            observation: self.obs(),
            info: ResetInfo::default(),
        }
    }

    fn step(&mut self, _action: &Action) -> StepResult {
        self.step_count += 1;
        StepResult {
            observation: self.obs(),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info: StepInfo::default(),
        }
    }

    fn obs_dim(&self) -> usize {
        self.obs_dim
    }
}

/// Per-env constant: env `i` always returns `[i; obs_dim]` on every
/// step. Used to assert Rayon-parallel result ordering does not get
/// scrambled by work stealing.
struct IndexedConstEnv {
    index: u32,
    obs_dim: usize,
}

impl IndexedConstEnv {
    const fn new(index: u32, obs_dim: usize) -> Self {
        Self { index, obs_dim }
    }
}

impl VecEnvInstance for IndexedConstEnv {
    fn reset(&mut self, _seed: Option<u64>) -> ResetResult {
        #[allow(clippy::cast_precision_loss)]
        let v = vec![self.index as f32; self.obs_dim];
        ResetResult {
            observation: Observation::new(v),
            info: ResetInfo::default(),
        }
    }

    fn step(&mut self, _action: &Action) -> StepResult {
        #[allow(clippy::cast_precision_loss)]
        let v = vec![self.index as f32; self.obs_dim];
        StepResult {
            observation: Observation::new(v),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info: StepInfo::default(),
        }
    }

    fn obs_dim(&self) -> usize {
        self.obs_dim
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `ParallelVecEnvRunner` of `n` `SeedEchoEnv` (`obs_dim=4`),
/// reset it with `base_seed`, then step it `step_count` times with
/// zero actions. Returns the bit-pattern of every f32 in the obs
/// buffer at the end — comparing snapshots with `==` on a
/// `Vec<u32>` dodges NaN traps even though the test data is not
/// NaN-bearing.
fn run_parallel_snapshot(n: u16, base_seed: u64, step_count: usize) -> Vec<u32> {
    let obs_dim = 4;
    let envs: Vec<Box<dyn VecEnvInstance + Send>> = (0..n)
        .map(|_| Box::new(SeedEchoEnv::new(obs_dim)) as Box<dyn VecEnvInstance + Send>)
        .collect();
    let config = VecEnvConfig::new(n).with_parallel(true);
    let mut runner = ParallelVecEnvRunner::new(envs, config);
    runner.reset_all(Some(base_seed));

    let zero_actions: Vec<Action> = (0..n).map(|_| Action::zeros(2)).collect();
    for _ in 0..step_count {
        runner.step_all(&zero_actions);
    }

    runner
        .obs_buffer()
        .as_flat()
        .iter()
        .map(|f| f.to_bits())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// 1. `parallel_vec_env_seed_assignment_is_deterministic`
///
/// Build 16 `SeedEchoEnv` wrapped in a `ParallelVecEnvRunner` with
/// `base_seed = 0xDEAD_BEEF`. Call `reset_all(Some(base_seed))` then
/// 100 `step_all(&zero_actions)`. Read the full
/// `obs_buffer().as_flat()` into a `Vec<u32>` bit-pattern snapshot.
///
/// **Repeats the whole construction-+-run cycle 5 times** (per
/// loop's intensification constraint). Asserts every snapshot is
/// `==` (bit-equal — bypasses NaN traps via `to_bits()`).
///
/// The gate runs this test under both the default Rayon pool size
/// and `RAYON_NUM_THREADS=4`; both invocations must yield the same
/// snapshot.
#[test]
fn parallel_vec_env_seed_assignment_is_deterministic() {
    let base_seed: u64 = 0xDEAD_BEEF;
    let n: u16 = 16;
    let steps: usize = 100;

    let snapshot0 = run_parallel_snapshot(n, base_seed, steps);
    assert!(!snapshot0.is_empty(), "snapshot must be non-empty");

    // Re-run the same configuration four more times; every snapshot
    // must be byte-equal to the first.
    for trial in 1..5 {
        let snap = run_parallel_snapshot(n, base_seed, steps);
        assert_eq!(
            snap, snapshot0,
            "trial {trial} snapshot differs from trial 0 — parallel \
             runner is non-deterministic under thread interleaving"
        );
    }

    // Spot-check that env 0 actually received a non-trivial seed.
    // derive_seed(base, 0) = base * 0x9E37_79B9_7F4A_7C15 + 0 ≠ 0
    // and is encoded in obs[0..2] as hi/lo halves.
    let expected_seed_0: u64 = base_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let hi0 = (expected_seed_0 >> 32) as u32;
    let lo0 = (expected_seed_0 & 0xFFFF_FFFF) as u32;
    assert_eq!(
        snapshot0[0],
        f32::from_bits(hi0).to_bits(),
        "env 0 obs[0] should encode derived seed hi half"
    );
    assert_eq!(
        snapshot0[1],
        f32::from_bits(lo0).to_bits(),
        "env 0 obs[1] should encode derived seed lo half"
    );
}

/// 2. `parallel_vec_env_changes_base_seed_changes_output`
///
/// Same fixture, two runs with different `base_seed`. The output
/// snapshot must differ in at least one f32 — proves the seed wiring
/// reaches the env.
#[test]
fn parallel_vec_env_changes_base_seed_changes_output() {
    let n: u16 = 16;
    let steps: usize = 50;
    let snap_a = run_parallel_snapshot(n, 0xDEAD_BEEF, steps);
    let snap_b = run_parallel_snapshot(n, 0xC0FF_EE00, steps);
    assert_ne!(
        snap_a, snap_b,
        "different base seeds should yield different obs snapshots"
    );
}

/// 3. `parallel_vec_env_preserves_result_ordering`
///
/// Build 8 envs where env `i` returns `Observation::Continuous(vec![i; 4])`
/// on every step (deterministic-by-index). After
/// `step_all(&zero_actions)`, assert
/// `obs_buffer.row(i).as_f32() == &[i; 4]` for all `i`. Catches
/// Rayon work-stealing reordering bugs.
#[test]
fn parallel_vec_env_preserves_result_ordering() {
    let n: u16 = 8;
    let obs_dim = 4;
    let envs: Vec<Box<dyn VecEnvInstance + Send>> = (0..n)
        .map(|i| {
            Box::new(IndexedConstEnv::new(u32::from(i), obs_dim)) as Box<dyn VecEnvInstance + Send>
        })
        .collect();
    let config = VecEnvConfig::new(n).with_parallel(true);
    let mut runner = ParallelVecEnvRunner::new(envs, config);
    runner.reset_all(Some(0xCAFE));

    let actions: Vec<Action> = (0..n).map(|_| Action::zeros(2)).collect();
    runner.step_all(&actions);

    let obs_buf = runner.obs_buffer();
    for i in 0..usize::from(n) {
        let row = obs_buf.row(i);
        #[allow(clippy::cast_precision_loss)]
        let expected = vec![i as f32; obs_dim];
        assert_eq!(
            row.as_f32(),
            expected.as_slice(),
            "env {i} obs scrambled — Rayon result ordering broke"
        );
    }
}

/// 4. `parallel_vec_env_reset_envs_round_trips`
///
/// Sanity check that the parallel runner's `reset_envs` is reachable
/// through the trait surface and produces fresh per-env observations.
#[test]
fn parallel_vec_env_reset_envs_round_trips() {
    let n: u16 = 4;
    let obs_dim = 4;
    let envs: Vec<Box<dyn VecEnvInstance + Send>> = (0..n)
        .map(|_| Box::new(SeedEchoEnv::new(obs_dim)) as Box<dyn VecEnvInstance + Send>)
        .collect();
    let config = VecEnvConfig::new(n).with_parallel(true);
    let mut runner = ParallelVecEnvRunner::new(envs, config);
    runner.reset_all(Some(0xCAFE));

    // Step a couple of times to drive step_count forward.
    let actions: Vec<Action> = (0..n).map(|_| Action::zeros(2)).collect();
    runner.step_all(&actions);
    runner.step_all(&actions);

    // Reset env 1 with an explicit seed; observations should encode it.
    let new_seed: u64 = 0x1234_5678_9ABC_DEF0;
    runner.reset_envs(&[EnvId(1)], Some(new_seed));
    let obs1 = runner.get_obs(1);
    let hi = (new_seed >> 32) as u32;
    let lo = (new_seed & 0xFFFF_FFFF) as u32;
    assert_eq!(obs1.as_slice()[0].to_bits(), f32::from_bits(hi).to_bits());
    assert_eq!(obs1.as_slice()[1].to_bits(), f32::from_bits(lo).to_bits());
}
