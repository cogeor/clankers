//! Criterion target for the W7 PR1 parallel vec-env runner.
//!
//! Mirrors the body of `clankers bench vec` so the
//! `cargo bench --bench vec` numbers track the CLI numbers. Both code
//! paths drive `clankers_env::runner_for(envs, VecEnvConfig)` over a
//! `ConstBenchEnv` (a synthetic constant-observation env that isolates
//! the runner overhead). The CSV-emitting CLI path lives in
//! `apps/clankers-app/src/commands/bench.rs::bench_vec_cell`; this
//! Criterion target reimplements the same primitive so the bench is
//! self-contained (binary-only `clankers-app` cannot be `extern
//! crate`'d into a bench target).
//!
//! # Groups
//!
//! - `vec_step_all` — overhead-floor measurement at `work_us == 0`.
//!   The synthetic env's `step()` is ~50ns so parallel is *expected* to
//!   lose to sequential here: the group exists to track rayon dispatch
//!   overhead in isolation.
//! - `vec_throughput_100us` — realistic-work measurement at
//!   `work_us == 100`. The synthetic env busy-spins on `Instant::now()`
//!   for 100µs per `step()`, so parallel should beat sequential by
//!   roughly `min(num_cores, n_envs)` minus rayon overhead. This is the
//!   group the CI ratio gate cares about.
//!
//! # Comparison
//!
//! - Criterion: warm-up + statistical analysis per Criterion's defaults;
//!   `cargo bench --bench vec -- --output-format bencher` emits a
//!   line-protocol per-sample summary.
//! - CLI: simple wall-clock + percentile summary; deterministic
//!   `runs × max_steps` budget.
//!
//! Both call `runner.step_all(&actions)` repeatedly after a single
//! `runner.reset_all(Some(seed))`. The wall-clock per `step_all` is the
//! headline metric.

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use clankers_core::types::{Action, Observation, ResetInfo, ResetResult, StepInfo, StepResult};
use clankers_env::vec_env::VecEnvConfig;
use clankers_env::vec_runner::{VecEnvInstance, runner_for};

// ---------------------------------------------------------------------------
// ConstBenchEnv — same shape as the CLI bench helper
// ---------------------------------------------------------------------------

/// Synthetic constant env. See
/// `apps/clankers-app/src/commands/bench.rs::ConstBenchEnv` for the
/// canonical doc comment; this is a verbatim copy so the bench target is
/// self-contained (the binary crate `clankers-app` can't be linked into
/// a bench target). The two copies must keep the same shape so a future
/// consolidation into a shared `clankers-bench-support` crate can dedupe
/// trivially.
struct ConstBenchEnv {
    obs_dim: usize,
    work_us: u32,
}

impl VecEnvInstance for ConstBenchEnv {
    fn reset(&mut self, _seed: Option<u64>) -> ResetResult {
        ResetResult {
            observation: Observation::zeros(self.obs_dim),
            info: ResetInfo::default(),
        }
    }

    fn step(&mut self, _action: &Action) -> StepResult {
        if self.work_us > 0 {
            let target = Duration::from_micros(u64::from(self.work_us));
            let start = std::time::Instant::now();
            while start.elapsed() < target {
                std::hint::black_box(());
            }
        }
        StepResult {
            observation: Observation::zeros(self.obs_dim),
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

fn build_envs(n: u16, obs_dim: usize, work_us: u32) -> Vec<Box<dyn VecEnvInstance + Send>> {
    (0..n)
        .map(|_| Box::new(ConstBenchEnv { obs_dim, work_us }) as Box<dyn VecEnvInstance + Send>)
        .collect()
}

// ---------------------------------------------------------------------------
// Bench bodies
// ---------------------------------------------------------------------------

fn vec_step_all_8_envs(c: &mut Criterion) {
    let obs_dim = 4_usize;
    let n = 8_u16;
    let mut group = c.benchmark_group("vec_step_all");
    // 100 step_all calls per measurement = ~800 env-steps; criterion
    // amortises overhead by repeating the closure body.
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("sequential_8_envs", |b| {
        b.iter_batched(
            || {
                let envs = build_envs(n, obs_dim, 0);
                let cfg = VecEnvConfig::new(n).with_parallel(false);
                let mut runner = runner_for(envs, cfg);
                runner.reset_all(Some(0));
                let actions: Vec<Action> = (0..n).map(|_| Action::zeros(1)).collect();
                (runner, actions)
            },
            |(mut runner, actions)| {
                for _ in 0..100 {
                    runner.step_all(&actions);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("parallel_8_envs", |b| {
        b.iter_batched(
            || {
                let envs = build_envs(n, obs_dim, 0);
                let cfg = VecEnvConfig::new(n).with_parallel(true);
                let mut runner = runner_for(envs, cfg);
                runner.reset_all(Some(0));
                let actions: Vec<Action> = (0..n).map(|_| Action::zeros(1)).collect();
                (runner, actions)
            },
            |(mut runner, actions)| {
                for _ in 0..100 {
                    runner.step_all(&actions);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Realistic-work mirror of `vec_step_all_8_envs`: each `step()` spends
/// 100µs in a busy-loop so the rayon dispatch overhead becomes a small
/// fraction of the per-step budget and parallelism actually wins. Inner
/// loop reduced from 100 → 10 iterations (10 × 100µs × 8 envs ≈ 8ms per
/// sample) and `measurement_time` bumped to 5s for stable stats.
fn vec_throughput_100us_8_envs(c: &mut Criterion) {
    let obs_dim = 4_usize;
    let n = 8_u16;
    let work_us = 100_u32;
    let mut group = c.benchmark_group("vec_throughput_100us");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("sequential_8_envs", |b| {
        b.iter_batched(
            || {
                let envs = build_envs(n, obs_dim, work_us);
                let cfg = VecEnvConfig::new(n).with_parallel(false);
                let mut runner = runner_for(envs, cfg);
                runner.reset_all(Some(0));
                let actions: Vec<Action> = (0..n).map(|_| Action::zeros(1)).collect();
                (runner, actions)
            },
            |(mut runner, actions)| {
                for _ in 0..10 {
                    runner.step_all(&actions);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("parallel_8_envs", |b| {
        b.iter_batched(
            || {
                let envs = build_envs(n, obs_dim, work_us);
                let cfg = VecEnvConfig::new(n).with_parallel(true);
                let mut runner = runner_for(envs, cfg);
                runner.reset_all(Some(0));
                let actions: Vec<Action> = (0..n).map(|_| Action::zeros(1)).collect();
                (runner, actions)
            },
            |(mut runner, actions)| {
                for _ in 0..10 {
                    runner.step_all(&actions);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, vec_step_all_8_envs, vec_throughput_100us_8_envs);
criterion_main!(benches);
