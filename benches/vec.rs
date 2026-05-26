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
//! # Comparison
//!
//! - Criterion: warm-up + statistical analysis per Criterion's defaults;
//!   `cargo bench --bench vec -- --output-format bencher` emits a
//!   line-protocol per-sample summary.
//! - CLI: simple wall-clock + percentile summary; deterministic
//!   `runs × max_steps` budget.
//!
//! Both call `runner.step_all(&actions)` `max_steps` times after a
//! single `runner.reset_all(Some(seed))`. The wall-clock per
//! `step_all` is the headline metric.

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use clankers_core::types::{Action, Observation, ResetInfo, ResetResult, StepInfo, StepResult};
use clankers_env::vec_env::VecEnvConfig;
use clankers_env::vec_runner::{VecEnvInstance, runner_for};

// ---------------------------------------------------------------------------
// ConstBenchEnv — same shape as the CLI bench helper
// ---------------------------------------------------------------------------

struct ConstBenchEnv {
    obs_dim: usize,
}

impl VecEnvInstance for ConstBenchEnv {
    fn reset(&mut self, _seed: Option<u64>) -> ResetResult {
        ResetResult {
            observation: Observation::zeros(self.obs_dim),
            info: ResetInfo::default(),
        }
    }

    fn step(&mut self, _action: &Action) -> StepResult {
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

fn build_envs(n: u16, obs_dim: usize) -> Vec<Box<dyn VecEnvInstance + Send>> {
    (0..n)
        .map(|_| Box::new(ConstBenchEnv { obs_dim }) as Box<dyn VecEnvInstance + Send>)
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
                let envs = build_envs(n, obs_dim);
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
                let envs = build_envs(n, obs_dim);
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

criterion_group!(benches, vec_step_all_8_envs);
criterion_main!(benches);
