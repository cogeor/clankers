//! `bench vec` body — parallel vec-env runner throughput sweep.
//!
//! Sweeps over a list of env counts (`--envs N1,N2,...`) and for each
//! cell measures sequential vs parallel `step_all` wall clock on a
//! synthetic [`ConstBenchEnv`]. Emits one V2 CSV row per cell with the
//! parallel/sequential `throughput_x` ratio populated, then optionally
//! gates on that ratio via [`super::gate::evaluate_gate`].
//!
//! Also exposes the public [`bench_vec_cell`] entry point used by the
//! Criterion target `benches/vec.rs` so `cargo bench --bench vec`
//! numbers can be cross-checked against `clankers bench vec`.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use clankers_core::types::{Action, Observation, ResetInfo, ResetResult, StepInfo, StepResult};
use clankers_env::prelude::*;
use clankers_env::vec_runner::{VecEnvInstance, runner_for};

use super::args::{BenchArgs, VecArgs};
use super::csv::{print_human_v2, write_csv_row_v2};
use super::gate::evaluate_gate;
use super::stats::{aggregate_v2, parse_env_list, seq_mean, seq_steps_per_sec_mean};

// ---------------------------------------------------------------------------
// ConstBenchEnv
// ---------------------------------------------------------------------------

/// Synthetic constant env used by `bench vec` / `bench protocol`.
///
/// Cheap to construct + step so the measurement isolates the runner
/// overhead. Uses an `obs_dim`-wide constant observation and ignores
/// the action.
///
/// `work_us` is an opt-in per-`step()` busy-loop knob. When zero (the
/// default) `step()` is a handful of nanoseconds — useful for measuring
/// runner overhead floors. When `>0`, `step()` busy-spins on
/// `Instant::now()` until the requested microsecond budget has elapsed,
/// with `std::hint::black_box(())` inside the loop body to prevent the
/// optimiser from eliding the wait. This makes the env representative
/// of realistic per-step work so the parallel runner can actually amortise
/// rayon's dispatch overhead. The busy-loop is sensitive to CPU frequency
/// scaling, but `bench vec` reports a *ratio* (parallel / sequential) so
/// both arms see the same scaling.
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
            let start = Instant::now();
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

// ---------------------------------------------------------------------------
// bench vec body
// ---------------------------------------------------------------------------

pub(super) fn execute(args: &BenchArgs, vec_args: &VecArgs) -> ExitCode {
    let env_counts = match parse_env_list(&vec_args.envs) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bench vec: {e}");
            return ExitCode::from(1);
        }
    };

    if env_counts.is_empty() {
        eprintln!("bench vec: --envs must contain at least one value");
        return ExitCode::from(1);
    }

    let obs_dim: usize = 4; // cartpole-like dims for representative numbers
    let seed = args.seed.unwrap_or(0);
    let work_us = vec_args.work_us;

    if vec_args.ratio_gate > 0.0 && work_us == 0 {
        eprintln!(
            "bench vec: --ratio-gate {:.2} requires --work-us > 0 (the overhead-floor scenario `vec_parallel` cannot satisfy a >1.0 ratio gate). Re-run with --work-us 100 or drop --ratio-gate.",
            vec_args.ratio_gate
        );
        return ExitCode::from(2);
    }

    let mut gate_rows: Vec<(u16, f64)> = Vec::with_capacity(env_counts.len());

    for &n in &env_counts {
        // Warmup: parallel and sequential each warmed once.
        let _ = run_vec_cell(
            n,
            obs_dim,
            args.warmup_runs,
            args.max_steps,
            seed,
            true,
            work_us,
        );
        let _ = run_vec_cell(
            n,
            obs_dim,
            args.warmup_runs,
            args.max_steps,
            seed,
            false,
            work_us,
        );

        let (par_wall_ms, par_steps_per_sec, par_step_durs) =
            run_vec_cell(n, obs_dim, args.runs, args.max_steps, seed, true, work_us);
        let (seq_wall_ms, seq_steps_per_sec, _) =
            run_vec_cell(n, obs_dim, args.runs, args.max_steps, seed, false, work_us);

        let throughput_ratio = if seq_steps_per_sec_mean(&seq_steps_per_sec) > 0.0 {
            seq_steps_per_sec_mean(&par_steps_per_sec) / seq_steps_per_sec_mean(&seq_steps_per_sec)
        } else {
            0.0
        };

        gate_rows.push((n, throughput_ratio));

        if n == 8 && throughput_ratio < 3.0 && throughput_ratio > 0.0 {
            eprintln!(
                "bench vec: WARNING — N=8 parallel/sequential ratio = {throughput_ratio:.2}, below the WS7 § 7 gate of 3.0 (informational; CI baseline gate decides PASS/FAIL)"
            );
        }

        let total_steps = u32::try_from(par_wall_ms.len())
            .unwrap_or(u32::MAX)
            .saturating_mul(args.max_steps);
        let mut step_durs_owned = par_step_durs.clone();
        let scenario = if work_us == 0 {
            "vec_parallel".to_string()
        } else {
            format!("vec_throughput_{work_us}us")
        };
        let row = aggregate_v2(
            &scenario,
            args,
            total_steps,
            &par_wall_ms,
            &par_steps_per_sec,
            &mut step_durs_owned,
            u32::from(n),
            0,
            throughput_ratio,
            &format!(
                "kind=vec;parallel=true;seq_wall_ms_mean={:.3};work_us={work_us}",
                seq_mean(&seq_wall_ms)
            ),
        );

        if let Some(path) = args.csv.as_ref()
            && let Err(e) = write_csv_row_v2(path, &row)
        {
            eprintln!("bench vec: failed to write CSV: {e}");
            return ExitCode::from(1);
        }

        if args.json {
            if let Ok(s) = serde_json::to_string(&row) {
                println!("{s}");
            }
        } else if args.csv.is_none() {
            print_human_v2(&row);
        }
    }

    evaluate_gate(&gate_rows, vec_args.ratio_gate)
}

fn run_vec_cell(
    n: u16,
    obs_dim: usize,
    runs: u32,
    max_steps: u32,
    seed: u64,
    parallel: bool,
    work_us: u32,
) -> (Vec<f64>, Vec<f64>, Vec<Duration>) {
    let mut per_run_wall_ms = Vec::with_capacity(runs as usize);
    let mut per_run_sps = Vec::with_capacity(runs as usize);
    let mut all_step_durs: Vec<Duration> = Vec::new();

    for _ in 0..runs {
        // Fresh runner per measurement run — matches scenario path
        // behaviour where each run gets a clean App.
        let envs: Vec<Box<dyn VecEnvInstance + Send>> = (0..n)
            .map(|_| Box::new(ConstBenchEnv { obs_dim, work_us }) as Box<dyn VecEnvInstance + Send>)
            .collect();
        let cfg = VecEnvConfig::new(n).with_parallel(parallel);
        let mut runner = runner_for(envs, cfg);
        runner.reset_all(Some(seed));
        let action_template = Action::zeros(1);
        let actions: Vec<Action> = (0..n).map(|_| action_template.clone()).collect();

        let wall_start = Instant::now();
        let mut step_durs = Vec::with_capacity(max_steps as usize);
        for _ in 0..max_steps {
            let s = Instant::now();
            runner.step_all(&actions);
            step_durs.push(s.elapsed());
        }
        let wall = wall_start.elapsed();
        let wall_ms = wall.as_secs_f64() * 1000.0;
        per_run_wall_ms.push(wall_ms);
        let total_env_steps = f64::from(max_steps) * f64::from(n);
        let sps = if wall_ms > 0.0 {
            total_env_steps / wall_ms * 1000.0
        } else {
            0.0
        };
        per_run_sps.push(sps);
        all_step_durs.extend(step_durs);
    }

    (per_run_wall_ms, per_run_sps, all_step_durs)
}

// ---------------------------------------------------------------------------
// Public entry point for the Criterion target
// ---------------------------------------------------------------------------

/// Public entry point used by the Criterion target `benches/vec.rs` so
/// `cargo bench --bench vec` numbers match `clankers bench vec`.
///
/// Runs `runs` measurement runs of `max_steps` `step_all` calls on
/// `n_envs` `ConstBenchEnv` instances, returning the mean steps-per-sec.
///
/// `work_us` is the per-`step()` busy-loop duration in microseconds. Pass
/// `0` for the overhead-floor measurement (matches the
/// `vec_step_all` Criterion group + `vec_parallel` CLI scenario). Pass a
/// positive value (e.g. `100`) to mirror the `vec_throughput_{work_us}us`
/// scenario where parallelism can actually amortise rayon's dispatch
/// overhead.
///
/// `#[allow(dead_code)]` because clankers-app is a binary crate; the
/// bench target is the only external consumer and Cargo doesn't see
/// the cross-target use as "live" from the main.rs perspective.
#[must_use]
#[allow(dead_code)]
pub fn bench_vec_cell(n: u16, runs: u32, max_steps: u32, parallel: bool, work_us: u32) -> f64 {
    let (_, sps, _) = run_vec_cell(n, 4, runs, max_steps, 0, parallel, work_us);
    seq_mean(&sps)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_bench_env_work_us_zero_is_fast() {
        let mut env = ConstBenchEnv {
            obs_dim: 4,
            work_us: 0,
        };
        let action = Action::zeros(1);
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = env.step(&action);
        }
        // 100 steps × ~50ns ≈ 5µs. Allow 5ms for CI noise.
        assert!(start.elapsed() < std::time::Duration::from_millis(5));
    }

    #[test]
    fn const_bench_env_work_us_busy_loops_for_at_least_target() {
        let mut env = ConstBenchEnv {
            obs_dim: 4,
            work_us: 500,
        };
        let action = Action::zeros(1);
        let start = std::time::Instant::now();
        let _ = env.step(&action);
        // 500µs target. Allow some slop downward (timer granularity on
        // Windows) but require >= 250µs to confirm the loop ran.
        assert!(
            start.elapsed() >= std::time::Duration::from_micros(250),
            "busy-loop returned too fast: {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn bench_vec_cell_produces_positive_throughput() {
        let sps = bench_vec_cell(2, 1, 100, false, 0);
        assert!(sps > 0.0, "expected positive sps, got {sps}");
    }
}
