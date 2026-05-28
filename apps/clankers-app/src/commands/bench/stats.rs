//! Percentile / stddev / aggregation helpers shared across V1 and V2
//! benchmark bodies.
//!
//! - [`percentile_us`] — q-th percentile of a slice of `Duration`s,
//!   returned in microseconds.
//! - [`stddev`] — Bessel-corrected sample standard deviation.
//! - [`seq_mean`] / [`seq_steps_per_sec_mean`] — plain arithmetic mean.
//! - [`aggregate_v1`] / [`aggregate_v2`] — assemble a `BenchRow` /
//!   `BenchRowV2` from per-run wall-time + step-duration samples.
//! - [`build_notes`] — host/commit/timestamp/profile string baked into
//!   every CSV row's `notes` column.
//! - [`parse_env_list`] — comma-separated `u16` parser used by
//!   `bench vec` and `bench protocol`.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::args::BenchArgs;
use super::csv::{BenchRow, BenchRowV2};

pub(super) fn percentile_us(samples: &mut [Duration], q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_unstable();
    let n = samples.len();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let mut idx = (n as f64 * q).floor() as usize;
    if idx >= n {
        idx = n - 1;
    }
    samples[idx].as_secs_f64() * 1_000_000.0
}

pub(super) fn stddev(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let sq_sum: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum();
    (sq_sum / (n - 1.0)).sqrt()
}

pub(super) fn seq_mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let n = samples.len() as f64;
        samples.iter().sum::<f64>() / n
    }
}

pub(super) fn seq_steps_per_sec_mean(samples: &[f64]) -> f64 {
    seq_mean(samples)
}

pub(super) fn build_notes(args: &BenchArgs) -> String {
    let host = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".to_owned());
    let commit = std::env::var("GITHUB_SHA")
        .or_else(|_| std::env::var("GIT_COMMIT"))
        .unwrap_or_else(|_| "local".to_owned());
    let unix_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release-like"
    };
    format!(
        "hostname={host};commit={commit};unix_ts={unix_ts};profile={profile};warmup_runs={}",
        args.warmup_runs
    )
}

pub(super) fn parse_env_list(s: &str) -> Result<Vec<u16>, String> {
    s.split(',')
        .map(|t| {
            t.trim()
                .parse::<u16>()
                .map_err(|e| format!("invalid env count '{t}': {e}"))
        })
        .collect()
}

pub(super) fn aggregate_v1(
    scenario_name: &str,
    args: &BenchArgs,
    total_steps: u32,
    per_run_wall_ms: &[f64],
    per_run_steps_per_sec: &[f64],
    all_step_durations: &mut [Duration],
) -> BenchRow {
    let wall_ms_total: f64 = per_run_wall_ms.iter().sum();
    let wall_ms_per_run_mean = if args.runs > 0 {
        wall_ms_total / f64::from(args.runs)
    } else {
        0.0
    };
    let wall_ms_per_run_std = stddev(per_run_wall_ms);
    #[allow(clippy::cast_precision_loss)]
    let steps_per_sec_mean = if per_run_steps_per_sec.is_empty() {
        0.0
    } else {
        per_run_steps_per_sec.iter().sum::<f64>() / per_run_steps_per_sec.len() as f64
    };
    let step_us_p50 = percentile_us(&mut all_step_durations.to_vec(), 0.50);
    let step_us_p99 = percentile_us(all_step_durations, 0.99);

    BenchRow {
        scenario: scenario_name.to_owned(),
        steps: total_steps,
        runs: args.runs,
        max_steps: args.max_steps,
        wall_ms_total,
        wall_ms_per_run_mean,
        wall_ms_per_run_std,
        steps_per_sec_mean,
        step_us_p50,
        step_us_p99,
        notes: build_notes(args),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn aggregate_v2(
    scenario_name: &str,
    args: &BenchArgs,
    total_steps: u32,
    per_run_wall_ms: &[f64],
    per_run_sps: &[f64],
    step_durs: &mut [Duration],
    num_envs: u32,
    dropped_frames: u64,
    throughput_x: f64,
    notes_suffix: &str,
) -> BenchRowV2 {
    let wall_ms_total: f64 = per_run_wall_ms.iter().sum();
    let wall_ms_per_run_mean = if args.runs > 0 {
        wall_ms_total / f64::from(args.runs)
    } else {
        0.0
    };
    let wall_ms_per_run_std = stddev(per_run_wall_ms);
    let steps_per_sec_mean = seq_mean(per_run_sps);
    let step_us_p50 = percentile_us(&mut step_durs.to_vec(), 0.50);
    let step_us_p95 = percentile_us(&mut step_durs.to_vec(), 0.95);
    let step_us_p99 = percentile_us(step_durs, 0.99);

    let base_notes = build_notes(args);
    let notes = if notes_suffix.is_empty() {
        base_notes
    } else {
        format!("{base_notes};{notes_suffix}")
    };

    BenchRowV2 {
        scenario: scenario_name.to_owned(),
        steps: total_steps,
        runs: args.runs,
        max_steps: args.max_steps,
        wall_ms_total,
        wall_ms_per_run_mean,
        wall_ms_per_run_std,
        steps_per_sec_mean,
        step_us_p50,
        step_us_p99,
        notes,
        num_envs,
        p95_us: step_us_p95,
        dropped_frames,
        throughput_x,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_us_returns_zero_for_empty() {
        let mut samples: Vec<Duration> = Vec::new();
        assert!(percentile_us(&mut samples, 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn percentile_us_median_of_known_samples() {
        let mut samples: Vec<Duration> = (1_u64..=100).map(Duration::from_micros).collect();
        assert!((percentile_us(&mut samples, 0.5) - 51.0).abs() < 0.01);
        let mut samples2: Vec<Duration> = (1_u64..=100).map(Duration::from_micros).collect();
        assert!((percentile_us(&mut samples2, 0.99) - 100.0).abs() < 0.01);
    }

    #[test]
    fn stddev_zero_for_one_sample() {
        assert!(stddev(&[42.0]).abs() < f64::EPSILON);
    }

    #[test]
    fn stddev_bessel_corrected() {
        let xs = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = stddev(&xs);
        assert!((s - 2.138_089_935_2).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn parse_env_list_basic() {
        assert_eq!(parse_env_list("1,2,4,8").unwrap(), vec![1, 2, 4, 8]);
        assert_eq!(parse_env_list(" 16 ").unwrap(), vec![16]);
        assert!(parse_env_list("1,abc,2").is_err());
    }
}
