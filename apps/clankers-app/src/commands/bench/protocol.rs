//! `bench protocol` body — binary frame vs JSON encoding throughput.
//!
//! For each requested env count, generates a synthetic obs batch and
//! times `binary_frame::encode_batch_f32` against `serde_json::to_vec`.
//! Emits one V2 CSV row per env count with `throughput_x` set to the
//! binary/JSON steps-per-sec ratio.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use clankers_gym::binary_frame;

use super::args::{BenchArgs, ProtocolArgs};
use super::csv::{print_human_v2, write_csv_row_v2};
use super::stats::{aggregate_v2, parse_env_list, seq_mean};

pub(super) fn execute(args: &BenchArgs, p_args: &ProtocolArgs) -> ExitCode {
    let env_counts = match parse_env_list(&p_args.envs) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bench protocol: {e}");
            return ExitCode::from(1);
        }
    };

    for &n in &env_counts {
        let total_floats = usize::from(n) * p_args.obs_dim as usize;
        // Synthetic batch payload (deterministic via the splitmix
        // hash of (seed, index)).
        let seed = args.seed.unwrap_or(0xDEAD_BEEF);
        let data: Vec<f32> = (0..total_floats)
            .map(|i| {
                let h = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(i as u64);
                f32::from_bits((h & 0x7FFF_FFFF) as u32 | 0x3F00_0000) - 1.0
            })
            .collect();

        // Warmup
        for _ in 0..args.warmup_runs {
            let _ = binary_frame::encode_batch_f32(u32::from(n), p_args.obs_dim, &data);
            let _ = serde_json::to_vec(&data).expect("serialise");
        }

        let mut per_run_wall_ms = Vec::with_capacity(args.runs as usize);
        let mut per_run_bin_sps = Vec::with_capacity(args.runs as usize);
        let mut per_run_json_sps = Vec::with_capacity(args.runs as usize);
        let mut step_durs: Vec<Duration> = Vec::new();

        for _ in 0..args.runs {
            // Binary
            let bin_start = Instant::now();
            for _ in 0..p_args.batches {
                let s = Instant::now();
                let _ = binary_frame::encode_batch_f32(u32::from(n), p_args.obs_dim, &data);
                step_durs.push(s.elapsed());
            }
            let bin_wall = bin_start.elapsed();

            // JSON
            let json_start = Instant::now();
            for _ in 0..p_args.batches {
                let _ = serde_json::to_vec(&data).expect("serialise");
            }
            let json_wall = json_start.elapsed();

            let bin_wall_ms = bin_wall.as_secs_f64() * 1000.0;
            let json_wall_ms = json_wall.as_secs_f64() * 1000.0;
            per_run_wall_ms.push(bin_wall_ms + json_wall_ms);
            per_run_bin_sps.push(if bin_wall_ms > 0.0 {
                f64::from(p_args.batches) / bin_wall_ms * 1000.0
            } else {
                0.0
            });
            per_run_json_sps.push(if json_wall_ms > 0.0 {
                f64::from(p_args.batches) / json_wall_ms * 1000.0
            } else {
                0.0
            });
        }

        let bin_sps = seq_mean(&per_run_bin_sps);
        let json_sps = seq_mean(&per_run_json_sps);
        let throughput_ratio = if json_sps > 0.0 {
            bin_sps / json_sps
        } else {
            0.0
        };
        let total_steps = args.runs * p_args.batches;

        let row = aggregate_v2(
            "protocol_binary_vs_json",
            args,
            total_steps,
            &per_run_wall_ms,
            &per_run_bin_sps,
            &mut step_durs,
            u32::from(n),
            0,
            throughput_ratio,
            &format!(
                "kind=protocol;obs_dim={};binary_sps={:.0};json_sps={:.0}",
                p_args.obs_dim, bin_sps, json_sps
            ),
        );

        if let Some(path) = args.csv.as_ref()
            && let Err(e) = write_csv_row_v2(path, &row)
        {
            eprintln!("bench protocol: failed to write CSV: {e}");
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

    ExitCode::SUCCESS
}
