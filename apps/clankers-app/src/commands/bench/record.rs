//! `bench record` body — MCAP recorder write rate, sync vs async.
//!
//! The first cell is always sync mode; remaining cells are async with
//! one cell per buffer capacity in `--buffers`. Each cell drives the
//! recorder hot path directly (bypassing the per-system writers) so the
//! measurement isn't polluted by Bevy scheduler noise.

use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bevy::MinimalPlugins;
use bevy::prelude::*;
use clankers_core::time::SimTime;
use clankers_record::prelude::*;

use super::args::{BenchArgs, RecordBenchArgs};
use super::csv::{print_human_v2, write_csv_row_v2};
use super::stats::aggregate_v2;

fn parse_buffer_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|t| {
            t.trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid buffer capacity '{t}': {e}"))
        })
        .collect()
}

pub(super) fn execute(args: &BenchArgs, r_args: &RecordBenchArgs) -> ExitCode {
    let buffers = match parse_buffer_list(&r_args.buffers) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bench record: {e}");
            return ExitCode::from(1);
        }
    };

    // Cell 1: sync mode.
    if let Err(code) = run_record_cell(args, r_args, None) {
        return code;
    }

    // Cells 2..N: async with each requested capacity.
    for &cap in &buffers {
        if let Err(code) = run_record_cell(args, r_args, Some(cap)) {
            return code;
        }
    }

    ExitCode::SUCCESS
}

fn run_record_cell(
    args: &BenchArgs,
    r_args: &RecordBenchArgs,
    async_capacity: Option<usize>,
) -> Result<(), ExitCode> {
    let mode_label = async_capacity.map_or_else(|| "sync".to_string(), |c| format!("async@{c}"));

    let mut per_run_wall_ms = Vec::with_capacity(args.runs as usize);
    let mut per_run_sps = Vec::with_capacity(args.runs as usize);
    let mut step_durs: Vec<Duration> = Vec::new();
    let mut last_dropped: u64 = 0;

    // Warmup
    for _ in 0..args.warmup_runs {
        let _ = drive_record_run(r_args, async_capacity);
    }

    for _ in 0..args.runs {
        let (wall, per_frame_durs, dropped) = drive_record_run(r_args, async_capacity);
        last_dropped = dropped;
        let wall_ms = wall.as_secs_f64() * 1000.0;
        per_run_wall_ms.push(wall_ms);
        let sps = if wall_ms > 0.0 {
            f64::from(r_args.frames) / wall_ms * 1000.0
        } else {
            0.0
        };
        per_run_sps.push(sps);
        step_durs.extend(per_frame_durs);
    }

    let total_steps = args.runs * r_args.frames;
    let row = aggregate_v2(
        &format!("record_{mode_label}"),
        args,
        total_steps,
        &per_run_wall_ms,
        &per_run_sps,
        &mut step_durs,
        0,
        last_dropped,
        0.0,
        &format!(
            "kind=record;mode={mode_label};joints={};frames={}",
            r_args.joints, r_args.frames
        ),
    );

    if let Some(path) = args.csv.as_ref()
        && let Err(e) = write_csv_row_v2(path, &row)
    {
        eprintln!("bench record: failed to write CSV: {e}");
        return Err(ExitCode::from(1));
    }
    if args.json {
        if let Ok(s) = serde_json::to_string(&row) {
            println!("{s}");
        }
    } else if args.csv.is_none() {
        print_human_v2(&row);
    }
    Ok(())
}

fn drive_record_run(
    r_args: &RecordBenchArgs,
    async_capacity: Option<usize>,
) -> (Duration, Vec<Duration>, u64) {
    // Synthetic frame payload — 8 joints by default. We bypass the
    // Bevy app and exercise the recorder hot path directly to keep
    // the measurement deterministic and free of scheduler noise.
    let tmp = std::env::temp_dir().join(format!(
        "clankers_bench_record_{}_{}.mcap",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    let cfg = RecordingConfig {
        output_path: tmp.clone(),
        record_joints: true,
        record_actions: false,
        record_rewards: false,
        record_body_poses: false,
        async_mode: async_capacity.is_some(),
        async_buffer_capacity: async_capacity.unwrap_or(256),
    };

    // Spin up a minimal Bevy app to drive setup_channels (which
    // registers the manifest + joint_states channels and installs
    // async if requested). After the first update the channels are
    // live; we then bypass the per-system writers and call
    // `Recorder::write_joint_frame` directly per frame.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(SimTime::new());
    app.insert_resource(cfg);
    app.add_plugins(RecorderPlugin);
    app.finish();
    app.cleanup();
    app.update();

    let channel_id = app
        .world()
        .resource::<clankers_record::recorder::ChannelIds>()
        .joints
        .expect("/joint_states channel registered");

    let names: Vec<String> = (0..r_args.joints).map(|i| format!("j{i}")).collect();
    let positions = vec![0.0_f32; r_args.joints];
    let velocities = vec![0.0_f32; r_args.joints];
    let torques = vec![0.0_f32; r_args.joints];
    let frames = r_args.frames;

    let wall_start = Instant::now();
    let mut per_frame_durs: Vec<Duration> = Vec::with_capacity(frames as usize);

    {
        // NonSend: we have to scope our access so we drop the borrow
        // before reading `DroppedFrames`.
        let world = app.world_mut();
        for i in 0..frames {
            let frame = JointFrame {
                timestamp_ns: u64::from(i) * 1_000_000,
                names: names.clone(),
                positions: positions.clone(),
                velocities: velocities.clone(),
                torques: torques.clone(),
            };
            let s = Instant::now();
            if let Some(mut rec) =
                world.get_non_send_resource_mut::<clankers_record::recorder::Recorder>()
            {
                let _ = rec.write_joint_frame(channel_id, &frame);
            }
            per_frame_durs.push(s.elapsed());
        }
    }
    let wall = wall_start.elapsed();

    let dropped = app.world().resource::<DroppedFrames>().get();

    // Drop the app (and the recorder) so the MCAP file finalises and
    // the worker thread joins before we measure.
    drop(app);
    // Best-effort cleanup; ignore failure (Windows may still hold a
    // handle briefly).
    let _ = std::fs::remove_file(&tmp);

    (wall, per_frame_durs, dropped)
}
