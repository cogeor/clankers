//! `clankers-app replay` — replay an MCAP recording.
//!
//! Reads `/joint_states`, `/actions`, `/reward` messages from an MCAP
//! file via the upstream `mcap` crate directly and emits one JSONL line
//! per logical step on stdout.
//!
//! # Why an inline reader instead of `clankers_record::Reader`?
//!
//! `clankers_record::Reader` does not yet exist — `clankers-record/src/lib.rs`
//! ships only `plugin`, `recorder`, and `types`. The canonical reader is
//! W6 scope. To keep the JSONL output contract locked in this loop (so
//! `cli_replay.rs::replay_iterates_recorded_episode_steps` can assert on
//! it), the reader is inlined here using `mcap::MessageStream::new(&[u8])`
//! and `serde_json::from_slice` over the public frame types from
//! `clankers_record::types`. W6 will lift this reader into
//! `clankers_record::Reader`; the JSONL contract stays byte-equal so the
//! switch is mechanical.
//!
//! # Step grouping
//!
//! The recorder emits messages per simulation frame in the order
//! `joints → actions → reward → body_poses` (see
//! `recorder.rs::setup_channels` and the `.chain()` ordering in
//! `plugin.rs::build`). Per recorded step we expect at most one of each.
//! The reader treats every `/joint_states` message as the start of a new
//! step record; subsequent `/actions` / `/reward` messages accumulate
//! into the current record; the next `/joint_states` (or EOF) flushes
//! the previous one. Action / reward channels are optional — defaults
//! are empty `Vec<f32>` and `0.0`.
//!
//! # Forward-compat exit-2 flags
//!
//! - `--policy <path>`: ONNX runtime is W7 scope; not wired into
//!   `clankers-app` yet.
//! - `--export <dir>`: image decode requires GPU + the recorder's
//!   `camera` feature; W8 scope.
//! - `--viz`: requires GPU; off-limits in this loop.
//!
//! Each prints a one-line stderr message and exits with code `2`.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clankers_record::types::{ActionFrame, JointFrame, RewardFrame};
use clap::Args;
use serde::Serialize;

/// CLI flags for `clankers-app replay`.
#[derive(Args, Debug)]
pub struct ReplayArgs {
    /// Input MCAP file.
    #[arg(long)]
    pub input: PathBuf,

    /// Camera label to emit alongside step records. Currently exit-2
    /// (W8).
    #[arg(long)]
    pub camera: Option<String>,

    /// ONNX policy path; compute per-step L2 between policy action and
    /// recorded action. Currently exit-2 (W7).
    #[arg(long)]
    pub policy: Option<PathBuf>,

    /// Export PNG frames to a directory. Currently exit-2 (W8).
    #[arg(long)]
    pub export: Option<PathBuf>,

    /// Interactive 3D visualisation. Currently exit-2 (GPU off-limits).
    #[arg(long)]
    pub viz: bool,

    /// Accepted as a no-op alias — JSONL is the only supported output
    /// for `replay` in this loop.
    #[arg(long)]
    pub json: bool,
}

/// One JSONL output row per recorded simulation step.
#[derive(Serialize)]
struct StepRecord {
    step: u32,
    timestamp_ns: u64,
    joint_names: Vec<String>,
    joint_positions: Vec<f32>,
    joint_velocities: Vec<f32>,
    joint_torques: Vec<f32>,
    action: Vec<f32>,
    reward: f32,
}

impl StepRecord {
    fn from_joint_frame(step: u32, frame: JointFrame) -> Self {
        Self {
            step,
            timestamp_ns: frame.timestamp_ns,
            joint_names: frame.names,
            joint_positions: frame.positions,
            joint_velocities: frame.velocities,
            joint_torques: frame.torques,
            action: Vec::new(),
            reward: 0.0,
        }
    }
}

/// Execute `clankers-app replay`.
pub fn execute(args: &ReplayArgs) -> ExitCode {
    // ---- forward-compat exit-2 flags ---------------------------------
    if args.policy.is_some() {
        eprintln!("replay --policy is W7 scope (ONNX runtime not wired into clankers-app yet)");
        return ExitCode::from(2);
    }
    if args.export.is_some() {
        eprintln!(
            "replay --export is W8 scope (image decode requires GPU + the `camera` cargo feature)"
        );
        return ExitCode::from(2);
    }
    if args.viz {
        eprintln!("replay --viz requires GPU; not available in this loop");
        return ExitCode::from(2);
    }
    if args.camera.is_some() {
        eprintln!("warning: --camera ignored without --viz / --export (W8)");
    }
    // `--json` is the default; accept silently.

    // ---- read MCAP --------------------------------------------------
    let data = match fs::read(&args.input) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to read MCAP file '{}': {e}", args.input.display());
            return ExitCode::from(1);
        }
    };

    let stream = match mcap::MessageStream::new(&data) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to parse MCAP file '{}': {e}", args.input.display());
            return ExitCode::from(1);
        }
    };

    // ---- step grouping state machine --------------------------------
    let mut current: Option<StepRecord> = None;
    let mut step_index: u32 = 0;

    for msg_result in stream {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                eprintln!("warning: skipping malformed MCAP message: {e}");
                continue;
            }
        };

        match msg.channel.topic.as_str() {
            "/joint_states" => {
                // Flush the previous record, if any, then open a new one.
                if let Some(prev) = current.take() {
                    emit_jsonl(&prev);
                }
                match serde_json::from_slice::<JointFrame>(&msg.data) {
                    Ok(frame) => {
                        current = Some(StepRecord::from_joint_frame(step_index, frame));
                        step_index += 1;
                    }
                    Err(e) => eprintln!("warning: malformed JointFrame: {e}"),
                }
            }
            "/actions" => match serde_json::from_slice::<ActionFrame>(&msg.data) {
                Ok(frame) => {
                    if let Some(rec) = current.as_mut() {
                        rec.action = frame.data;
                    } else {
                        eprintln!(
                            "warning: /actions message arrived before any /joint_states; dropping"
                        );
                    }
                }
                Err(e) => eprintln!("warning: malformed ActionFrame: {e}"),
            },
            "/reward" => match serde_json::from_slice::<RewardFrame>(&msg.data) {
                Ok(frame) => {
                    if let Some(rec) = current.as_mut() {
                        rec.reward = frame.reward;
                    } else {
                        eprintln!(
                            "warning: /reward message arrived before any /joint_states; dropping"
                        );
                    }
                }
                Err(e) => eprintln!("warning: malformed RewardFrame: {e}"),
            },
            // Other channels (body_poses, camera/*) are silently ignored
            // in this loop; W6 will surface them.
            _ => {}
        }
    }

    // Flush trailing record.
    if let Some(last) = current.take() {
        emit_jsonl(&last);
    }

    ExitCode::SUCCESS
}

fn emit_jsonl(rec: &StepRecord) {
    match serde_json::to_string(rec) {
        Ok(line) => println!("{line}"),
        Err(e) => eprintln!("warning: failed to serialise step record: {e}"),
    }
}
