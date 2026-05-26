//! `clankers-app record` — capture an episode to MCAP.
//!
//! Builds a scenario via `ScenarioRegistry`, attaches
//! `clankers_record::RecorderPlugin` to the already-built `App`, and runs
//! the headless episode loop for `--max-steps` ticks. The `Recorder::Drop`
//! impl finalises the MCAP footer on app teardown.
//!
//! # `PendingAction` / `PendingReward` are unpopulated this loop
//!
//! Neither `arm_pick` nor `cartpole` writes into `PendingAction` or
//! `PendingReward`. The `/actions` and `/reward` channels therefore record
//! the default values (empty `Vec<f32>`, `0.0`) once per `PostUpdate` frame.
//! Real policy-driven action capture lands in W7. `/joint_states` is fully
//! populated by `record_joint_states_system` reading the `JointState`
//! components the scenario spawned.
//!
//! # Forward-compat flags (soft no-ops)
//!
//! - `--camera <label>`: the `clankers-record` `camera` feature requires
//!   `clankers-render` + GPU, which is off-limits in this loop. Stderr
//!   warning, flag dropped. Wired in W8.
//! - `--frame-decimation N` with `N > 1`: the current recorder has no
//!   decimation knob. Stderr warning, every frame recorded. W6 follow-up.
//! - `--metadata KEY=VAL`: no MCAP metadata API exposed by the recorder
//!   today. Stderr warning, dropped. W6 follow-up.
//!
//! # `Dropped frames` is always `0`
//!
//! The recorder writes synchronously through a `BufWriter<File>` — there
//! is no bounded channel and no drop accounting. Absent disk-full / OS
//! write errors, no frames can be dropped, so the printed counter is an
//! honest zero. Async recorder + real drop counter ships in W7 PR4 / W6.

use std::path::PathBuf;
use std::process::ExitCode;

use bevy::prelude::*;
use clankers_env::prelude::*;
use clankers_record::prelude::{RecorderPlugin, RecordingConfig};
use clankers_sim::scenarios::register_builtin;
use clankers_sim::{ClankersSimPlugin, ScenarioConfig, ScenarioRegistry};
use clap::Args;

/// CLI flags for `clankers-app record`.
#[derive(Args, Debug)]
pub struct RecordArgs {
    /// Built-in scenario name (`cartpole`, `arm_pick`).
    #[arg(long)]
    pub scenario: String,

    /// Output MCAP path.
    #[arg(long)]
    pub output: PathBuf,

    /// Maximum simulation steps to record.
    #[arg(long, default_value_t = 100)]
    pub max_steps: u32,

    /// Random seed forwarded to `Episode::reset`.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Comma-separated topic list. Recognised names: `joint_states`,
    /// `actions`, `reward`, `body_poses`. Unknown names emit a stderr
    /// warning and fall back to the default toggles.
    #[arg(long, value_delimiter = ',')]
    pub topics: Vec<String>,

    /// Camera label. Accepted for forward-compat; soft no-op this loop
    /// (GPU + `clankers-record/camera` feature off-limits).
    #[arg(long)]
    pub camera: Option<String>,

    /// Per-frame decimation. Values > 1 are accepted but ignored this
    /// loop; W6 will wire it.
    #[arg(long)]
    pub frame_decimation: Option<u32>,

    /// `KEY=VAL` metadata pairs. Accepted but dropped this loop; the
    /// recorder has no metadata API today.
    #[arg(long, value_delimiter = ',')]
    pub metadata: Vec<String>,
}

/// Translate `--topics` into a [`RecordingConfig`] toggle set. Missing
/// flag → defaults (`joints + actions + reward`, `body_poses` off).
fn build_recording_config(args: &RecordArgs) -> RecordingConfig {
    let mut cfg = RecordingConfig {
        output_path: args.output.clone(),
        ..RecordingConfig::default()
    };

    if args.topics.is_empty() {
        return cfg;
    }

    // User supplied an explicit list — reset everything to false then
    // re-enable the requested topics.
    cfg.record_joints = false;
    cfg.record_actions = false;
    cfg.record_rewards = false;
    cfg.record_body_poses = false;

    for topic in &args.topics {
        match topic.trim() {
            "joint_states" | "joints" => cfg.record_joints = true,
            "actions" | "action" => cfg.record_actions = true,
            "reward" | "rewards" => cfg.record_rewards = true,
            "body_poses" | "poses" => cfg.record_body_poses = true,
            other => {
                eprintln!(
                    "warning: unknown topic '{other}'; recognised: joint_states, actions, reward, body_poses"
                );
            }
        }
    }

    // If the user typed nonsense everywhere, fall back to defaults so the
    // file is non-empty.
    if !(cfg.record_joints || cfg.record_actions || cfg.record_rewards || cfg.record_body_poses) {
        eprintln!("warning: no recognised topics in --topics; falling back to defaults");
        cfg.record_joints = true;
        cfg.record_actions = true;
        cfg.record_rewards = true;
    }

    cfg
}

/// Execute `clankers-app record`.
pub fn execute(args: &RecordArgs) -> ExitCode {
    // ---- soft no-op forward-compat flags ------------------------------
    if args.camera.is_some() {
        eprintln!(
            "warning: camera recording requires GPU + the `camera` cargo feature of clankers-record; ignoring --camera"
        );
    }
    if let Some(n) = args.frame_decimation
        && n > 1
    {
        eprintln!("warning: frame decimation not yet implemented; recording every frame");
    }
    if !args.metadata.is_empty() {
        eprintln!("warning: --metadata not yet plumbed through; ignoring");
    }

    // ---- registry lookup ---------------------------------------------
    let mut registry = ScenarioRegistry::new();
    register_builtin(&mut registry);
    let Some(builder) = registry.get(&args.scenario) else {
        eprintln!("unknown scenario: {}", args.scenario);
        return ExitCode::from(1);
    };

    // ---- scenario config (record_path populated for the first time) --
    let cfg = ScenarioConfig {
        seed: args.seed,
        max_steps: args.max_steps,
        headless: true,
        record_path: Some(args.output.clone()),
    };

    // ---- build the App + scenario, then bolt RecorderPlugin on top ---
    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);
    let handle = builder.build(&mut app, &cfg);

    // The URDF spawner inserts `JointName(String)` but not `bevy::prelude::Name`,
    // and `record_joint_states_system` queries `&Name`. Walk the layout
    // returned by the scenario and attach `Name` components so the
    // recorder sees one row per joint per `PostUpdate` frame. Mirrors the
    // pattern in `examples/src/bin/arm_pick_record.rs` (lines 363–374).
    if let Some(layout) = handle.layout.as_ref() {
        let world = app.world_mut();
        for spec in layout.joints() {
            if let Some(entity) = spec.entity {
                world
                    .entity_mut(entity)
                    .insert(Name::new(spec.name.clone()));
            }
        }
    }

    let record_cfg = build_recording_config(args);
    app.insert_resource(record_cfg);
    app.add_plugins(RecorderPlugin);

    app.finish();
    app.cleanup();

    app.world_mut()
        .resource_mut::<EpisodeConfig>()
        .max_episode_steps = handle.max_steps;

    app.world_mut().resource_mut::<Episode>().reset(args.seed);

    // ---- headless episode loop ---------------------------------------
    for _ in 0..handle.max_steps {
        app.update();
        if app.world().resource::<Episode>().is_done() {
            break;
        }
    }

    // Drop `app` → `Recorder::Drop` finalises the MCAP footer.
    drop(app);

    // Synchronous BufWriter cannot drop frames absent disk-full / OS
    // write errors. Print 0 to keep the contract stable for W6/W7.
    eprintln!("Dropped frames: 0");

    ExitCode::SUCCESS
}
