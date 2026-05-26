//! `clankers-app serve` — TCP gym server.
//!
//! W5 PR2 — scenarios via `--scenario <name>`. Without `--scenario`,
//! the legacy synthetic-N-joint body in [`execute_default`] runs
//! unchanged (back-compat).
//!
//! # Protocol negotiation
//!
//! `--protocol json|binary` is a **hint**, not new encoding code. The
//! server in `clankers-gym` already routes every observation through
//! [`clankers_gym::encoding::encode_observation`]; the
//! `binary_obs` capability is unconditionally advertised by
//! [`clankers_gym::server::ServerConfig::default`]. The client opts
//! into binary on the `Init` handshake. The CLI flag exists so the
//! listening banner can report the operator's intent and so smoke
//! tests can grep for the mode.
//!
//! # `MotorOverrides` applicator gap (`arm_pick` + serve)
//!
//! Per loop 06 PLAN Design choice D, the generic
//! [`JointCommandApplicator`] writes `JointCommand`, not
//! `MotorOverrides`. For `cartpole` this works natively. For
//! `arm_pick` the proper applicator (per-joint `MotorOverrides` keyed
//! by entity) is W7 follow-up. The W5 PR2 `serve` integration test
//! exercises only `cartpole`.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::prelude::*;
use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
use clankers_core::layout::{
    JointKind, JointLayout, JointLayoutBuilder, JointSpec, JointSpecLimits,
};
use clankers_core::prelude::*;
use clankers_env::prelude::*;
use clankers_sim::scenarios::register_builtin;
use clankers_sim::{ClankersSimPlugin, ScenarioConfig, ScenarioRegistry};
use clap::{Args, ValueEnum};

// ---------------------------------------------------------------------------
// Clap types
// ---------------------------------------------------------------------------

/// Wire protocol selection for `serve --protocol`.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Protocol {
    /// Standard JSON-only framing.
    Json,
    /// JSON header + raw binary observation frames (negotiated via
    /// the `binary_obs` capability at Init time).
    Binary,
}

/// Observation packaging mode for `serve --obs`. Forward-compat for
/// W7 image envs; PR2 always uses `flat`.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum ObsMode {
    /// Flat float vector (cartpole, `arm_pick`).
    Flat,
    /// Image (W7).
    Image,
    /// Dict (W7).
    Dict,
}

/// CLI flags for `clankers-app serve`.
#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Built-in scenario name. Without this flag the legacy
    /// synthetic-N-joint server path runs via [`execute_default`].
    #[arg(long)]
    pub scenario: Option<String>,

    /// Address to bind (e.g. `127.0.0.1:9876`). Pass `127.0.0.1:0` for
    /// an OS-assigned ephemeral port.
    #[arg(short, long, default_value = "127.0.0.1:9876")]
    pub address: String,

    /// Number of parallel envs (PR2 always 1; >1 is W7).
    #[arg(long, default_value_t = 1)]
    pub num_envs: u32,

    /// Wire protocol.
    #[arg(long, default_value = "json", value_enum)]
    pub protocol: Protocol,

    /// Path to an MCAP recording sink (W5 PR3 — same fallback).
    #[arg(long)]
    pub record: Option<PathBuf>,

    /// Random seed (forward-compat — accepted but unused in PR2).
    #[arg(long)]
    pub seed: Option<u64>,

    /// Maximum steps per episode.
    #[arg(long, default_value_t = 1000)]
    pub max_steps: u32,

    /// Observation packaging mode (PR2 always `flat`).
    #[arg(long, default_value = "flat", value_enum)]
    pub obs: ObsMode,
}

// ---------------------------------------------------------------------------
// execute
// ---------------------------------------------------------------------------

/// Scenario-driven gym server (`clankers-app serve --scenario
/// <name>`). When `--scenario` is `None`, delegates to
/// [`execute_default`].
pub fn execute(args: &ServeArgs) -> ExitCode {
    use clankers_gym::prelude::*;

    let Some(name) = args.scenario.as_deref() else {
        // Back-compat: legacy synthetic-N-joint demo path. `num_envs`
        // defaults to 1 and is interpreted as `joints` for the legacy
        // path — a happy coincidence (the legacy `--joints` flag is no
        // longer surfaced, but the synthetic layout still spawns 1
        // joint by default).
        execute_default(&args.address, args.num_envs as usize, args.max_steps);
        return ExitCode::SUCCESS;
    };

    if args.record.is_some() {
        eprintln!("--record not yet supported (W5 PR3)");
        return ExitCode::from(2);
    }
    if args.num_envs > 1 {
        eprintln!("--num-envs > 1 not yet supported (W7)");
        return ExitCode::from(2);
    }
    if args.obs != ObsMode::Flat {
        eprintln!("--obs {:?} not yet supported (W7)", args.obs);
        return ExitCode::from(2);
    }
    let _ = args.seed; // accepted; PR2 unused

    let mut registry = ScenarioRegistry::new();
    register_builtin(&mut registry);
    let Some(builder) = registry.get(name) else {
        eprintln!("unknown scenario: {name}");
        return ExitCode::from(1);
    };

    let cfg = ScenarioConfig {
        seed: args.seed,
        max_steps: args.max_steps,
        headless: true,
        record_path: None,
    };

    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);
    let handle = builder.build(&mut app, &cfg);
    app.finish();
    app.cleanup();
    app.world_mut()
        .resource_mut::<EpisodeConfig>()
        .max_episode_steps = handle.max_steps;

    let layout = handle
        .layout
        .clone()
        .expect("scenario must expose a JointLayout");
    let obs_dim = layout.len() * 2; // position + velocity
    let act_dim = layout.len();
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; act_dim],
        high: vec![1.0; act_dim],
    };

    let mut env = GymEnv::new(
        app,
        obs_space,
        act_space,
        Box::new(JointCommandApplicator { layout }),
    );

    let server = GymServer::bind(&args.address).expect("failed to bind gym server");
    let addr = server.local_addr().expect("failed to read local addr");
    println!(
        "clankers-app serve listening on {addr} (scenario={name}, protocol={:?}, obs_dim={obs_dim}, act_dim={act_dim}, max_steps={})",
        args.protocol, handle.max_steps,
    );

    // SIGINT handler — flip the atomic flag between accept()s so the
    // server loop exits cleanly after the current client disconnects.
    // Note: a blocking `accept()` does not interleave with the flag
    // check, so Ctrl+C while waiting for a client may still terminate
    // the process (acceptable for PR2 — see PLAN Constraints item 6).
    let running = Arc::new(AtomicBool::new(true));
    let r2 = running.clone();
    let _ = ctrlc::set_handler(move || r2.store(false, Ordering::SeqCst));

    while running.load(Ordering::SeqCst) {
        match server.serve_one(&mut env) {
            Ok(()) => {}
            Err(e) => eprintln!("client error: {e}"),
        }
    }
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// execute_default — legacy synthetic body, preserved verbatim
// ---------------------------------------------------------------------------

/// Legacy `Serve` mode body — synthetic-layout N-joint demo.
/// Preserved verbatim from the pre-PR1 `apps/clankers-app/src/main.rs::run_serve`.
pub fn execute_default(address: &str, num_joints: usize, max_steps: u32) {
    use clankers_gym::prelude::*;

    // Build the Bevy app for the gym environment
    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);

    // Spawn N joint entities and collect their ids for the layout.
    let joint_entities: Vec<Entity> = (0..num_joints)
        .map(|_| {
            app.world_mut()
                .spawn((
                    Actuator::default(),
                    JointCommand::default(),
                    JointState::default(),
                    JointTorque::default(),
                ))
                .id()
        })
        .collect();

    // Build a synthetic layout — no URDF in this demo path.
    let layout = synthetic_layout(&joint_entities);

    // Register a sensor
    {
        let world = app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(layout.clone())), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    app.world_mut()
        .resource_mut::<EpisodeConfig>()
        .max_episode_steps = max_steps;

    let obs_dim = num_joints * 2; // position + velocity
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; num_joints],
        high: vec![1.0; num_joints],
    };

    let mut env = GymEnv::new(
        app,
        obs_space,
        act_space,
        Box::new(JointCommandApplicator { layout }),
    );

    let server = GymServer::bind(address).expect("failed to bind server");
    let addr = server.local_addr().expect("failed to get address");
    println!("clankers gym server listening on {addr}");
    println!("joints={num_joints}, obs_dim={obs_dim}, act_dim={num_joints}, max_steps={max_steps}");

    loop {
        println!("waiting for client...");
        match server.serve_one(&mut env) {
            Ok(()) => println!("client disconnected cleanly"),
            Err(e) => eprintln!("client error: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// JointCommandApplicator
// ---------------------------------------------------------------------------

/// Default action applicator that writes action values to joint commands
/// in layout slot order.
pub struct JointCommandApplicator {
    layout: Arc<JointLayout>,
}

impl ActionApplicator for JointCommandApplicator {
    fn apply(&self, world: &mut World, action: &Action) {
        let values = action
            .as_continuous()
            .expect("ActionApplicator contract: continuous action expected");
        for (i, entity) in self.layout.bound_entities().enumerate() {
            if i >= values.len() {
                break;
            }
            if let Some(mut cmd) = world.get_mut::<JointCommand>(entity) {
                cmd.value = values[i];
            }
        }
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "JointCommandApplicator"
    }

    fn layout(&self) -> &JointLayout {
        &self.layout
    }
}

/// Build a synthetic [`JointLayout`] of `n` revolute joints, bound to the
/// supplied entity ids in spawn order. Used by the CLI demo, which
/// spawns N raw joint entities without a URDF.
fn synthetic_layout(entities: &[Entity]) -> Arc<JointLayout> {
    let mut builder = JointLayoutBuilder::default();
    for (i, _) in entities.iter().enumerate() {
        builder = builder.push(JointSpec {
            name: format!("joint_{i}"),
            entity: None,
            joint_type: JointKind::Revolute,
            limits: JointSpecLimits::default(),
            axis: [0.0, 0.0, 1.0],
        });
    }
    let mut layout = builder.build();
    layout.bind_entities(entities);
    Arc::new(layout)
}
