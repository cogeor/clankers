//! `clankers-app validate` — read-only schema / motor-coverage checks.
//!
//! See PLAN.md Design choice C for the full feature matrix. In W5 PR1:
//!
//! - `--urdf <path>` — implemented end-to-end (parse + optional JSON).
//! - `--scenario <name>` — implemented as registry lookup; PR1 always
//!   returns `UnknownScenario` because `register_builtin` ships empty.
//! - `--strict` — without `--scenario`, emits a one-line stderr warning
//!   and exits 0 (no-op). Full strict-mode lands in PR2.
//! - `--scene`, `--policy`, `--recording-schema` — exit 2 with a
//!   "not yet implemented" message.
//! - `--seed` — accepted but ignored (forward-compat with `--scenario`).
//! - `--json` — toggles structured stdout for the `--urdf` and
//!   `--scenario` paths.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clankers_sim::{ScenarioError, ScenarioRegistry, scenarios::register_builtin};
use clap::Args;
use serde::Serialize;

/// CLI flags for `clankers-app validate`.
#[derive(Args, Debug)]
pub struct ValidateArgs {
    /// URDF file to parse.
    #[arg(long)]
    pub urdf: Option<PathBuf>,

    /// Scene config to load (W5 PR3+).
    #[arg(long)]
    pub scene: Option<PathBuf>,

    /// Policy file to validate (W7).
    #[arg(long)]
    pub policy: Option<PathBuf>,

    /// Recording schema to validate (W5 PR3+).
    #[arg(long)]
    pub recording_schema: Option<PathBuf>,

    /// Named scenario from the registry.
    #[arg(long)]
    pub scenario: Option<String>,

    /// Run strict checks (motor coverage etc.) when a scenario builds.
    #[arg(long)]
    pub strict: bool,

    /// Emit a structured JSON report to stdout.
    #[arg(long)]
    pub json: bool,

    /// Seed (forward-compat with `--scenario`; ignored in PR1).
    #[arg(long)]
    pub seed: Option<u64>,
}

#[derive(Serialize)]
struct ValidateReport {
    status: &'static str,
    target: &'static str,
    /// For `--urdf`: number of actuated joints in the layout.
    joint_count: Option<usize>,
    errors: Vec<String>,
}

/// Dispatch on `ValidateArgs` and emit a JSON or text report.
pub fn execute(args: &ValidateArgs) -> ExitCode {
    // Reject the not-yet-implemented branches first.
    if args.scene.is_some() {
        eprintln!("--scene validation not yet implemented — see W5 PR3");
        return ExitCode::from(2);
    }
    if args.policy.is_some() {
        eprintln!("--policy validation not yet implemented — see W7");
        return ExitCode::from(2);
    }
    if args.recording_schema.is_some() {
        eprintln!("--recording-schema validation not yet implemented — see W5 PR3");
        return ExitCode::from(2);
    }

    // `--seed` is forward-compat with `--scenario` builds that randomise
    // initial conditions. PR1 validation is deterministic; the flag is
    // a no-op but accepted for stable CLI shape.
    let _ = args.seed;

    if let Some(name) = &args.scenario {
        return validate_scenario(name, args.json);
    }

    if let Some(urdf) = &args.urdf {
        if args.strict {
            // Design choice C: --strict without --scenario is a no-op.
            // Emit the warning but still run the URDF parse so the
            // command does something useful.
            eprintln!("--strict has no effect without --scenario (W5 PR2)");
        }
        return validate_urdf(urdf, args.json);
    }

    // Neither --urdf nor --scenario nor any other implemented branch.
    eprintln!("validate: pass one of --urdf, --scenario (see --help)");
    ExitCode::FAILURE
}

fn validate_urdf(path: &Path, json: bool) -> ExitCode {
    match clankers_urdf::parse_file(path) {
        Ok(model) => {
            let layout = model.to_layout();
            let report = ValidateReport {
                status: "ok",
                target: "urdf",
                joint_count: Some(layout.len()),
                errors: Vec::new(),
            };
            emit_report(&report, json);
            ExitCode::SUCCESS
        }
        Err(e) => {
            let report = ValidateReport {
                status: "error",
                target: "urdf",
                joint_count: None,
                errors: error_chain(&e),
            };
            emit_report(&report, json);
            ExitCode::FAILURE
        }
    }
}

fn validate_scenario(name: &str, json: bool) -> ExitCode {
    let mut registry = ScenarioRegistry::new();
    register_builtin(&mut registry);

    if registry.get(name).is_some() {
        // No built-in scenarios ship in PR1, so this branch is dead at
        // PR1 time. It activates starting in PR2.
        let report = ValidateReport {
            status: "ok",
            target: "scenario",
            joint_count: None,
            errors: Vec::new(),
        };
        emit_report(&report, json);
        ExitCode::SUCCESS
    } else {
        // PR1 hard-codes the "not yet implemented" message for two
        // future names. Anything else is genuinely unknown.
        let err = if matches!(name, "arm_pick" | "cartpole") {
            ScenarioError::NotImplementedYet {
                name: name.into(),
                migrated_in: "W5 PR2",
            }
        } else {
            ScenarioError::UnknownScenario { name: name.into() }
        };
        let report = ValidateReport {
            status: "error",
            target: "scenario",
            joint_count: None,
            errors: vec![err.to_string()],
        };
        emit_report(&report, json);
        ExitCode::FAILURE
    }
}

fn error_chain<E: std::error::Error>(err: &E) -> Vec<String> {
    let mut out = vec![err.to_string()];
    let mut src = err.source();
    while let Some(e) = src {
        out.push(e.to_string());
        src = e.source();
    }
    out
}

fn emit_report(report: &ValidateReport, json: bool) {
    if json {
        // Pretty-printed JSON. Keys are stable order via the struct
        // field declaration order (serde respects this).
        let s = serde_json::to_string_pretty(report).expect("ValidateReport is serializable");
        println!("{s}");
    } else if report.status == "ok" {
        let dim = report
            .joint_count
            .map(|n| format!(" ({n} joints)"))
            .unwrap_or_default();
        println!("validate {}: ok{}", report.target, dim);
    } else {
        println!("validate {}: error", report.target);
        for e in &report.errors {
            println!("  {e}");
        }
    }
}
