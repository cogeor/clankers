//! `clankers-app inspect` — read-only artefact dumps.
//!
//! In W5 PR1 only `inspect urdf` is implemented. The `mcap`, `onnx`,
//! and `scene` subtargets ship as clear exit-2 placeholders so the CLI
//! tree is in its final shape before PR3 / future workstreams fill the
//! bodies. See PLAN.md Design choice F for the rationale.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clankers_core::layout::JointLayout;
use clap::Subcommand;
use serde::Serialize;

/// Which artefact `inspect` should dump. `--json` lives on each
/// subcommand so it can be passed after the positional path
/// (`inspect urdf <path> --json`) rather than before the subcommand.
#[derive(Subcommand, Debug)]
pub enum InspectTarget {
    /// Parse a URDF and print joint layout, hash, and per-joint metadata.
    Urdf {
        /// Path to the URDF file.
        path: PathBuf,
        /// Emit structured JSON to stdout.
        #[arg(long)]
        json: bool,
    },
    /// Dump MCAP metadata. Placeholder (W5 PR3+).
    Mcap {
        /// Path to the MCAP file.
        path: PathBuf,
        /// Emit structured JSON to stdout.
        #[arg(long)]
        json: bool,
    },
    /// Dump ONNX inputs / outputs. Placeholder (post-W7 follow-up).
    Onnx {
        /// Path to the ONNX file.
        path: PathBuf,
        /// Emit structured JSON to stdout.
        #[arg(long)]
        json: bool,
    },
    /// Dump scene config. Placeholder (W5 PR3+).
    Scene {
        /// Path to the scene config file.
        path: PathBuf,
        /// Emit structured JSON to stdout.
        #[arg(long)]
        json: bool,
    },
}

/// Dispatch on the inspect target.
pub fn execute(target: InspectTarget) -> ExitCode {
    match target {
        InspectTarget::Urdf { path, json } => inspect_urdf(&path, json),
        InspectTarget::Mcap { path: _, json: _ } => {
            eprintln!("MCAP inspect not yet implemented — see W5 PR3 (record/replay)");
            ExitCode::from(2)
        }
        InspectTarget::Onnx { path: _, json: _ } => {
            eprintln!(
                "ONNX inspect not yet implemented — clankers-onnx crate is not yet part of the workspace"
            );
            ExitCode::from(2)
        }
        InspectTarget::Scene { path: _, json: _ } => {
            eprintln!(
                "scene inspect not yet implemented — SceneConfig parsing is not wired into the CLI yet"
            );
            ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------------------
// JSON shape
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct LimitsJson {
    lower: Option<f32>,
    upper: Option<f32>,
    effort: f32,
    velocity: f32,
}

#[derive(Serialize)]
struct JointJson {
    name: String,
    joint_type: String,
    axis: [f32; 3],
    limits: LimitsJson,
}

#[derive(Serialize)]
struct LayoutJson {
    hash: String,
    limits_hash: String,
    count: usize,
    names_in_order: Vec<String>,
    version: u32,
}

#[derive(Serialize)]
struct InspectUrdfOutput {
    joints: Vec<JointJson>,
    joint_layout: LayoutJson,
}

// ---------------------------------------------------------------------------
// inspect urdf
// ---------------------------------------------------------------------------

fn inspect_urdf(path: &Path, json: bool) -> ExitCode {
    let model = match clankers_urdf::parse_file(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to parse URDF: {e}");
            return ExitCode::FAILURE;
        }
    };
    let layout = model.to_layout();

    // Structural hash via std::hash::Hash + DefaultHasher
    let mut h = DefaultHasher::new();
    layout.hash(&mut h);
    let structural = format!("{:016x}", h.finish());
    let limits = format!("{:016x}", layout.limits_hash());

    let out = InspectUrdfOutput {
        joints: layout
            .joints()
            .iter()
            .map(|j| JointJson {
                name: j.name.clone(),
                joint_type: format!("{:?}", j.joint_type),
                axis: j.axis,
                limits: LimitsJson {
                    lower: j.limits.lower,
                    upper: j.limits.upper,
                    effort: j.limits.effort,
                    velocity: j.limits.velocity,
                },
            })
            .collect(),
        joint_layout: LayoutJson {
            hash: structural,
            limits_hash: limits,
            count: layout.len(),
            names_in_order: layout.joint_names().map(String::from).collect(),
            version: JointLayout::SCHEMA_VERSION,
        },
    };

    if json {
        let s = serde_json::to_string_pretty(&out).expect("InspectUrdfOutput is serializable");
        println!("{s}");
    } else {
        println!("URDF: {}", path.display());
        println!("  robot name: {}", model.name);
        println!("  joint layout hash: {}", out.joint_layout.hash);
        println!("  limits hash:       {}", out.joint_layout.limits_hash);
        println!("  joints ({} in layout order):", out.joint_layout.count);
        for j in &out.joints {
            println!(
                "    {:<24} {} axis={:?} limits=[{:?}, {:?}, eff={} vel={}]",
                j.name,
                j.joint_type,
                j.axis,
                j.limits.lower,
                j.limits.upper,
                j.limits.effort,
                j.limits.velocity,
            );
        }
    }

    ExitCode::SUCCESS
}
