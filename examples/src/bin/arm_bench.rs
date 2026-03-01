//! Headless 6-DOF arm IK benchmark with MCAP recording.
//!
//! Runs the arm through IK target cycles and records expert demonstrations
//! (joint states + actions) to MCAP files for offline imitation learning.
//!
//! Usage:
//!   cargo run -p clankers-examples --bin arm_bench
//!   cargo run -p clankers-examples --bin arm_bench -- --episodes 5 --output recordings/

use std::path::PathBuf;

use bevy::prelude::*;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::ClankersSet;
use clankers_env::prelude::*;
use clankers_examples::arm_setup::{
    ArmIkState, ArmSetupConfig, arm_ik_control_system, arm_ik_solver, arm_ik_targets, setup_arm,
};
use clankers_ik::IkTarget;
use clankers_record::prelude::*;
use clap::Parser;

#[derive(Parser)]
#[command(about = "Headless 6-DOF arm IK benchmark with MCAP recording")]
struct Args {
    /// Number of episodes to record
    #[arg(long, default_value_t = 1)]
    episodes: u32,

    /// Steps per IK target before advancing
    #[arg(long, default_value_t = 50)]
    steps_per_target: u32,

    /// Output directory for MCAP files
    #[arg(long, default_value = ".")]
    output: PathBuf,

    /// Maximum steps per episode
    #[arg(long, default_value_t = 500)]
    max_steps: u32,
}

/// Capture action system: reads `JointCommand` values from arm entities
/// and writes them to `PendingAction` for the recorder.
#[allow(clippy::needless_pass_by_value)]
fn capture_action_system(
    ik: Res<ArmIkState>,
    mut pending: ResMut<PendingAction>,
    query: Query<&JointCommand>,
) {
    let mut values = Vec::with_capacity(ik.joint_entities.len());
    for &entity in &ik.joint_entities {
        if let Ok(cmd) = query.get(entity) {
            values.push(cmd.value);
        }
    }
    pending.0 = values;
}

fn main() {
    let args = Args::parse();

    println!("=== 6-DOF Arm IK Bench ===");
    println!(
        "  episodes={}, steps_per_target={}, max_steps={}, output={}\n",
        args.episodes, args.steps_per_target, args.max_steps, args.output.display(),
    );

    // Create output directory if needed
    if !args.output.exists() {
        std::fs::create_dir_all(&args.output).expect("failed to create output directory");
    }

    for ep in 0..args.episodes {
        println!("--- Episode {}/{} ---", ep + 1, args.episodes);

        let output_path = args.output.join(format!("arm_episode_{ep:03}.mcap"));

        // 1. Setup arm
        let setup = setup_arm(ArmSetupConfig {
            max_episode_steps: args.max_steps,
            ..ArmSetupConfig::default()
        });
        let mut scene = setup.scene;

        // 2. Add recorder plugin
        scene.app.insert_resource(RecordingConfig {
            output_path: output_path.clone(),
            ..RecordingConfig::default()
        });
        scene.app.add_plugins(RecorderPlugin);

        // 3. IK state
        let targets = arm_ik_targets();
        let solver = arm_ik_solver();

        let joint_entities = setup.joint_entities.clone();

        scene.app.insert_resource(ArmIkState {
            chain: setup.chain,
            solver,
            joint_entities: setup.joint_entities,
            targets,
            current_target: 0,
            steps_at_target: 0,
            steps_per_target: args.steps_per_target,
        });

        // 4. Add systems
        scene.app.add_systems(
            Update,
            (
                arm_ik_control_system.in_set(ClankersSet::Decide),
                capture_action_system.after(arm_ik_control_system),
            ),
        );

        // 5. Run episode
        scene.app.world_mut().resource_mut::<Episode>().reset(None);

        let mut final_step = 0;
        for step in 0..args.max_steps {
            scene.app.update();
            final_step = step;

            if scene.app.world().resource::<Episode>().is_done() {
                break;
            }
        }

        // 6. Print metrics
        let ik = scene.app.world().resource::<ArmIkState>();
        let target_pos = ik.targets[ik.current_target];

        // Read current joint positions for FK
        let mut q_current = Vec::new();
        for &entity in &joint_entities {
            if let Some(state) = scene.app.world().get::<JointState>(entity) {
                q_current.push(state.position);
            }
        }

        if q_current.len() == ik.chain.dof() {
            let ee = ik.chain.forward_kinematics(&q_current);
            let err = (target_pos - ee.translation.vector).norm();
            let result = ik
                .solver
                .solve(&ik.chain, &IkTarget::Position(target_pos), &q_current);
            println!(
                "  steps={}, target=[{:.2},{:.2},{:.2}], ee=[{:.3},{:.3},{:.3}], err={:.4}m, conv={}",
                final_step + 1,
                target_pos.x,
                target_pos.y,
                target_pos.z,
                ee.translation.x,
                ee.translation.y,
                ee.translation.z,
                err,
                result.converged,
            );
        }

        println!("  wrote {}\n", output_path.display());
    }

    println!("Arm bench DONE");
}
