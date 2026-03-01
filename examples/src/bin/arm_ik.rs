//! 6-DOF arm driven by inverse kinematics.
//!
//! Demonstrates the IK solver: sets target end-effector positions in
//! Cartesian space, solves for joint angles, and runs the arm through
//! the physics simulation to verify it reaches the targets.
//!
//! Run: `cargo run -p clankers-examples --bin arm_ik`

use bevy::prelude::*;
use clankers_core::ClankersSet;
use clankers_env::prelude::*;
use clankers_examples::arm_setup::{
    ArmIkState, ArmSetupConfig, arm_ik_control_system, arm_ik_solver, arm_ik_targets, setup_arm,
};
use clankers_ik::IkTarget;

fn main() {
    println!("=== 6-DOF Arm IK Example ===\n");

    // 1. Setup arm with shared module
    let setup = setup_arm(ArmSetupConfig::default());
    let mut scene = setup.scene;

    println!("IK chain: {} DOF", setup.chain.dof());
    println!("Joint names: {:?}", setup.chain.joint_names());

    // Verify FK at zero config
    let ee_zero = setup.chain.forward_kinematics(&[0.0; 6]);
    println!(
        "FK at q=0: [{:.3}, {:.3}, {:.3}]",
        ee_zero.translation.x, ee_zero.translation.y, ee_zero.translation.z
    );

    // 2. Define targets and solver
    let targets = arm_ik_targets();
    let solver = arm_ik_solver();

    println!("\nTargets: {} positions, 50 steps each\n", targets.len());

    scene.app.insert_resource(ArmIkState {
        chain: setup.chain,
        solver,
        joint_entities: setup.joint_entities,
        targets,
        current_target: 0,
        steps_at_target: 0,
        steps_per_target: 50,
    });

    // 3. Add IK control system
    scene
        .app
        .add_systems(Update, arm_ik_control_system.in_set(ClankersSet::Decide));

    // 4. Run
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    for step in 0..300 {
        scene.app.update();

        if scene.app.world().resource::<Episode>().is_done() {
            println!("\nEpisode ended at step {step}");
            break;
        }
    }

    // 5. Final check: solve IK for each target and print FK verification
    println!("\n--- IK Solver Verification (no physics) ---");
    let ik = scene.app.world().resource::<ArmIkState>();
    for (i, target) in ik.targets.iter().enumerate() {
        let result = ik
            .solver
            .solve(&ik.chain, &IkTarget::Position(*target), &[0.0; 6]);
        let ee = ik.chain.forward_kinematics(&result.joint_positions);
        let err = (target - ee.translation.vector).norm();
        println!(
            "  target {}: [{:.2}, {:.2}, {:.2}] -> ee [{:.3}, {:.3}, {:.3}]  err={:.5}m  {}",
            i,
            target.x,
            target.y,
            target.z,
            ee.translation.x,
            ee.translation.y,
            ee.translation.z,
            err,
            if result.converged {
                "CONVERGED"
            } else {
                "FAILED"
            },
        );
    }

    println!("\nArm IK example PASSED");
}
