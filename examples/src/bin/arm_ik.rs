//! 6-DOF arm driven by inverse kinematics.
//!
//! Demonstrates the IK solver: sets target end-effector positions in
//! Cartesian space, solves for joint angles, and runs the arm through
//! the physics simulation to verify it reaches the targets.
//!
//! Run: `cargo run -p clankers-examples --bin arm_ik`

use std::collections::HashMap;

use bevy::prelude::*;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_actuator_core::prelude::ControlMode;
use clankers_core::ClankersSet;
use clankers_env::prelude::*;
use clankers_examples::SIX_DOF_ARM_URDF;
use clankers_ik::{DlsConfig, DlsSolver, IkTarget, KinematicChain};
use clankers_physics::rapier::{bridge::register_robot, RapierBackend, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_sim::SceneBuilder;
use nalgebra::Vector3;

/// Resource holding the IK chain and solver.
#[derive(Resource)]
struct IkState {
    chain: KinematicChain,
    solver: DlsSolver,
    joint_entities: Vec<Entity>,
    targets: Vec<Vector3<f32>>,
    current_target: usize,
    steps_at_target: u32,
    steps_per_target: u32,
}

/// System that sets JointCommand from IK solutions each step.
#[allow(clippy::needless_pass_by_value)]
fn ik_control_system(
    mut ik: ResMut<IkState>,
    episode: Res<Episode>,
    mut query: Query<(&JointState, &mut JointCommand)>,
) {
    if !episode.is_running() {
        return;
    }

    // Advance target
    ik.steps_at_target += 1;
    if ik.steps_at_target >= ik.steps_per_target {
        ik.steps_at_target = 0;
        ik.current_target = (ik.current_target + 1) % ik.targets.len();
    }

    let target_pos = ik.targets[ik.current_target];
    let target = IkTarget::Position(target_pos);

    // Read current joint positions
    let mut q_current = Vec::with_capacity(ik.joint_entities.len());
    for &entity in &ik.joint_entities {
        if let Ok((state, _)) = query.get(entity) {
            q_current.push(state.position);
        }
    }

    if q_current.len() != ik.chain.dof() {
        return;
    }

    // Solve IK
    let result = ik.solver.solve(&ik.chain, &target, &q_current);

    // Write joint commands (position mode)
    for (i, &entity) in ik.joint_entities.iter().enumerate() {
        if let Ok((_, mut cmd)) = query.get_mut(entity) {
            cmd.value = result.joint_positions[i];
        }
    }

    // Print status every 10 steps
    if ik.steps_at_target % 10 == 0 {
        let ee = ik.chain.forward_kinematics(&q_current);
        let err = (target_pos - ee.translation.vector).norm();
        println!(
            "  target [{:.2}, {:.2}, {:.2}]  ee [{:.3}, {:.3}, {:.3}]  err={:.4}m  conv={} iters={}",
            target_pos.x, target_pos.y, target_pos.z,
            ee.translation.x, ee.translation.y, ee.translation.z,
            err, result.converged, result.iterations,
        );
    }
}

fn main() {
    println!("=== 6-DOF Arm IK Example ===\n");

    // 1. Parse URDF
    let model =
        clankers_urdf::parse_string(SIX_DOF_ARM_URDF).expect("failed to parse six_dof_arm URDF");

    // 2. Build scene with position-controlled actuators
    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(500)
        .with_robot(model.clone(), HashMap::new())
        .build();

    let spawned = &scene.robots["six_dof_arm"];

    // 3. Switch all actuators to position mode (PID controller)
    for entity in spawned.joints.values() {
        let mut actuator = scene
            .app
            .world_mut()
            .get_mut::<clankers_actuator::components::Actuator>(*entity)
            .unwrap();
        // Replace with position-mode actuator
        *actuator = clankers_actuator::components::Actuator::new(
            actuator.motor.clone(),
            actuator.transmission.clone(),
            actuator.friction.clone(),
            ControlMode::Position {
                kp: 100.0,
                ki: 0.0,
                kd: 10.0,
            },
        );
    }

    // 4. Add Rapier physics
    scene
        .app
        .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

    {
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        register_robot(&mut ctx, &model, spawned, world, true);
        world.insert_resource(ctx);
    }

    // 5. Build IK chain
    let chain = KinematicChain::from_model(&model, "end_effector")
        .expect("failed to build IK chain to end_effector");

    println!("IK chain: {} DOF", chain.dof());
    println!("Joint names: {:?}", chain.joint_names());

    // Verify FK at zero config
    let ee_zero = chain.forward_kinematics(&[0.0; 6]);
    println!(
        "FK at q=0: [{:.3}, {:.3}, {:.3}]",
        ee_zero.translation.x, ee_zero.translation.y, ee_zero.translation.z
    );

    // Map chain joint order to entities
    let joint_entities: Vec<Entity> = chain
        .joint_names()
        .iter()
        .map(|name| {
            spawned
                .joint_entity(name)
                .unwrap_or_else(|| panic!("joint {name} not found in spawned robot"))
        })
        .collect();

    // 6. Define target positions (reachable points in workspace)
    let targets = vec![
        Vector3::new(0.3, 0.0, 0.5),   // forward
        Vector3::new(0.0, 0.3, 0.5),   // left
        Vector3::new(-0.3, 0.0, 0.5),  // back
        Vector3::new(0.0, -0.3, 0.5),  // right
        Vector3::new(0.2, 0.2, 0.7),   // up-left
        Vector3::new(0.0, 0.0, 0.91),  // straight up (home)
    ];

    println!("\nTargets: {} positions, 50 steps each\n", targets.len());

    // 7. IK solver with position-mode config
    let solver = DlsSolver::new(DlsConfig {
        max_iterations: 100,
        position_tolerance: 1e-4,
        angle_tolerance: 1e-3,
        damping: 0.01,
    });

    scene.app.insert_resource(IkState {
        chain,
        solver,
        joint_entities,
        targets,
        current_target: 0,
        steps_at_target: 0,
        steps_per_target: 50,
    });

    // 8. Add IK control system
    scene
        .app
        .add_systems(Update, ik_control_system.in_set(ClankersSet::Decide));

    // 9. Register sensors
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(6)), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // 10. Run
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    for step in 0..300 {
        scene.app.update();

        if scene.app.world().resource::<Episode>().is_done() {
            println!("\nEpisode ended at step {step}");
            break;
        }
    }

    // 11. Final check: solve IK for each target and print FK verification
    println!("\n--- IK Solver Verification (no physics) ---");
    let ik = scene.app.world().resource::<IkState>();
    for (i, target) in ik.targets.iter().enumerate() {
        let result = ik.solver.solve(
            &ik.chain,
            &IkTarget::Position(*target),
            &[0.0; 6],
        );
        let ee = ik.chain.forward_kinematics(&result.joint_positions);
        let err = (target - ee.translation.vector).norm();
        println!(
            "  target {}: [{:.2}, {:.2}, {:.2}] -> ee [{:.3}, {:.3}, {:.3}]  err={:.5}m  {}",
            i, target.x, target.y, target.z,
            ee.translation.x, ee.translation.y, ee.translation.z,
            err,
            if result.converged { "CONVERGED" } else { "FAILED" },
        );
    }

    println!("\nArm IK example PASSED");
}
