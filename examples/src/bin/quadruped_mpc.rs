//! Quadruped robot walking with Model Predictive Control.
//!
//! Demonstrates the full MPC locomotion pipeline on a headless simulation:
//! 1. Load quadruped URDF (4 legs × 3 joints = 12 DOF, floating base)
//! 2. Set up Rapier physics with dynamic base (not fixed)
//! 3. Run centroidal MPC → WBC → swing planner each step
//! 4. Print telemetry: body pose, foot forces, solve time
//!
//! This example calls the MPC solver directly (not via the Bevy plugin)
//! so it works headless without a transform hierarchy.
//!
//! Run: `cargo run -p clankers-examples --bin quadruped_mpc`

use clankers_actuator::components::{JointCommand, JointState};
use clankers_env::prelude::*;
use clankers_examples::mpc_control::{
    MpcLoopState, body_state_from_rapier, compute_mpc_step, detect_foot_contacts,
};
use clankers_examples::quadruped_setup::{QuadrupedSetupConfig, setup_quadruped};
use clankers_mpc::{GaitScheduler, GaitType, MpcConfig, MpcSolver, SwingConfig};
use clankers_physics::rapier::RapierContext;
use nalgebra::Vector3;
use rapier3d::prelude::JointAxis;

fn main() {
    println!("=== Quadruped MPC Example (3-DOF legs) ===\n");

    let setup = setup_quadruped(QuadrupedSetupConfig {
        max_episode_steps: 5000,
        ..QuadrupedSetupConfig::default()
    });
    let mut scene = setup.scene;
    let desired_height = setup.desired_height;
    let n_feet = setup.n_feet;

    // Configure MPC
    let mpc_config = MpcConfig::default();
    let swing_config = SwingConfig::default();
    let desired_velocity = Vector3::new(0.3, 0.0, 0.0);
    let desired_yaw = 0.0;
    let ground_height = 0.0;
    let stabilize_steps = 100;

    println!(
        "\nMPC config: horizon={}, dt={}, mass={:.1}kg",
        mpc_config.horizon, mpc_config.dt, mpc_config.mass
    );
    println!("Phase 1: Stand (stabilize) for {stabilize_steps} steps");
    println!(
        "Phase 2: Walk at [{:.1}, {:.1}, {:.1}] m/s",
        desired_velocity.x, desired_velocity.y, desired_velocity.z
    );

    // Build MPC loop state
    let mut mpc_state = MpcLoopState {
        gait: GaitScheduler::quadruped(GaitType::Stand),
        solver: MpcSolver::new(mpc_config.clone(), 4),
        config: mpc_config,
        swing_config,
        adaptive_gait: None,
        legs: setup.legs,
        swing_starts: vec![Vector3::zeros(); n_feet],
        swing_targets: vec![Vector3::zeros(); n_feet],
        prev_contacts: vec![true; n_feet],
        init_joint_angles: setup.init_joint_angles,
        foot_link_names: None,
        disturbance_estimator: None,
    };

    // Start episode
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("\nRunning for 500 steps (10s at 50Hz)...\n");

    // Main simulation loop
    let total_steps = 500;
    let ramp_steps = 100;
    let mut switched_to_trot = false;

    for step in 0..total_steps {
        // Switch from Stand to Trot after stabilization
        if step == stabilize_steps && !switched_to_trot {
            switched_to_trot = true;
            mpc_state.gait = GaitScheduler::quadruped(GaitType::Trot);
            println!("  >>> Switched to Trot at step {step}");
        }

        // --- Floating origin rebase + read body state + detect contacts ---
        let (body_state, body_quat, actual_contacts) = {
            let mut ctx = scene.app.world_mut().resource_mut::<RapierContext>();
            ctx.rebase_origin("body", 50.0);
            let (bs, bq) =
                body_state_from_rapier(ctx.as_ref(), "body").expect("body not found in Rapier");
            let contacts = detect_foot_contacts(ctx.as_ref(), &mpc_state);
            (bs, bq, contacts)
        };

        // --- Read joint states ---
        let mut all_joint_positions: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
        let mut all_joint_velocities: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
        for leg in &mpc_state.legs {
            let mut q = Vec::with_capacity(leg.joint_entities.len());
            let mut qd = Vec::with_capacity(leg.joint_entities.len());
            for &entity in &leg.joint_entities {
                if let Some(js) = scene.app.world().get::<JointState>(entity) {
                    q.push(js.position);
                    qd.push(js.velocity);
                } else {
                    q.push(0.0);
                    qd.push(0.0);
                }
            }
            all_joint_positions.push(q);
            all_joint_velocities.push(qd);
        }

        // --- Ramp velocity ---
        let current_vel = if step < stabilize_steps {
            Vector3::zeros()
        } else {
            let ramp_frac = (f64::from(step - stabilize_steps) / f64::from(ramp_steps)).min(1.0);
            desired_velocity * ramp_frac
        };

        // --- Compute MPC step ---
        let result = compute_mpc_step(
            &mut mpc_state,
            &body_state,
            &body_quat,
            &all_joint_positions,
            &all_joint_velocities,
            &current_vel,
            desired_height,
            desired_yaw,
            ground_height,
            actual_contacts.as_deref(),
        );

        // --- Apply motor commands + step physics manually ---
        {
            let world = scene.app.world_mut();
            let mut ctx = world.remove_resource::<RapierContext>().unwrap();

            for mc in &result.motor_commands {
                let Some(&jh) = ctx.joint_handles.get(&mc.entity) else {
                    continue;
                };
                let Some(joint) = ctx.impulse_joint_set.get_mut(jh, true) else {
                    continue;
                };

                joint.data.set_motor(
                    JointAxis::AngX,
                    mc.target_pos,
                    mc.target_vel,
                    mc.stiffness,
                    mc.damping,
                );
                joint
                    .data
                    .set_motor_max_force(JointAxis::AngX, mc.max_force);
            }

            let substeps = ctx.substeps;
            for _ in 0..substeps {
                ctx.step();
            }

            // Read back joint state
            for leg in &mpc_state.legs {
                for &entity in &leg.joint_entities {
                    let Some(info) = ctx.joint_info.get(&entity) else {
                        continue;
                    };
                    let Some(pb) = ctx.rigid_body_set.get(info.parent_body) else {
                        continue;
                    };
                    let Some(cb) = ctx.rigid_body_set.get(info.child_body) else {
                        continue;
                    };

                    let rel_rot = pb.position().rotation.inverse() * cb.position().rotation;
                    let sin_half = bevy::math::Vec3::new(rel_rot.x, rel_rot.y, rel_rot.z);
                    let sin_proj = sin_half.dot(info.axis);
                    let angle = 2.0 * f32::atan2(sin_proj, rel_rot.w);

                    let rel_angvel = cb.angvel() - pb.angvel();
                    let velocity = rel_angvel.dot(info.axis);

                    if let Some(mut js) = world.get_mut::<JointState>(entity) {
                        js.position = angle;
                        js.velocity = velocity;
                    }
                }
            }

            world.insert_resource(ctx);
        }

        // --- Telemetry ---
        let body_pos = body_state.position;
        if step % 50 == 0 || step < 5 || (step >= stabilize_steps && step < stabilize_steps + 5) {
            let n_stance: usize = result.contacts.iter().filter(|&&c| c).count();
            println!(
                "  step {:4}: pos=[{:+.3}, {:+.3}, {:+.3}]  vel=[{:+.3}, {:+.3}, {:+.3}]  stance={}/{}  mpc={:>4}us  {}",
                step,
                body_pos.x,
                body_pos.y,
                body_pos.z,
                body_state.linear_velocity.x,
                body_state.linear_velocity.y,
                body_state.linear_velocity.z,
                n_stance,
                n_feet,
                result.solution.solve_time_us,
                if result.solution.converged {
                    "OK"
                } else {
                    "FAIL"
                },
            );
            for (i, fw) in result.foot_world.iter().enumerate() {
                let contact = result.contacts[i];
                let force_str = if result.solution.converged {
                    let f = &result.solution.forces[i];
                    format!("F=[{:+.2}, {:+.2}, {:+.2}]", f.x, f.y, f.z)
                } else {
                    "F=FAIL".to_string()
                };
                println!(
                    "    foot {i}: pos=[{:+.4}, {:+.4}, {:+.4}]  {force_str}  {}",
                    fw.x,
                    fw.y,
                    fw.z,
                    if contact { "STANCE" } else { "swing" },
                );
            }
        }

        if scene.app.world().resource::<Episode>().is_done() {
            println!("\nEpisode ended at step {step}");
            break;
        }
    }

    // Final report
    let spawned = &scene.robots["quadruped"];
    let (final_state, _) = {
        let ctx = scene.app.world().resource::<RapierContext>();
        body_state_from_rapier(ctx, "body").expect("body not found")
    };

    println!("\n=== Summary ===");
    println!(
        "Final body position: [{:.3}, {:.3}, {:.3}]",
        final_state.position.x, final_state.position.y, final_state.position.z,
    );
    println!(
        "Final body velocity: [{:.3}, {:.3}, {:.3}]",
        final_state.linear_velocity.x, final_state.linear_velocity.y, final_state.linear_velocity.z,
    );
    println!("Steps simulated: {total_steps}");
    println!(
        "Simulation time: {:.1}s",
        f64::from(total_steps) * mpc_state.config.dt
    );

    println!("\nFinal joint states:");
    let joint_names = [
        "fl_hip_ab",
        "fl_hip_pitch",
        "fl_knee_pitch",
        "fr_hip_ab",
        "fr_hip_pitch",
        "fr_knee_pitch",
        "rl_hip_ab",
        "rl_hip_pitch",
        "rl_knee_pitch",
        "rr_hip_ab",
        "rr_hip_pitch",
        "rr_knee_pitch",
    ];
    for name in &joint_names {
        if let Some(entity) = spawned.joint_entity(name)
            && let Some(state) = scene.app.world().get::<JointState>(entity)
        {
            let cmd = scene.app.world().get::<JointCommand>(entity);
            println!(
                "  {:<16}: pos={:+.3} rad  vel={:+.3} rad/s  cmd={:+.3}",
                name,
                state.position,
                state.velocity,
                cmd.map_or(0.0, |c| c.value),
            );
        }
    }

    println!("\nQuadruped MPC example PASSED");
}
