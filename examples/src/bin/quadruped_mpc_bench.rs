//! Headless MPC benchmark for quadruped locomotion.
//!
//! Uses the shared `mpc_control` module with position motors, matching the
//! real headless/viz examples exactly. Reports performance metrics.
//!
//! Usage:
//!   cargo run -p clankers-examples --bin quadruped_mpc_bench -- --velocity 0.3
//!   cargo run -p clankers-examples --bin quadruped_mpc_bench -- --velocity 0.5 --gait trot --steps 500

use clap::Parser;
use clankers_actuator::components::JointState;
use clankers_env::prelude::*;
use clankers_examples::mpc_control::{MpcLoopState, body_state_from_rapier, compute_mpc_step, detect_foot_contacts};
use clankers_examples::quadruped_setup::{QuadrupedSetupConfig, setup_quadruped};
use clankers_mpc::{AdaptiveGaitConfig, GaitScheduler, GaitType, MpcConfig, MpcSolver, SwingConfig};
use clankers_physics::rapier::RapierContext;
use nalgebra::Vector3;
use rapier3d::prelude::JointAxis;

#[derive(Parser)]
#[command(about = "Headless MPC benchmark for quadruped locomotion")]
struct Args {
    /// Target forward velocity (m/s)
    #[arg(long, default_value_t = 0.3)]
    velocity: f64,

    /// Gait type: stand, trot, walk, bound
    #[arg(long, default_value = "trot")]
    gait: String,

    /// Number of locomotion steps (at 50Hz)
    #[arg(long, default_value_t = 500)]
    steps: u32,

    /// Stabilization steps before locomotion
    #[arg(long, default_value_t = 100)]
    stabilize: u32,

    /// Velocity ramp steps after gait switch
    #[arg(long, default_value_t = 100)]
    ramp: u32,

    /// Override q_weights[9] and [10] (vx, vy velocity tracking)
    #[arg(long)]
    q_vx: Option<f64>,

    /// Override q_weights[5] (pz height tracking)
    #[arg(long)]
    q_pz: Option<f64>,

    /// Override r_weight (control effort cost)
    #[arg(long)]
    r_weight: Option<f64>,

    /// Override MPC horizon (solver re-created with correct dimensions)
    #[arg(long)]
    horizon: Option<usize>,

    /// Override friction coefficient (Coulomb)
    #[arg(long)]
    mu: Option<f64>,

    /// Override max normal force per foot (N)
    #[arg(long)]
    f_max: Option<f64>,

    /// Override capture-point gain for foot placement
    #[arg(long)]
    cp_gain: Option<f64>,

    /// Override q_weights[0,1] (roll, pitch orientation tracking)
    #[arg(long)]
    q_roll: Option<f64>,

    /// Override q_weights[6,7] (wx, wy angular velocity damping)
    #[arg(long)]
    q_omega: Option<f64>,

    /// Override trot gait cycle time in seconds (default 0.35)
    #[arg(long)]
    cycle_time: Option<f64>,

    /// Override trot gait duty factor (default 0.5)
    #[arg(long)]
    duty_factor: Option<f64>,

    /// Override swing step height in meters (default 0.10)
    #[arg(long)]
    step_height: Option<f64>,

    /// Override MPC timestep in seconds (default 0.02 = 50Hz)
    #[arg(long)]
    mpc_dt: Option<f64>,

    /// Override simulation ground/foot friction coefficient (default 0.6)
    #[arg(long)]
    mu_sim: Option<f32>,

    /// Enable velocity-dependent adaptive gait timing
    #[arg(long)]
    adaptive_gait: bool,
}

fn parse_gait(s: &str) -> GaitType {
    match s.to_lowercase().as_str() {
        "stand" => GaitType::Stand,
        "trot" => GaitType::Trot,
        "walk" => GaitType::Walk,
        "bound" => GaitType::Bound,
        _ => {
            eprintln!("Unknown gait '{s}', using Trot");
            GaitType::Trot
        }
    }
}

fn main() {
    let args = Args::parse();
    let gait_type = parse_gait(&args.gait);
    let desired_velocity = Vector3::new(args.velocity, 0.0, 0.0);
    let stabilize_steps = args.stabilize as usize;
    let ramp_steps = args.ramp as usize;
    let total_steps = args.steps as usize;

    println!("=== Quadruped MPC Benchmark ===");
    println!("  Gait: {:?}", gait_type);
    println!("  Velocity: {:.2} m/s", args.velocity);
    println!("  Steps: {} stabilize + {} locomotion", stabilize_steps, total_steps);
    println!();

    // --- Setup ---
    let setup = setup_quadruped(QuadrupedSetupConfig {
        sim_friction: args.mu_sim.unwrap_or(0.6),
        ..QuadrupedSetupConfig::default()
    });
    let mut scene = setup.scene;
    let desired_height = setup.desired_height;
    let n_feet = setup.n_feet;

    // Build MPC config with CLI overrides BEFORE creating solver
    let mut mpc_config = MpcConfig::default();
    if let Some(v) = args.q_vx {
        mpc_config.q_weights[9] = v;
        mpc_config.q_weights[10] = v;
    }
    if let Some(v) = args.q_pz {
        mpc_config.q_weights[5] = v;
    }
    if let Some(v) = args.r_weight {
        mpc_config.r_weight = v;
    }
    if let Some(h) = args.horizon {
        mpc_config.horizon = h;
    }
    if let Some(v) = args.mu {
        mpc_config.friction_coeff = v;
    }
    if let Some(v) = args.f_max {
        mpc_config.f_max = v;
    }
    if let Some(v) = args.q_roll {
        mpc_config.q_weights[0] = v; // roll
        mpc_config.q_weights[1] = v; // pitch
    }
    if let Some(v) = args.q_omega {
        mpc_config.q_weights[6] = v; // wx
        mpc_config.q_weights[7] = v; // wy
    }
    if let Some(v) = args.mpc_dt {
        mpc_config.dt = v;
    }
    let dt = mpc_config.dt;

    // Build swing config with CLI overrides
    let mut swing_config = SwingConfig::default();
    if let Some(v) = args.cp_gain {
        swing_config.cp_gain = v;
    }
    if let Some(v) = args.step_height {
        swing_config.step_height = v;
    }

    // Print active overrides
    {
        let has_overrides = args.q_vx.is_some() || args.q_pz.is_some() || args.r_weight.is_some()
            || args.horizon.is_some() || args.mu.is_some() || args.f_max.is_some()
            || args.cp_gain.is_some()
            || args.q_roll.is_some() || args.q_omega.is_some()
            || args.cycle_time.is_some() || args.duty_factor.is_some() || args.step_height.is_some()
            || args.mpc_dt.is_some()
            || args.mu_sim.is_some();
        if has_overrides {
            println!("Overrides:");
            if let Some(v) = args.q_vx { println!("  q_vx={v}"); }
            if let Some(v) = args.q_pz { println!("  q_pz={v}"); }
            if let Some(v) = args.r_weight { println!("  r_weight={v}"); }
            if let Some(h) = args.horizon { println!("  horizon={h}"); }
            if let Some(v) = args.mu { println!("  mu={v}"); }
            if let Some(v) = args.f_max { println!("  f_max={v}"); }
            if let Some(v) = args.cp_gain { println!("  cp_gain={v}"); }
            if let Some(v) = args.q_roll { println!("  q_roll={v}"); }
            if let Some(v) = args.q_omega { println!("  q_omega={v}"); }
            if let Some(v) = args.cycle_time { println!("  cycle_time={v}"); }
            if let Some(v) = args.duty_factor { println!("  duty_factor={v}"); }
            if let Some(v) = args.step_height { println!("  step_height={v}"); }
            if let Some(v) = args.mpc_dt { println!("  mpc_dt={v}"); }
            if let Some(v) = args.mu_sim { println!("  mu_sim={v}"); }
            println!();
        }
    }

    // Create solver AFTER all config overrides (pre-allocates matrices sized by horizon)
    let mut mpc_state = MpcLoopState {
        gait: GaitScheduler::quadruped(GaitType::Stand),
        solver: MpcSolver::new(mpc_config.clone(), 4),
        config: mpc_config,
        swing_config,
        adaptive_gait: if args.adaptive_gait { Some(AdaptiveGaitConfig::default()) } else { None },
        legs: setup.legs,
        swing_starts: vec![Vector3::zeros(); n_feet],
        swing_targets: vec![Vector3::zeros(); n_feet],
        prev_contacts: vec![true; n_feet],
        init_joint_angles: setup.init_joint_angles,
        foot_link_names: None,
        disturbance_estimator: None,
    };

    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    // --- Run simulation ---
    let mut min_z = f64::MAX;
    let mut max_roll = 0.0_f64;
    let mut max_pitch = 0.0_f64;
    let mut total_solve_us: u64 = 0;
    let mut solve_count: u64 = 0;
    let all_steps = stabilize_steps + total_steps;

    println!("\nRunning {} total steps ({:.1}s)...", all_steps, all_steps as f64 * dt);

    for step in 0..all_steps {
        // Switch gait after stabilization
        if step == stabilize_steps {
            let mut gait = GaitScheduler::quadruped(gait_type);
            // Apply gait overrides if any
            if args.cycle_time.is_some() || args.duty_factor.is_some() {
                let base = GaitScheduler::quadruped(gait_type);
                let ct = args.cycle_time.unwrap_or(base.cycle_time());
                let df = args.duty_factor.unwrap_or(base.duty_factor());
                let offsets = match gait_type {
                    GaitType::Trot => vec![0.0, 0.5, 0.5, 0.0],
                    GaitType::Walk => vec![0.0, 0.5, 0.25, 0.75],
                    GaitType::Bound => vec![0.0, 0.0, 0.5, 0.5],
                    GaitType::Stand => vec![0.0; 4],
                };
                gait = GaitScheduler::custom(offsets, df, ct);
            }
            mpc_state.gait = gait;
            println!("  >>> Switched to {:?} at step {step}", gait_type);
        }

        let (body_state, body_quat, actual_contacts) = {
            let mut ctx = scene.app.world_mut().resource_mut::<RapierContext>();
            ctx.rebase_origin("body", 50.0);
            let (bs, bq) = body_state_from_rapier(ctx.as_ref(), "body").expect("body not found");
            let contacts = detect_foot_contacts(ctx.as_ref(), &mpc_state);
            (bs, bq, contacts)
        };

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

        // Ramp velocity
        let current_vel = if step < stabilize_steps {
            Vector3::zeros()
        } else {
            let ramp_frac = ((step - stabilize_steps) as f64 / ramp_steps as f64).min(1.0);
            desired_velocity * ramp_frac
        };

        let result = compute_mpc_step(
            &mut mpc_state,
            &body_state,
            &body_quat,
            &all_joint_positions,
            &all_joint_velocities,
            &current_vel,
            desired_height,
            0.0,
            0.0,
            actual_contacts.as_deref(),
        );

        // Apply motor commands via manual Rapier stepping (matching headless)
        {
            let world = scene.app.world_mut();
            let mut ctx = world.remove_resource::<RapierContext>().unwrap();

            for mc in &result.motor_commands {
                let Some(&jh) = ctx.joint_handles.get(&mc.entity) else { continue };
                let Some(joint) = ctx.impulse_joint_set.get_mut(jh, true) else { continue };

                joint.data.set_motor(
                    JointAxis::AngX,
                    mc.target_pos,
                    mc.target_vel,
                    mc.stiffness,
                    mc.damping,
                );
                joint.data.set_motor_max_force(JointAxis::AngX, mc.max_force);
            }

            // Compute substeps from MPC dt / physics dt (0.001s)
            let substeps = (dt / 0.001).round() as usize;
            for _ in 0..substeps {
                ctx.step();
            }

            // Read back joint state
            for leg in &mpc_state.legs {
                for &entity in &leg.joint_entities {
                    let Some(info) = ctx.joint_info.get(&entity) else { continue };
                    let Some(pb) = ctx.rigid_body_set.get(info.parent_body) else { continue };
                    let Some(cb) = ctx.rigid_body_set.get(info.child_body) else { continue };

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

        // Track metrics (only during locomotion phase)
        if step >= stabilize_steps {
            min_z = min_z.min(body_state.position.z);
            max_roll = max_roll.max(body_state.orientation.x.abs());
            max_pitch = max_pitch.max(body_state.orientation.y.abs());
            total_solve_us += result.solution.solve_time_us;
            solve_count += 1;
        }

        // Periodic telemetry
        if step % 100 == 0 {
            println!(
                "  step {:4}: pos=[{:+.3}, {:+.3}, {:+.3}]  vel=[{:+.3}, {:+.3}, {:+.3}]  mpc={:>4}us  {}",
                step,
                body_state.position.x, body_state.position.y, body_state.position.z,
                body_state.linear_velocity.x, body_state.linear_velocity.y, body_state.linear_velocity.z,
                result.solution.solve_time_us,
                if result.solution.converged { "OK" } else { "FAIL" },
            );
        }
    }

    // --- Final report ---
    let (final_state, _) = {
        let ctx = scene.app.world().resource::<RapierContext>();
        body_state_from_rapier(ctx, "body").expect("body not found")
    };

    let sim_time = total_steps as f64 * dt;
    let avg_speed = final_state.position.x / (all_steps as f64 * dt);
    let avg_solve = if solve_count > 0 { total_solve_us / solve_count } else { 0 };

    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK RESULTS");
    println!("{}", "=".repeat(60));
    println!("  Gait:          {:?}", gait_type);
    println!("  Target vel:    {:.2} m/s", args.velocity);
    println!("  Sim time:      {:.1}s ({} loco steps)", sim_time, total_steps);
    println!("  Final X:       {:+.3} m", final_state.position.x);
    println!("  Final Z:       {:+.3} m", final_state.position.z);
    println!("  Avg speed:     {:.3} m/s", avg_speed);
    println!("  Min Z:         {:.3} m", min_z);
    println!("  Max roll:      {:.3} rad ({:.1} deg)", max_roll, max_roll.to_degrees());
    println!("  Max pitch:     {:.3} rad ({:.1} deg)", max_pitch, max_pitch.to_degrees());
    println!("  Avg MPC solve: {} us", avg_solve);
    println!("{}", "=".repeat(60));

    // Quick sanity check
    if final_state.position.z < 0.05 {
        println!("\nWARNING: Robot may have fallen (z={:.3})", final_state.position.z);
    }
}
