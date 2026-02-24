//! Domain randomization across episodes.
//!
//! Tests: DomainRandPlugin, actuator randomization, deterministic seeding,
//! parameter variance across episodes, per-episode stats.
//!
//! Run: `cargo run -p clankers-examples --bin domain_rand`

use std::collections::HashMap;

use clankers_actuator::components::{Actuator, JointCommand, JointState};
use clankers_actuator_core::motor::MotorType;
use clankers_domain_rand::prelude::*;
use clankers_env::prelude::*;
use clankers_examples::PENDULUM_URDF;
use clankers_sim::{EpisodeStats, SceneBuilder};

fn main() {
    println!("=== Domain Randomization Example ===\n");

    let num_episodes = 10;
    let max_steps: u32 = 20;

    // ---------------------------------------------------------------
    // 1. Build scene
    // ---------------------------------------------------------------
    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(max_steps)
        .with_robot_urdf(PENDULUM_URDF, HashMap::new())
        .expect("failed to parse pendulum URDF")
        .build();

    let pivot = scene.robots["pendulum"]
        .joint_entity("pivot")
        .expect("missing pivot joint");

    // Print original actuator parameters
    let actuator = scene.app.world().get::<Actuator>(pivot).unwrap();
    if let MotorType::Ideal(motor) = &actuator.motor {
        println!(
            "Original: max_torque={:.1} max_velocity={:.1}",
            motor.max_torque, motor.max_velocity
        );
    }
    println!(
        "Original friction: coulomb={:.3} viscous={:.3}",
        actuator.friction.coulomb, actuator.friction.viscous
    );

    // ---------------------------------------------------------------
    // 2. Add domain randomization
    // ---------------------------------------------------------------
    scene.app.add_plugins(ClankersDomainRandPlugin);

    let motor_rand = MotorRandomizer {
        max_torque: Some(RandomizationRange::uniform(5.0, 20.0).unwrap()),
        max_velocity: Some(RandomizationRange::uniform(2.0, 10.0).unwrap()),
        ..Default::default()
    };
    let friction_rand = FrictionRandomizer {
        coulomb: Some(RandomizationRange::uniform(0.01, 0.2).unwrap()),
        viscous: Some(RandomizationRange::uniform(0.01, 0.5).unwrap()),
        ..Default::default()
    };
    let config = DomainRandConfig::default()
        .with_seed(42)
        .with_actuator(ActuatorRandomizer {
            motor: motor_rand,
            friction: friction_rand,
            ..Default::default()
        });
    scene.app.insert_resource(config);

    // ---------------------------------------------------------------
    // 3. Run episodes, observe randomized parameters
    // ---------------------------------------------------------------
    println!("\n{:<5} {:>10} {:>10} {:>10} {:>10} {:>12}",
             "Ep", "MaxTorque", "MaxVel", "Coulomb", "Viscous", "FinalPos");

    let mut torques = Vec::new();
    let mut velocities = Vec::new();

    for ep in 0..num_episodes {
        scene.app.world_mut().resource_mut::<Episode>().reset(None);

        // Run one step to trigger randomization
        scene.app.update();

        // Read randomized parameters
        let actuator = scene.app.world().get::<Actuator>(pivot).unwrap();
        let (max_torque, max_vel) = if let MotorType::Ideal(motor) = &actuator.motor {
            (motor.max_torque, motor.max_velocity)
        } else {
            (0.0, 0.0)
        };
        let coulomb = actuator.friction.coulomb;
        let viscous = actuator.friction.viscous;

        torques.push(max_torque);
        velocities.push(max_vel);

        // Apply constant command and run remaining steps
        for _ in 1..max_steps {
            scene
                .app
                .world_mut()
                .get_mut::<JointCommand>(pivot)
                .unwrap()
                .value = 5.0;
            scene.app.update();

            if scene.app.world().resource::<Episode>().is_done() {
                break;
            }
        }

        let final_pos = scene
            .app
            .world()
            .get::<JointState>(pivot)
            .unwrap()
            .position;

        println!(
            "{:>3}   {:>10.3} {:>10.3} {:>10.4} {:>10.4} {:>12.4}",
            ep + 1,
            max_torque,
            max_vel,
            coulomb,
            viscous,
            final_pos,
        );
    }

    // ---------------------------------------------------------------
    // 4. Verify randomization produced variance
    // ---------------------------------------------------------------
    println!("\n--- Variance check ---");

    let mean_torque: f32 = torques.iter().sum::<f32>() / torques.len() as f32;
    let var_torque: f32 = torques.iter().map(|t| (t - mean_torque).powi(2)).sum::<f32>()
        / torques.len() as f32;

    let mean_vel: f32 = velocities.iter().sum::<f32>() / velocities.len() as f32;
    let var_vel: f32 = velocities.iter().map(|v| (v - mean_vel).powi(2)).sum::<f32>()
        / velocities.len() as f32;

    println!("Max torque: mean={mean_torque:.2}, variance={var_torque:.2}");
    println!("Max velocity: mean={mean_vel:.2}, variance={var_vel:.2}");

    assert!(
        var_torque > 0.1,
        "Expected significant torque variance, got {var_torque}"
    );
    assert!(
        var_vel > 0.1,
        "Expected significant velocity variance, got {var_vel}"
    );
    println!("Variance checks PASSED (randomization is working)");

    // ---------------------------------------------------------------
    // 5. Verify deterministic seeding
    // ---------------------------------------------------------------
    println!("\n--- Determinism check ---");

    // Build two identical scenes with the same seed
    fn run_one_episode(seed: u64) -> (f32, f32) {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(5)
            .with_robot_urdf(PENDULUM_URDF, HashMap::new())
            .unwrap()
            .build();
        scene.app.add_plugins(ClankersDomainRandPlugin);
        scene.app.insert_resource(
            DomainRandConfig::default()
                .with_seed(seed)
                .with_actuator(ActuatorRandomizer {
                    motor: MotorRandomizer {
                        max_torque: Some(RandomizationRange::uniform(1.0, 100.0).unwrap()),
                        ..Default::default()
                    },
                    ..Default::default()
                }),
        );
        let pivot = scene.robots["pendulum"].joint_entity("pivot").unwrap();
        scene.app.world_mut().resource_mut::<Episode>().reset(None);
        scene.app.update();
        let actuator = scene.app.world().get::<Actuator>(pivot).unwrap();
        let max_t = if let MotorType::Ideal(m) = &actuator.motor {
            m.max_torque
        } else {
            0.0
        };
        let pos = scene
            .app
            .world()
            .get::<JointState>(pivot)
            .unwrap()
            .position;
        (max_t, pos)
    }

    let (t1, p1) = run_one_episode(123);
    let (t2, p2) = run_one_episode(123);
    let (t3, _) = run_one_episode(456);

    println!("Seed 123 run 1: torque={t1:.4}  pos={p1:.6}");
    println!("Seed 123 run 2: torque={t2:.4}  pos={p2:.6}");
    println!("Seed 456 run 1: torque={t3:.4}");

    assert!(
        (t1 - t2).abs() < f32::EPSILON,
        "Same seed must produce same results"
    );
    assert!(
        (t1 - t3).abs() > f32::EPSILON,
        "Different seeds should produce different results"
    );
    println!("Determinism checks PASSED");

    // ---------------------------------------------------------------
    // 6. Stats
    // ---------------------------------------------------------------
    let stats = scene.app.world().resource::<EpisodeStats>();
    println!("\n=== Summary ===");
    println!(
        "Episodes: {}  Total steps: {}",
        stats.episodes_completed, stats.total_steps
    );

    println!("\nDomain randomization example PASSED");
}
