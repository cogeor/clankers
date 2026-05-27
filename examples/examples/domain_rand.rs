//! Domain randomization across episodes. Thin wrapper over
//! [`clankers_sim::scenarios::domain_rand_pendulum::DomainRandPendulumScenario`].
//!
//! Run: `cargo run -p clankers-examples --example domain_rand`

use clankers_actuator::components::{Actuator, JointCommand, JointState};
use clankers_actuator_core::motor::MotorType;
use clankers_env::prelude::Episode;
use clankers_sim::ScenarioConfig;
use clankers_sim::scenarios::domain_rand_pendulum::{
    DomainRandPendulumConfig, DomainRandPendulumScenario,
};

fn run_one(seed: u64, max_steps: u32) -> (f32, f32) {
    let dr_cfg = DomainRandPendulumConfig {
        seed,
        max_episode_steps: max_steps,
        max_torque_range: (1.0, 100.0),
        ..DomainRandPendulumConfig::default()
    };
    let mut a = DomainRandPendulumScenario::build_with(&ScenarioConfig::default(), &dr_cfg);
    a.scene
        .app
        .world_mut()
        .resource_mut::<Episode>()
        .reset(None);
    a.scene.app.update();
    let act = a.scene.app.world().get::<Actuator>(a.pivot).unwrap();
    let mt = if let MotorType::Ideal(m) = &act.motor {
        m.max_torque
    } else {
        0.0
    };
    let pos = a
        .scene
        .app
        .world()
        .get::<JointState>(a.pivot)
        .unwrap()
        .position;
    (mt, pos)
}

fn main() {
    println!("=== Domain Randomization Example ===\n");
    let mut a = DomainRandPendulumScenario::build_with(
        &ScenarioConfig::default(),
        &DomainRandPendulumConfig::default(),
    );
    println!(
        "{:<5} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Ep", "MaxTorque", "MaxVel", "Coulomb", "Viscous", "FinalPos"
    );
    let (mut torques, mut vels) = (Vec::new(), Vec::new());
    for ep in 0..10 {
        a.scene
            .app
            .world_mut()
            .resource_mut::<Episode>()
            .reset(None);
        a.scene.app.update();
        let act = a.scene.app.world().get::<Actuator>(a.pivot).unwrap();
        let (mt, mv) = if let MotorType::Ideal(m) = &act.motor {
            (m.max_torque, m.max_velocity)
        } else {
            (0.0, 0.0)
        };
        let (c, v) = (act.friction.coulomb, act.friction.viscous);
        torques.push(mt);
        vels.push(mv);
        for _ in 1..20 {
            a.scene
                .app
                .world_mut()
                .get_mut::<JointCommand>(a.pivot)
                .unwrap()
                .value = 5.0;
            a.scene.app.update();
            if a.scene.app.world().resource::<Episode>().is_done() {
                break;
            }
        }
        let fp = a
            .scene
            .app
            .world()
            .get::<JointState>(a.pivot)
            .unwrap()
            .position;
        println!(
            "{:>3}   {mt:>10.3} {mv:>10.3} {c:>10.4} {v:>10.4} {fp:>12.4}",
            ep + 1
        );
    }
    let torque_mean: f32 = torques.iter().sum::<f32>() / torques.len() as f32;
    let torque_var: f32 = torques
        .iter()
        .map(|tval| (tval - torque_mean).powi(2))
        .sum::<f32>()
        / torques.len() as f32;
    let vel_mean: f32 = vels.iter().sum::<f32>() / vels.len() as f32;
    let vel_var: f32 = vels
        .iter()
        .map(|vval| (vval - vel_mean).powi(2))
        .sum::<f32>()
        / vels.len() as f32;
    println!("\nMax torque: mean={torque_mean:.2}, variance={torque_var:.2}");
    println!("Max velocity: mean={vel_mean:.2}, variance={vel_var:.2}");
    assert!(
        torque_var > 0.1 && vel_var > 0.1,
        "Expected significant variance"
    );
    let (t1, p1) = run_one(123, 5);
    let (t2, p2) = run_one(123, 5);
    let (t3, _) = run_one(456, 5);
    println!("Seed 123 run 1: torque={t1:.4}  pos={p1:.6}");
    println!("Seed 123 run 2: torque={t2:.4}  pos={p2:.6}");
    println!("Seed 456 run 1: torque={t3:.4}");
    assert!((t1 - t2).abs() < f32::EPSILON && (t1 - t3).abs() > f32::EPSILON);
    println!("Determinism checks PASSED\nDomain randomization example PASSED");
}
