//! Headless multi-environment cart-pole benchmark. Thin wrapper over
//! [`clankers_sim::scenarios::cartpole::CartpoleScenario`].
//!
//! Run: `cargo run -p clankers-examples --example cartpole_vec_benchmark --release`

use std::time::Instant;

use clankers_core::types::{Action, ActionSpace, ObservationSpace};
use clankers_env::vec_env::VecEnvConfig;
use clankers_env::vec_runner::VecEnvInstance;
use clankers_gym::prelude::{GymEnv, GymVecEnv};
use clankers_sim::ScenarioConfig;
use clankers_sim::scenarios::cartpole::{CartPoleApplicator, CartpoleConfig, CartpoleScenario};

fn make_env() -> GymEnv {
    let mut a =
        CartpoleScenario::build_with(&ScenarioConfig::default(), &CartpoleConfig::default());
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; 4],
        high: vec![10.0; 4],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; 2],
        high: vec![1.0; 2],
    };
    let app = std::mem::take(&mut a.scene.app);
    GymEnv::new(
        app,
        obs_space,
        act_space,
        Box::new(CartPoleApplicator::new(a.layout)),
    )
    .with_reset_fn(CartpoleScenario::reset_world)
}

fn main() {
    let env_counts = [1, 2, 4, 8, 16, 32];
    let steps_per_env: u32 = 500;
    println!("=== Cart-Pole Multi-Environment Benchmark ===\nSteps per env: {steps_per_env}\n");
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; 4],
        high: vec![10.0; 4],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; 2],
        high: vec![1.0; 2],
    };
    for &n in &env_counts {
        let c_start = Instant::now();
        let envs: Vec<Box<dyn VecEnvInstance>> = (0..n)
            .map(|_| Box::new(make_env()) as Box<dyn VecEnvInstance>)
            .collect();
        let c_el = c_start.elapsed();
        let config = VecEnvConfig::new(n as u16);
        let mut vec_env = GymVecEnv::new(envs, config, obs_space.clone(), act_space.clone());
        let r_start = Instant::now();
        vec_env.reset_all(Some(42));
        let r_el = r_start.elapsed();
        let s_start = Instant::now();
        let mut total: u64 = 0;
        for _ in 0..steps_per_env {
            let actions: Vec<Action> = (0..n).map(|_| Action::Continuous(vec![0.5, 0.0])).collect();
            vec_env.step_all(&actions);
            total += n as u64;
        }
        let s_el = s_start.elapsed();
        let sps = total as f64 / s_el.as_secs_f64();
        let ms = s_el.as_secs_f64() * 1000.0 / f64::from(steps_per_env);
        let fcp = vec_env.runner().get_obs(0).as_slice()[0];
        println!(
            "envs={n:3} | create={:.1}s | reset={:.0}ms | step={:.2}s ({sps:.0} sps, {ms:.2}ms/batch) | final_cart_pos={fcp:+.2}",
            c_el.as_secs_f64(),
            r_el.as_secs_f64() * 1000.0,
            s_el.as_secs_f64()
        );
    }
    println!("\nDone.");
}
