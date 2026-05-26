//! Vectorized cart-pole gym server. Thin wrapper over
//! [`clankers_sim::scenarios::cartpole::CartpoleScenario`].
//!
//! Run: `cargo run -p clankers-examples --bin cartpole_vec_gym -- 8`

use std::env;

use clankers_core::types::{Action, ActionSpace, ObservationSpace};
use clankers_env::vec_env::VecEnvConfig;
use clankers_env::vec_runner::VecEnvInstance;
use clankers_gym::prelude::{GymEnv, GymVecEnv, VecGymServer};
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
    let num_envs: usize = env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let address = "127.0.0.1:9878";
    println!("=== Vectorized Cart-Pole Gym Server ===\nCreating {num_envs} envs with Rapier...");
    let envs: Vec<Box<dyn VecEnvInstance>> = (0..num_envs)
        .map(|i| {
            let e = make_env();
            println!("  env {i}: ready");
            Box::new(e) as Box<dyn VecEnvInstance>
        })
        .collect();
    let config = VecEnvConfig::new(num_envs as u16);
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; 4],
        high: vec![10.0; 4],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; 2],
        high: vec![1.0; 2],
    };
    let mut vec_env = GymVecEnv::new(envs, config, obs_space, act_space);
    let _ = std::mem::size_of::<Action>();
    let server = VecGymServer::bind(address).expect("bind");
    println!(
        "\nVec cart-pole gym server listening on {}\nnum_envs={num_envs}, obs_dim=4, act_dim=2",
        server.local_addr().expect("addr")
    );
    loop {
        println!("waiting for client...");
        match server.serve_one(&mut vec_env) {
            Ok(()) => println!("client disconnected cleanly"),
            Err(e) => eprintln!("client error: {e}"),
        }
    }
}
