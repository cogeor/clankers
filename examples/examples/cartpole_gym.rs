//! Cart-pole gym server. Thin wrapper over
//! [`clankers_sim::scenarios::cartpole::CartpoleScenario`].
//!
//! Run: `cargo run -p clankers-examples --example cartpole_gym`

use clankers_core::types::{ActionSpace, ObservationSpace};
use clankers_gym::prelude::{GymEnv, GymServer};
use clankers_sim::ScenarioConfig;
use clankers_sim::scenarios::cartpole::{CartPoleApplicator, CartpoleConfig, CartpoleScenario};

fn main() {
    println!("=== Cart-Pole Gym Server (with Rapier Physics) ===\n");
    let address = "127.0.0.1:9877";
    let mut artefacts =
        CartpoleScenario::build_with(&ScenarioConfig::default(), &CartpoleConfig::default());
    println!("Robot: {}", artefacts.model.name);
    println!("DOF:   {}", artefacts.model.dof());
    let obs_dim = 4usize;
    let act_dim = 2usize;
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; act_dim],
        high: vec![1.0; act_dim],
    };
    let app = std::mem::take(&mut artefacts.scene.app);
    let mut env = GymEnv::new(
        app,
        obs_space,
        act_space,
        Box::new(CartPoleApplicator::new(artefacts.layout.clone())),
    )
    .with_reset_fn(CartpoleScenario::reset_world);
    let server = GymServer::bind(address).expect("bind");
    let addr = server.local_addr().expect("addr");
    println!("\nCart-pole gym server listening on {addr}");
    println!("joints={act_dim}, obs_dim={obs_dim}, act_dim={act_dim}, max_steps=500");
    println!("Connect with: python python/examples/cartpole_read_state.py\n");
    loop {
        println!("waiting for client...");
        match server.serve_one(&mut env) {
            Ok(()) => println!("client disconnected cleanly"),
            Err(e) => eprintln!("client error: {e}"),
        }
    }
}
