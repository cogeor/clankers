//! Integration test for `clankers-app serve --protocol binary` (W5
//! PR2, loop 6 gate item 4 / WS5-plan § 6).
//!
//! Per loop 06 PLAN Design choice E, this test uses an **in-process**
//! server thread rather than spawning the CLI subprocess. The CLI
//! argument parsing is exercised separately by `cli_run_scenario.rs`;
//! the value here is verifying the binary-protocol negotiation +
//! round-trip works end-to-end against the `cartpole` scenario built
//! through the public `ScenarioBuilder` API.

use std::net::TcpStream;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use bevy::prelude::App;
use clankers_core::layout::JointLayout;
use clankers_core::traits::ActionApplicator;
use clankers_core::types::{Action, ActionSpace, ObservationSpace};
use clankers_gym::env::GymEnv;
use clankers_gym::framing;
use clankers_gym::protocol::{Capabilities, Request, Response};
use clankers_gym::server::GymServer;
use clankers_sim::scenarios::cartpole::CartpoleScenario;
use clankers_sim::{ClankersSimPlugin, ScenarioBuilder, ScenarioConfig};

/// Action applicator that writes action values to `JointCommand` in
/// layout slot order. A copy of the production `JointCommandApplicator`
/// in `apps/clankers-app/src/commands/serve.rs` — kept here so the
/// integration test doesn't depend on a `pub(crate)` symbol.
struct JointCommandApplicator {
    layout: Arc<JointLayout>,
}

impl ActionApplicator for JointCommandApplicator {
    fn apply(&self, world: &mut bevy::prelude::World, action: &Action) {
        let values = action
            .as_continuous()
            .expect("ActionApplicator contract: continuous action expected");
        for (i, entity) in self.layout.bound_entities().enumerate() {
            if i >= values.len() {
                break;
            }
            if let Some(mut cmd) =
                world.get_mut::<clankers_actuator::components::JointCommand>(entity)
            {
                cmd.value = values[i];
            }
        }
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "JointCommandApplicator"
    }

    fn layout(&self) -> &JointLayout {
        &self.layout
    }
}

fn build_cartpole_env() -> GymEnv {
    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);
    let handle = CartpoleScenario.build(&mut app, &ScenarioConfig::default());
    app.finish();
    app.cleanup();
    let layout = handle
        .layout
        .expect("cartpole scenario must expose a layout");
    let obs_dim = layout.len() * 2; // 2 joints * (pos + vel) = 4
    let act_dim = layout.len();
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; act_dim],
        high: vec![1.0; act_dim],
    };
    GymEnv::new(
        app,
        obs_space,
        act_space,
        Box::new(JointCommandApplicator { layout }),
    )
}

#[test]
fn serve_binary_protocol_round_trips_observation() {
    // Spin up a single-use server on an ephemeral port.
    let server = GymServer::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = server.local_addr().expect("read local addr");

    let server_thread = thread::spawn(move || {
        let mut env = build_cartpole_env();
        server.serve_one(&mut env).expect("server serve_one");
    });

    // Connect; give the server a moment to enter `accept()` if the
    // OS hasn't yet scheduled the thread.
    let mut stream = loop {
        match TcpStream::connect(addr) {
            Ok(s) => break s,
            Err(_) => thread::sleep(Duration::from_millis(10)),
        }
    };

    // 1. Init with binary_obs=true so the server marks the session as
    //    binary-capable. For a flat (non-image) observation space, the
    //    server still encodes via `Response::from_reset` (no
    //    `obs_encoding` field) — the binary path only fires for
    //    `ObservationSpace::Image`. We assert the negotiation succeeds
    //    and the flat observation round-trips correctly.
    let caps = Capabilities {
        binary_obs: true,
        ..Default::default()
    };
    let init = Request::Init {
        protocol_version: "1.0.0".into(),
        client_name: "test".into(),
        client_version: "0.1.0".into(),
        capabilities: caps,
        seed: Some(7),
    };
    framing::write_message(&mut stream, &init).expect("write Init");
    let init_resp: Response = framing::read_message(&mut stream)
        .expect("read InitResponse")
        .expect("non-empty InitResponse");
    match &init_resp {
        Response::InitResponse {
            capabilities,
            seed_accepted,
            env_info,
            ..
        } => {
            assert!(
                capabilities.binary_obs,
                "binary_obs must be negotiated to true"
            );
            assert!(*seed_accepted, "seed must be acknowledged");
            // cartpole = 2 joints * 2 = 4-dim observation space.
            match &env_info.observation_space {
                ObservationSpace::Box { low, high } => {
                    assert_eq!(low.len(), 4, "cartpole obs dim = 4");
                    assert_eq!(high.len(), 4);
                }
                other => panic!("unexpected obs space: {other:?}"),
            }
        }
        other => panic!("expected InitResponse, got {other:?}"),
    }

    // 2. Reset — confirms the observation flows back correctly under
    //    the negotiated session.
    framing::write_message(&mut stream, &Request::Reset { seed: Some(7) }).expect("write Reset");
    let reset_resp: Response = framing::read_message(&mut stream)
        .expect("read Reset response")
        .expect("non-empty Reset response");
    match &reset_resp {
        Response::Reset {
            observation,
            obs_encoding,
            ..
        } => {
            // Cartpole flat path: obs_encoding stays None because the
            // server's dispatch only emits Some(_) for the image
            // binary path. The flat observation rides the JSON field.
            assert!(
                obs_encoding.is_none(),
                "flat (non-image) reset response should not carry obs_encoding"
            );
            assert_eq!(
                observation.dim(),
                4,
                "cartpole reset observation dim = 4 (cart_pos, cart_vel, pole_pos, pole_vel)"
            );
        }
        other => panic!("expected Reset, got {other:?}"),
    }

    // 3. Close cleanly so the server thread exits.
    framing::write_message(&mut stream, &Request::Close).expect("write Close");
    let _: Response = framing::read_message(&mut stream)
        .expect("read Close ack")
        .expect("non-empty Close ack");
    drop(stream);
    server_thread.join().expect("server thread joined");
}
