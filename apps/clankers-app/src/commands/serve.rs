//! `clankers-app serve` — TCP gym server.
//!
//! In W5 PR1, the legacy `Serve` variant body lives in
//! [`execute_default`] (preserving the pre-PR1 behaviour). The new
//! scenario-aware [`execute`] (`--scenario <name>`) lands in W5 PR2
//! (loop 6 of the W3/W4/W5 orchestration).

use std::sync::Arc;

use bevy::prelude::*;
use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
use clankers_core::layout::{
    JointKind, JointLayout, JointLayoutBuilder, JointSpec, JointSpecLimits,
};
use clankers_core::prelude::*;
use clankers_env::prelude::*;
use clankers_sim::ClankersSimPlugin;

/// Scenario-driven gym server (`clankers-app serve --scenario <name>`).
///
/// Implemented in W5 PR2 (loop 6 of `20260526-013019-w3-w4-w5-impl`).
#[allow(dead_code)]
pub fn execute() -> ! {
    unimplemented!("WS5 PR2 — see .delegate/work/20260526-013019-w3-w4-w5-impl/06");
}

/// Legacy `Serve` mode body — synthetic-layout N-joint demo.
/// Preserved verbatim from the pre-PR1 `apps/clankers-app/src/main.rs::run_serve`.
pub fn execute_default(address: &str, num_joints: usize, max_steps: u32) {
    use clankers_gym::prelude::*;

    // Build the Bevy app for the gym environment
    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);

    // Spawn N joint entities and collect their ids for the layout.
    let joint_entities: Vec<Entity> = (0..num_joints)
        .map(|_| {
            app.world_mut()
                .spawn((
                    Actuator::default(),
                    JointCommand::default(),
                    JointState::default(),
                    JointTorque::default(),
                ))
                .id()
        })
        .collect();

    // Build a synthetic layout — no URDF in this demo path.
    let layout = synthetic_layout(&joint_entities);

    // Register a sensor
    {
        let world = app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(layout.clone())), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    app.world_mut()
        .resource_mut::<EpisodeConfig>()
        .max_episode_steps = max_steps;

    let obs_dim = num_joints * 2; // position + velocity
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; num_joints],
        high: vec![1.0; num_joints],
    };

    let mut env = GymEnv::new(
        app,
        obs_space,
        act_space,
        Box::new(JointCommandApplicator { layout }),
    );

    let server = GymServer::bind(address).expect("failed to bind server");
    let addr = server.local_addr().expect("failed to get address");
    println!("clankers gym server listening on {addr}");
    println!("joints={num_joints}, obs_dim={obs_dim}, act_dim={num_joints}, max_steps={max_steps}");

    loop {
        println!("waiting for client...");
        match server.serve_one(&mut env) {
            Ok(()) => println!("client disconnected cleanly"),
            Err(e) => eprintln!("client error: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// JointCommandApplicator
// ---------------------------------------------------------------------------

/// Default action applicator that writes action values to joint commands
/// in layout slot order.
struct JointCommandApplicator {
    layout: Arc<JointLayout>,
}

impl ActionApplicator for JointCommandApplicator {
    fn apply(&self, world: &mut World, action: &Action) {
        let values = action
            .as_continuous()
            .expect("ActionApplicator contract: continuous action expected");
        for (i, entity) in self.layout.bound_entities().enumerate() {
            if i >= values.len() {
                break;
            }
            if let Some(mut cmd) = world.get_mut::<JointCommand>(entity) {
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

/// Build a synthetic [`JointLayout`] of `n` revolute joints, bound to the
/// supplied entity ids in spawn order. Used by the CLI demo, which
/// spawns N raw joint entities without a URDF.
fn synthetic_layout(entities: &[Entity]) -> Arc<JointLayout> {
    let mut builder = JointLayoutBuilder::default();
    for (i, _) in entities.iter().enumerate() {
        builder = builder.push(JointSpec {
            name: format!("joint_{i}"),
            entity: None,
            joint_type: JointKind::Revolute,
            limits: JointSpecLimits::default(),
            axis: [0.0, 0.0, 1.0],
        });
    }
    let mut layout = builder.build();
    layout.bind_entities(entities);
    Arc::new(layout)
}
