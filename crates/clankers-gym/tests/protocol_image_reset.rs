//! W4 PR1 integration test: `Response::Reset` carries
//! `EncodedObservation::RawU8Image` and a follow-up binary frame when
//! the env declares an image observation space and the client
//! negotiated `binary_obs`.
//!
//! Fixes finding #2 in
//! `notes/clankers_codebase_quality_report_2026-05-25.md` — image envs
//! were returning flat-float JSON on reset, breaking the Gymnasium
//! space contract on the first observation.

use std::collections::HashMap;
use std::net::TcpStream;
use std::sync::Arc;

use bevy::prelude::*;

use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
use clankers_core::layout::{JointKind, JointLayoutBuilder, JointSpec, JointSpecLimits};
use clankers_core::traits::{ActionApplicator, ObservationSensor, Sensor};
use clankers_core::types::{Action, ActionSpace, Observation, ObservationSpace};
use clankers_env::prelude::*;
use clankers_gym::encoding::{EncodedObservation, ImageLayout};
use clankers_gym::framing;
use clankers_gym::protocol::{Request, Response};
use clankers_gym::{GymEnv, GymServer};

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

const IMG_W: u32 = 64;
const IMG_H: u32 = 64;
const IMG_C: u32 = 3;

/// Sensor that emits a fixed pattern of `IMG_W * IMG_H * IMG_C` f32
/// pixel values clamped to `[0.0, 1.0]`. The server's
/// `encode_observation` helper multiplies by 255 and clamps to `u8`.
struct ConstImageSensor {
    dim: usize,
}

impl ConstImageSensor {
    const fn new() -> Self {
        Self {
            dim: (IMG_W * IMG_H * IMG_C) as usize,
        }
    }
}

impl Sensor for ConstImageSensor {
    type Output = Observation;

    fn read(&mut self, _world: &mut World) -> Observation {
        // Constant 1.0 pattern -> all-white image (every byte 255).
        Observation::new(vec![1.0_f32; self.dim])
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "ConstImageSensor"
    }
}

impl ObservationSensor for ConstImageSensor {
    fn observation_dim(&self) -> usize {
        self.dim
    }
}

struct NoopApplicator {
    layout: Arc<clankers_core::layout::JointLayout>,
}

impl ActionApplicator for NoopApplicator {
    fn apply(&self, _world: &mut World, _action: &Action) {}

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "NoopApplicator"
    }

    fn layout(&self) -> &clankers_core::layout::JointLayout {
        &self.layout
    }
}

fn build_image_env() -> GymEnv {
    let mut app = App::new();
    app.add_plugins(clankers_core::ClankersCorePlugin);
    app.add_plugins(ClankersEnvPlugin);

    // One joint to satisfy the applicator's layout contract; the
    // applicator is a noop so its action doesn't matter for the test.
    let joint = app
        .world_mut()
        .spawn((
            Actuator::default(),
            JointCommand::default(),
            JointState::default(),
            JointTorque::default(),
        ))
        .id();

    let layout = {
        let mut layout = JointLayoutBuilder::default()
            .push(JointSpec {
                name: "j0".into(),
                entity: None,
                joint_type: JointKind::Revolute,
                limits: JointSpecLimits::default(),
                axis: [0.0, 0.0, 1.0],
            })
            .build();
        layout.bind_entities(&[joint]);
        Arc::new(layout)
    };

    // Register the const image sensor.
    {
        let world = app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(ConstImageSensor::new()), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    let obs_space = ObservationSpace::Image {
        width: IMG_W,
        height: IMG_H,
        channels: IMG_C,
    };
    let act_space = ActionSpace::Discrete { n: 2 };

    GymEnv::new(
        app,
        obs_space,
        act_space,
        Box::new(NoopApplicator { layout }),
    )
}

fn init_request_with_binary_obs() -> Request {
    let mut caps = HashMap::new();
    caps.insert("binary_obs".into(), true);
    Request::Init {
        protocol_version: "1.1.0".into(),
        client_name: "image_reset_test".into(),
        client_version: "0.1.0".into(),
        capabilities: caps,
        seed: None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn protocol_image_reset_returns_raw_u8() {
    let server = GymServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();

    let handle = std::thread::spawn(move || {
        let mut env = build_image_env();
        server.serve_one(&mut env).unwrap();
    });

    let mut stream = TcpStream::connect(addr).unwrap();

    // Handshake — negotiate `binary_obs`.
    framing::write_message(&mut stream, &init_request_with_binary_obs()).unwrap();
    let init: Response = framing::read_message(&mut stream).unwrap().unwrap();
    match init {
        Response::InitResponse {
            capabilities,
            protocol_version,
            ..
        } => {
            // Server advertises binary_obs by default; with the client
            // also requesting it, the negotiated value is true.
            assert_eq!(capabilities.get("binary_obs"), Some(&true));
            assert_eq!(protocol_version, "1.1.0");
        }
        other => panic!("expected InitResponse, got {other:?}"),
    }

    // Reset — must yield Response::Reset { obs_encoding: Some(RawU8Image), .. }
    // plus a follow-up binary frame of exactly W * H * C bytes.
    framing::write_message(&mut stream, &Request::Reset { seed: Some(0) }).unwrap();
    let resp: Response = framing::read_message(&mut stream).unwrap().unwrap();

    match resp {
        Response::Reset {
            observation,
            obs_encoding,
            ..
        } => {
            // Empty sentinel observation per from_reset_binary contract.
            assert_eq!(
                observation.len(),
                0,
                "binary-path Reset has empty inline observation"
            );
            let enc = obs_encoding.expect("obs_encoding must be Some on image-binary reset");
            match enc {
                EncodedObservation::RawU8Image {
                    width,
                    height,
                    channels,
                    layout,
                    payload,
                } => {
                    assert_eq!(width, IMG_W);
                    assert_eq!(height, IMG_H);
                    assert_eq!(channels, 3);
                    assert_eq!(layout, ImageLayout::Hwc);
                    // payload is #[serde(skip)] — never on the wire.
                    assert!(payload.is_empty());
                }
                other => panic!("expected RawU8Image, got {other:?}"),
            }
        }
        other => panic!("expected Reset, got {other:?}"),
    }

    // Read the follow-up binary frame.
    let bytes = framing::read_binary_frame(&mut stream).unwrap();
    assert_eq!(
        bytes.len(),
        (IMG_W * IMG_H * IMG_C) as usize,
        "binary frame must carry exactly W * H * C raw bytes"
    );
    // ConstImageSensor emits 1.0 -> server clamps to [0.0, 1.0] and
    // multiplies by 255 -> every byte must be 0xFF.
    assert!(
        bytes.iter().all(|&b| b == 255),
        "all pixels should be 255 (white) for ConstImageSensor"
    );

    // Close cleanly.
    framing::write_message(&mut stream, &Request::Close).unwrap();
    let _close: Response = framing::read_message(&mut stream).unwrap().unwrap();
    drop(stream);
    handle.join().unwrap();
}

#[test]
fn protocol_image_reset_falls_back_to_flat_without_binary_obs() {
    // Without binary_obs negotiated, the same image env should
    // serialise the flat float observation in the JSON (back-compat
    // with pre-W4 clients). No follow-up binary frame.
    let server = GymServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();

    let handle = std::thread::spawn(move || {
        let mut env = build_image_env();
        server.serve_one(&mut env).unwrap();
    });

    let mut stream = TcpStream::connect(addr).unwrap();

    let req = Request::Init {
        protocol_version: "1.1.0".into(),
        client_name: "image_reset_test".into(),
        client_version: "0.1.0".into(),
        capabilities: HashMap::new(),
        seed: None,
    };
    framing::write_message(&mut stream, &req).unwrap();
    let _init: Response = framing::read_message(&mut stream).unwrap().unwrap();

    framing::write_message(&mut stream, &Request::Reset { seed: Some(0) }).unwrap();
    let resp: Response = framing::read_message(&mut stream).unwrap().unwrap();
    match resp {
        Response::Reset {
            observation,
            obs_encoding,
            ..
        } => {
            // The full flat-float observation is carried inline.
            assert_eq!(observation.len(), (IMG_W * IMG_H * IMG_C) as usize);
            // No binary encoding on the wire.
            assert!(obs_encoding.is_none());
        }
        other => panic!("expected Reset, got {other:?}"),
    }

    framing::write_message(&mut stream, &Request::Close).unwrap();
    let _close: Response = framing::read_message(&mut stream).unwrap().unwrap();
    drop(stream);
    handle.join().unwrap();
}
