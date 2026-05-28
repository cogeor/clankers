//! W7 PR2 integration tests: binary batch frame wire format and
//! `binary_batch` capability handshake.
//!
//! Six tests in this file:
//!
//! - `batch_f32_roundtrip_byte_equal` — encode/decode a `BatchF32`
//!   frame, header fields match, payload bytes (as `f32::to_bits`)
//!   are byte-equal.
//! - `batch_raw_u8_roundtrip_byte_equal` — same for `BatchRawU8Image`.
//! - `binary_frame_header_size_is_24_bytes` — gate item: the
//!   `#[repr(C)]` header is exactly 24 bytes.
//! - `binary_frame_rejects_version_mismatch` — manually crafted frame
//!   with `version: 0xFFFF_FFFF` returns
//!   `BinaryFrameError::UnsupportedVersion`.
//! - `legacy_client_v110_negotiates_without_binary_batch` — Python-like
//!   `1.1.0` client without `binary_batch` opt-in stays on the
//!   JSON-only path.
//! - `client_v120_with_binary_batch_receives_binary_frame` — `1.2.0`
//!   client opting into `binary_batch` receives the JSON envelope plus
//!   the binary frame.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use clankers_core::types::{
    Action, ActionSpace, Observation, ObservationSpace, ResetInfo, ResetResult, StepInfo,
    StepResult,
};
use clankers_env::vec_env::VecEnvConfig;
use clankers_env::vec_runner::VecEnvInstance;
use clankers_gym::binary_frame::{
    BinaryFrameError, BinaryFrameHeader, FRAME_VERSION, KIND_BATCH_F32, KIND_BATCH_RAW_U8,
    decode_batch_f32, decode_batch_raw_u8, encode_batch_f32, encode_batch_raw_u8,
};
use clankers_gym::encoding::{EncodedObservation, ImageLayout};
use clankers_gym::framing;
use clankers_gym::protocol::{Capabilities, Request, Response};
use clankers_gym::{GymVecEnv, VecGymServer};

// ---------------------------------------------------------------------------
// splitmix64 — deterministic pseudo-random stream for byte-equality tests.
// ---------------------------------------------------------------------------

const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// 1. batch_f32_roundtrip_byte_equal
// ---------------------------------------------------------------------------

#[test]
fn batch_f32_roundtrip_byte_equal() {
    const NUM_ENVS: u32 = 8;
    const OBS_DIM: u32 = 17;
    let total = NUM_ENVS as usize * OBS_DIM as usize;

    let mut state = 0xCAFE_BABE_u64;
    let data: Vec<f32> = (0..total)
        .map(|_| f32::from_bits((splitmix64(&mut state) & 0xFFFF_FFFF) as u32))
        .collect();

    let bytes = encode_batch_f32(NUM_ENVS, OBS_DIM, &data);
    let (header, decoded) = decode_batch_f32(&bytes).expect("decode_batch_f32");

    assert_eq!(header.version, FRAME_VERSION);
    assert_eq!(header.kind, KIND_BATCH_F32);
    assert_eq!(header.num_envs, NUM_ENVS);
    assert_eq!(header.dim, OBS_DIM);
    assert_eq!(decoded.len(), data.len());

    // Use to_bits to dodge NaN inequality.
    for (i, (a, b)) in decoded.iter().zip(data.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "f32 mismatch at index {i}");
    }
}

// ---------------------------------------------------------------------------
// 2. batch_raw_u8_roundtrip_byte_equal
// ---------------------------------------------------------------------------

#[test]
fn batch_raw_u8_roundtrip_byte_equal() {
    const NUM_ENVS: u32 = 4;
    const W: u32 = 64;
    const H: u32 = 64;
    const C: u8 = 3;
    let total = NUM_ENVS as usize * W as usize * H as usize * C as usize;

    let mut state = 0xDEAD_BEEF_u64;
    let data: Vec<u8> = (0..total)
        .map(|_| (splitmix64(&mut state) & 0xFF) as u8)
        .collect();

    let bytes = encode_batch_raw_u8(NUM_ENVS, W, H, C, ImageLayout::Hwc, &data);
    let (header, decoded) = decode_batch_raw_u8(&bytes).expect("decode_batch_raw_u8");

    assert_eq!(header.version, FRAME_VERSION);
    assert_eq!(header.kind, KIND_BATCH_RAW_U8);
    assert_eq!(header.num_envs, NUM_ENVS);
    // dim encodes the per-env tile size W * H * C.
    assert_eq!(header.dim, W * H * u32::from(C));
    assert_eq!(decoded.len(), data.len());
    assert_eq!(decoded, data.as_slice());
}

// ---------------------------------------------------------------------------
// 3. binary_frame_header_size_is_24_bytes (gate item)
// ---------------------------------------------------------------------------

#[test]
fn binary_frame_header_size_is_24_bytes() {
    assert_eq!(std::mem::size_of::<BinaryFrameHeader>(), 24);
}

// ---------------------------------------------------------------------------
// 4. binary_frame_rejects_version_mismatch
// ---------------------------------------------------------------------------

#[test]
fn binary_frame_rejects_version_mismatch() {
    // Manually craft a 24-byte header with version = 0xFFFF_FFFF + a
    // 4-byte f32 payload.
    let mut bytes = Vec::with_capacity(24 + 4);
    bytes.extend_from_slice(&0xFFFF_FFFF_u32.to_le_bytes()); // version
    bytes.push(KIND_BATCH_F32); // kind
    bytes.extend_from_slice(&[0u8; 3]); // _pad
    bytes.extend_from_slice(&1u32.to_le_bytes()); // num_envs = 1
    bytes.extend_from_slice(&1u32.to_le_bytes()); // dim = 1
    bytes.extend_from_slice(&[0u8; 8]); // _reserved = [0; 2] (8 bytes)
    // Header sanity: 4 + 1 + 3 + 4 + 4 + 8 = 24.
    assert_eq!(bytes.len(), 24);
    // Payload: 1 f32 = 4 bytes.
    bytes.extend_from_slice(&1.0_f32.to_le_bytes());

    let err = decode_batch_f32(&bytes).expect_err("must reject unsupported version");
    match err {
        BinaryFrameError::UnsupportedVersion { got } => assert_eq!(got, 0xFFFF_FFFF),
        other => panic!("expected UnsupportedVersion, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 5–6. Handshake tests against a live VecGymServer
// ---------------------------------------------------------------------------

/// Minimal `VecEnvInstance` returning a constant observation of length
/// `obs_dim`. Mirrors `clankers-gym::server::tests::ConstVecEnvInstance`.
struct ConstVecEnvInstance {
    obs_dim: usize,
}

impl VecEnvInstance for ConstVecEnvInstance {
    fn reset(&mut self, _seed: Option<u64>) -> ResetResult {
        ResetResult {
            observation: Observation::zeros(self.obs_dim),
            info: ResetInfo::default(),
        }
    }

    fn step(&mut self, _action: &Action) -> StepResult {
        StepResult {
            observation: Observation::zeros(self.obs_dim),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info: StepInfo::default(),
        }
    }

    fn obs_dim(&self) -> usize {
        self.obs_dim
    }
}

fn build_test_vec_env(num_envs: usize, obs_dim: usize) -> GymVecEnv {
    let envs: Vec<Box<dyn VecEnvInstance>> = (0..num_envs)
        .map(|_| Box::new(ConstVecEnvInstance { obs_dim }) as Box<dyn VecEnvInstance>)
        .collect();
    let config = VecEnvConfig::new(u16::try_from(num_envs).expect("too many envs"));
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; obs_dim],
        high: vec![1.0; obs_dim],
    };
    GymVecEnv::new(envs, config, obs_space, act_space)
}

#[test]
fn legacy_client_v110_negotiates_without_binary_batch() {
    // Spin up a VecGymServer (server advertises binary_batch by default).
    let server = VecGymServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();

    let handle = std::thread::spawn(move || {
        let mut env = build_test_vec_env(2, 2);
        server.serve_one(&mut env).unwrap();
    });

    let mut stream = TcpStream::connect(addr).unwrap();

    // Client says protocol_version=1.1.0, capabilities EMPTY (no
    // binary_batch opt-in). This is the legacy Python client behaviour.
    let init = Request::Init {
        protocol_version: "1.1.0".into(),
        client_name: "legacy_test".into(),
        client_version: "0.1.0".into(),
        capabilities: Capabilities::default(),
        seed: None,
    };
    framing::write_message(&mut stream, &init).unwrap();
    let resp: Response = framing::read_message(&mut stream).unwrap().unwrap();
    match resp {
        Response::InitResponse {
            protocol_version,
            capabilities,
            ..
        } => {
            // Server negotiates down to 1.1.0.
            assert_eq!(protocol_version, "1.1.0");
            // Client did not request binary_batch, so the AND-product
            // for that key is absent (or false). The PLAN spec says
            // capabilities.get("binary_batch") == None for legacy
            // clients (the empty map produces no entries).
            assert!(
                !capabilities.binary_batch,
                "binary_batch must NOT be active for legacy client"
            );
        }
        other => panic!("expected InitResponse, got {other:?}"),
    }

    // BatchReset: must yield Response::BatchReset { obs_encoding: None, .. }
    // and NO follow-up binary frame.
    framing::write_message(
        &mut stream,
        &Request::BatchReset {
            env_ids: vec![0, 1],
            seeds: None,
        },
    )
    .unwrap();
    let resp: Response = framing::read_message(&mut stream).unwrap().unwrap();
    match resp {
        Response::BatchReset {
            observations,
            obs_encoding,
            ..
        } => {
            assert_eq!(observations.len(), 2);
            // Inline observations carry the data (no binary path).
            assert_eq!(observations[0].len(), 2);
            assert!(
                obs_encoding.is_none(),
                "obs_encoding must be None on legacy JSON path"
            );
        }
        other => panic!("expected BatchReset, got {other:?}"),
    }

    // No follow-up binary frame: read with a short timeout, expect
    // either WouldBlock or EOF after Close.
    stream
        .set_read_timeout(Some(Duration::from_millis(50)))
        .unwrap();
    let mut buf = [0u8; 1];
    match stream.read(&mut buf) {
        Ok(0) => {} // EOF — clean.
        Ok(_) => panic!("unexpected bytes on wire after JSON BatchReset"),
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ) => {} // expected
        Err(e) => panic!("unexpected I/O error: {e}"),
    }
    stream.set_read_timeout(None).unwrap();

    // Close cleanly.
    framing::write_message(&mut stream, &Request::Close).unwrap();
    let _close: Response = framing::read_message(&mut stream).unwrap().unwrap();
    drop(stream);
    handle.join().unwrap();
}

#[test]
fn client_v120_with_binary_batch_receives_binary_frame() {
    let server = VecGymServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();

    let handle = std::thread::spawn(move || {
        let mut env = build_test_vec_env(2, 2);
        server.serve_one(&mut env).unwrap();
    });

    let mut stream = TcpStream::connect(addr).unwrap();

    // 1.2.0 client opting into binary_batch.
    let init = Request::Init {
        protocol_version: "1.2.0".into(),
        client_name: "binary_batch_test".into(),
        client_version: "0.1.0".into(),
        capabilities: Capabilities {
            binary_batch: true,
            ..Default::default()
        },
        seed: None,
    };
    framing::write_message(&mut stream, &init).unwrap();
    let resp: Response = framing::read_message(&mut stream).unwrap().unwrap();
    match resp {
        Response::InitResponse {
            protocol_version,
            capabilities,
            ..
        } => {
            assert_eq!(protocol_version, "1.2.0");
            assert!(
                capabilities.binary_batch,
                "binary_batch should be negotiated true (client and server both true)"
            );
        }
        other => panic!("expected InitResponse, got {other:?}"),
    }

    // BatchReset: must yield Response::BatchReset { obs_encoding: Some(BatchF32 { .. }), .. }
    // and a follow-up binary frame.
    framing::write_message(
        &mut stream,
        &Request::BatchReset {
            env_ids: vec![0, 1],
            seeds: None,
        },
    )
    .unwrap();
    let resp: Response = framing::read_message(&mut stream).unwrap().unwrap();
    match resp {
        Response::BatchReset {
            observations,
            obs_encoding,
            ..
        } => {
            // Empty sentinel observations per from_batch_reset_binary contract.
            assert_eq!(observations.len(), 2);
            assert!(observations.iter().all(Observation::is_empty));
            let enc = obs_encoding.expect("obs_encoding must be Some on binary batch path");
            match enc {
                EncodedObservation::BatchF32 {
                    num_envs,
                    obs_dim,
                    payload,
                } => {
                    assert_eq!(num_envs, 2);
                    assert_eq!(obs_dim, 2);
                    // payload is #[serde(skip)] — empty on the JSON side.
                    assert!(payload.is_empty());
                }
                other => panic!("expected BatchF32, got {other:?}"),
            }
        }
        other => panic!("expected BatchReset, got {other:?}"),
    }

    // Read the follow-up binary frame.
    let bytes = framing::read_binary_frame(&mut stream).unwrap();
    let (header, flat) = decode_batch_f32(&bytes).expect("decode_batch_f32");
    assert_eq!(header.num_envs, 2);
    assert_eq!(header.dim, 2);
    assert_eq!(flat.len(), 4);
    // ConstVecEnvInstance::reset emits zeros.
    assert!(flat.iter().all(|&v| v == 0.0));

    // Close cleanly.
    framing::write_message(&mut stream, &Request::Close).unwrap();
    let _close: Response = framing::read_message(&mut stream).unwrap().unwrap();
    drop(stream);
    handle.join().unwrap();

    // Touch `Write` import so it isn't dead.
    let _ = std::io::sink().write(&[]);
}
