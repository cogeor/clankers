//! Integration tests for the clankers-record MCAP recording pipeline.
//!
//! These tests exercise the Recorder API end-to-end: opening an MCAP file,
//! writing frames with known data, closing the file, then reading it back
//! with `mcap::MessageStream` to verify correctness.

use std::collections::HashMap;
use std::path::PathBuf;

use clankers_record::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a unique temp file path for a test MCAP file.
fn temp_mcap_path(label: &str) -> PathBuf {
    let id = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("clankers_test_{label}_{id}_{ts}.mcap"))
}

/// Open a Recorder and register the standard three channels.
/// Returns `(recorder, joints_ch, action_ch, reward_ch)`.
fn open_with_channels(path: &PathBuf) -> (Recorder, u16, u16, u16) {
    let mut recorder = Recorder::open(path).expect("open recorder");
    let schema_id = recorder.register_schema().expect("register schema");
    let joints_ch = recorder
        .add_channel(schema_id, "/joint_states")
        .expect("add /joint_states");
    let action_ch = recorder
        .add_channel(schema_id, "/actions")
        .expect("add /actions");
    let reward_ch = recorder
        .add_channel(schema_id, "/reward")
        .expect("add /reward");
    (recorder, joints_ch, action_ch, reward_ch)
}

/// Build a `JointFrame` with 2 joints and a given index for deterministic data.
#[allow(clippy::cast_precision_loss)] // test indices are tiny
fn make_joint_frame(i: u32) -> JointFrame {
    let fi = i as f32;
    JointFrame {
        timestamp_ns: (u64::from(i) + 1) * 1_000_000,
        names: vec!["shoulder".to_string(), "elbow".to_string()],
        positions: vec![0.1 * fi, -0.1 * fi],
        velocities: vec![0.01 * fi, 0.02 * fi],
        torques: vec![1.0 * fi, -0.5 * fi],
    }
}

/// Build an `ActionFrame` with 3 action dims.
#[allow(clippy::cast_precision_loss)]
fn make_action_frame(i: u32) -> ActionFrame {
    let fi = i as f32;
    ActionFrame {
        timestamp_ns: (u64::from(i) + 1) * 1_000_000,
        data: vec![0.5 * fi, -0.5 * fi, 0.0],
    }
}

/// Build a `RewardFrame`.
#[allow(clippy::cast_precision_loss)]
fn make_reward_frame(i: u32) -> RewardFrame {
    RewardFrame {
        timestamp_ns: (u64::from(i) + 1) * 1_000_000,
        reward: i as f32 * 0.1,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Write 10 of each frame type, finish the file, read it back and verify
/// message counts, topics, and payload deserialization.
#[test]
fn write_and_verify_mcap() {
    let path = temp_mcap_path("write_verify");

    // -- Write phase --------------------------------------------------------
    {
        let (mut recorder, joints_ch, action_ch, reward_ch) = open_with_channels(&path);

        for i in 0..10u32 {
            recorder
                .write_joint_frame(joints_ch, &make_joint_frame(i))
                .expect("write joint frame");
            recorder
                .write_action_frame(action_ch, &make_action_frame(i))
                .expect("write action frame");
            recorder
                .write_reward_frame(reward_ch, &make_reward_frame(i))
                .expect("write reward frame");
        }

        recorder.finish().expect("finish");
    }

    // -- Read phase ---------------------------------------------------------
    let data = std::fs::read(&path).expect("read mcap file");
    let stream = mcap::MessageStream::new(&data).expect("parse mcap");

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut joint_frames: Vec<JointFrame> = Vec::new();
    let mut action_frames: Vec<ActionFrame> = Vec::new();
    let mut reward_frames: Vec<RewardFrame> = Vec::new();

    for msg_result in stream {
        let msg = msg_result.expect("read message");
        let topic = msg.channel.topic.clone();
        *counts.entry(topic.clone()).or_insert(0) += 1;

        match topic.as_str() {
            "/joint_states" => {
                let frame: JointFrame =
                    serde_json::from_slice(&msg.data).expect("deserialize JointFrame");
                joint_frames.push(frame);
            }
            "/actions" => {
                let frame: ActionFrame =
                    serde_json::from_slice(&msg.data).expect("deserialize ActionFrame");
                action_frames.push(frame);
            }
            "/reward" => {
                let frame: RewardFrame =
                    serde_json::from_slice(&msg.data).expect("deserialize RewardFrame");
                reward_frames.push(frame);
            }
            other => panic!("unexpected topic: {other}"),
        }
    }

    // Verify counts.
    assert_eq!(
        counts.get("/joint_states"),
        Some(&10),
        "expected 10 /joint_states messages"
    );
    assert_eq!(
        counts.get("/actions"),
        Some(&10),
        "expected 10 /actions messages"
    );
    assert_eq!(
        counts.get("/reward"),
        Some(&10),
        "expected 10 /reward messages"
    );

    // Verify payload content for a few frames.
    assert_eq!(joint_frames.len(), 10);
    assert_eq!(joint_frames[0], make_joint_frame(0));
    assert_eq!(joint_frames[9], make_joint_frame(9));

    assert_eq!(action_frames.len(), 10);
    assert_eq!(action_frames[0], make_action_frame(0));
    assert_eq!(action_frames[9], make_action_frame(9));

    assert_eq!(reward_frames.len(), 10);
    assert_eq!(reward_frames[0], make_reward_frame(0));
    assert_eq!(reward_frames[9], make_reward_frame(9));

    // Verify joint frame structure.
    assert_eq!(joint_frames[5].names.len(), 2);
    assert_eq!(joint_frames[5].names[0], "shoulder");
    assert_eq!(joint_frames[5].names[1], "elbow");

    // Clean up.
    let _ = std::fs::remove_file(&path);
}

/// Dropping a `Recorder` without calling `finish()` should still produce a valid
/// (readable) MCAP file, because the `Drop` impl calls `finish()`.
#[test]
fn recorder_drop_finalizes() {
    let path = temp_mcap_path("drop_finalize");

    {
        let (mut recorder, joints_ch, _action_ch, _reward_ch) = open_with_channels(&path);
        recorder
            .write_joint_frame(joints_ch, &make_joint_frame(0))
            .expect("write one frame");
        // Drop without calling finish().
    }

    // File should exist and be readable.
    assert!(path.exists(), "mcap file should exist after drop");
    let data = std::fs::read(&path).expect("read mcap file");
    let stream = mcap::MessageStream::new(&data).expect("parse mcap after drop");

    let count = stream.flatten().count();
    assert_eq!(count, 1, "expected 1 message after drop-finalize");

    let _ = std::fs::remove_file(&path);
}

/// Opening a Recorder and finishing immediately should produce a valid MCAP
/// file with zero messages.
#[test]
fn empty_recording() {
    let path = temp_mcap_path("empty");

    {
        let (mut recorder, _joints_ch, _action_ch, _reward_ch) = open_with_channels(&path);
        recorder.finish().expect("finish empty recording");
    }

    assert!(path.exists(), "mcap file should exist");
    let data = std::fs::read(&path).expect("read mcap file");
    let stream = mcap::MessageStream::new(&data).expect("parse empty mcap");

    let count = stream.flatten().count();
    assert_eq!(count, 0, "expected 0 messages in empty recording");

    let _ = std::fs::remove_file(&path);
}

/// Verify JSON serialization round-trip for all three frame types.
#[test]
fn frame_serialization_roundtrip() {
    // JointFrame
    let joint = make_joint_frame(7);
    let json = serde_json::to_vec(&joint).expect("serialize JointFrame");
    let decoded: JointFrame = serde_json::from_slice(&json).expect("deserialize JointFrame");
    assert_eq!(joint, decoded);

    // ActionFrame
    let action = make_action_frame(3);
    let json = serde_json::to_vec(&action).expect("serialize ActionFrame");
    let decoded: ActionFrame = serde_json::from_slice(&json).expect("deserialize ActionFrame");
    assert_eq!(action, decoded);

    // RewardFrame
    let reward = make_reward_frame(5);
    let json = serde_json::to_vec(&reward).expect("serialize RewardFrame");
    let decoded: RewardFrame = serde_json::from_slice(&json).expect("deserialize RewardFrame");
    assert_eq!(reward, decoded);
}
