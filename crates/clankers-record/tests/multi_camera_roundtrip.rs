//! Integration tests for the W6 PR1 multi-camera recorder pipeline.
//!
//! Exercises the recorder end-to-end with two camera labels and asserts
//! that channel topics surface through the MCAP reader exactly as
//! [`clankers_record::schema::camera_topic`] builds them. The Python
//! loop-02 rewrite consumes the same constants; these tests pin the
//! Rust half of the WS6 finding #3 contract (no more
//! `/camera/image` hard-coding).
//!
//! Reader pattern is `std::fs::read` + `mcap::MessageStream::new(&[u8])`
//! per the workspace convention (see
//! `crates/clankers-record/tests/recording_pipeline.rs`). The PLAN
//! deviates from WS6-plan § 5's `mcap::reader::make_reader` for
//! consistency with the established workspace pattern; the assertions
//! here (channel-set discovery) are satisfied by `MessageStream`
//! iteration because every camera channel has ≥1 message.

#![cfg(feature = "camera")]

use std::collections::HashSet;
use std::path::PathBuf;

use bevy::MinimalPlugins;
use bevy::prelude::*;

use clankers_core::time::SimTime;
use clankers_record::prelude::*;
use clankers_render::buffer::{CameraFrameBuffers, FrameBuffer};
use clankers_render::config::PixelFormat;

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

/// Seed `CameraFrameBuffers` with one frame per camera so the
/// recorder's `PostUpdate` writes at least one record per channel
/// during a single `app.update()`.
fn seed_camera_buffers(labels: &[&str]) -> CameraFrameBuffers {
    let mut buffers = CameraFrameBuffers::default();
    for label in labels {
        let mut buf = FrameBuffer::new(8, 8, PixelFormat::Rgb8);
        // Fill with the label's first byte so different labels produce
        // visibly distinct payloads (helps diagnose mix-ups).
        let fill = label.as_bytes()[0];
        buf.write_frame(vec![fill; 8 * 8 * 3]);
        buffers.insert((*label).to_string(), buf);
    }
    buffers
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Drive the recorder through `RecorderPlugin` with two camera labels
/// and assert that every camera channel discovered via
/// `mcap::MessageStream` matches a topic built by
/// [`clankers_record::schema::camera_topic`].
#[test]
fn recorder_writes_per_label_topics() {
    let path = temp_mcap_path("multi_camera");
    {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(SimTime::new());
        app.insert_resource(seed_camera_buffers(&["front", "wrist"]));
        app.insert_resource(RecordingConfig {
            output_path: path.clone(),
            record_joints: false,
            record_actions: false,
            record_rewards: false,
            record_body_poses: false,
        });
        app.add_plugins(RecorderPlugin);
        app.finish();
        app.cleanup();
        app.update();
        // Drop the app to flush the MCAP footer via Recorder::Drop.
    }

    let data = std::fs::read(&path).expect("read mcap file");
    let stream = mcap::MessageStream::new(&data).expect("parse mcap");

    let mut topics: HashSet<String> = HashSet::new();
    for msg_result in stream {
        let msg = msg_result.expect("read message");
        topics.insert(msg.channel.topic.clone());
    }

    // Filter to camera-only topics so any non-camera channel (e.g. a
    // future metadata record) does not break this assertion.
    let camera_topics: HashSet<String> = topics
        .iter()
        .filter(|t| t.starts_with(clankers_record::schema::CAMERA_TOPIC_PREFIX))
        .cloned()
        .collect();

    let expected: HashSet<String> = [
        clankers_record::schema::camera_topic("front"),
        clankers_record::schema::camera_topic("wrist"),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        camera_topics, expected,
        "expected channel topics {expected:?}, found {camera_topics:?}",
    );

    let _ = std::fs::remove_file(&path);
}

/// Pin every public topic constant to its literal wire string so the
/// recorder hot path cannot drift from the doc table in
/// `crates/clankers-record/src/recorder.rs:18`.
#[test]
fn recorder_topic_constants_match_doc_table() {
    assert_eq!(clankers_record::schema::JOINT_STATES_TOPIC, "/joint_states");
    assert_eq!(clankers_record::schema::ACTIONS_TOPIC, "/actions");
    assert_eq!(clankers_record::schema::REWARD_TOPIC, "/reward");
    assert_eq!(clankers_record::schema::BODY_POSES_TOPIC, "/body_poses");
    assert_eq!(clankers_record::schema::CAMERA_TOPIC_PREFIX, "/camera/");
}

/// `camera_topic(label)` is the single source of truth — verify the
/// prefix / suffix invariants hold for representative labels.
#[test]
fn camera_topic_label_round_trip() {
    let t = clankers_record::schema::camera_topic("front");
    assert_eq!(t, "/camera/front");
    assert!(t.starts_with(clankers_record::schema::CAMERA_TOPIC_PREFIX));
    assert!(t.ends_with("front"));

    // Edge labels — empty and uncommon characters — still concatenate
    // cleanly (the recorder is the only consumer; we do not need to
    // validate label content here).
    assert_eq!(
        clankers_record::schema::camera_topic(""),
        clankers_record::schema::CAMERA_TOPIC_PREFIX
    );
    assert_eq!(
        clankers_record::schema::camera_topic("rear_left"),
        "/camera/rear_left"
    );
}
