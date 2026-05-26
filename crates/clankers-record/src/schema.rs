//! Canonical topic strings and the versioned [`RecorderSchema`] builder
//! consumed by the recorder hot path and downstream loaders.
//!
//! The constants here are the **wire format** for the MCAP channel
//! topics. Cross-process consumers (Python `mcap_loader.py`, the W7 async
//! recorder, any future replay tooling) must observe these exact strings
//! when discovering channels in a recording.
//!
//! Per WS6-plan § 4 the strings live in `clankers-record` because they
//! are owned by the recorder. The W1 schema types
//! ([`clankers_core::schema::RecorderSchema`] and
//! [`clankers_core::schema::FrameSchema`]) live in `clankers-core` so any
//! producer can construct a manifest. This module assembles the manifest
//! that PR1's recorder publishes.
//!
//! # Stability
//!
//! Changing a constant in this module is a breaking change for any
//! recording produced by an older recorder. Bump
//! [`clankers_core::schema::RecorderSchema::SCHEMA_VERSION`] when the
//! channel set or encoding changes; the constants are intentionally
//! plain `&'static str` (no `const fn`) so `format!` callers stay
//! simple.
//!
//! # PR1 scope note
//!
//! The [`recorder_schema`] factory builds the manifest. PR1 does **not**
//! yet attach it to the MCAP file as a metadata record — that is W7 PR4
//! work (async recorder + initial manifest record). The factory exists
//! today so the W6 loop 02 Python loader and the W7 async writer share
//! a single source of truth for the channel set.

use clankers_core::schema::{FrameEncoding, FrameSchema, RecorderSchema};

// ---------------------------------------------------------------------------
// Topic constants
// ---------------------------------------------------------------------------

/// Topic for joint-state frames written by
/// [`crate::recorder::record_joint_states_system`].
pub const JOINT_STATES_TOPIC: &str = "/joint_states";

/// Topic for action frames written by
/// [`crate::recorder::record_action_system`].
pub const ACTIONS_TOPIC: &str = "/actions";

/// Topic for scalar-reward frames written by
/// [`crate::recorder::record_reward_system`].
pub const REWARD_TOPIC: &str = "/reward";

/// Topic for body-pose frames written by
/// [`crate::recorder::record_body_poses_system`].
pub const BODY_POSES_TOPIC: &str = "/body_poses";

/// Prefix shared by every per-camera topic. The full topic for a label
/// `L` is `"/camera/L"` — build it with [`camera_topic`].
pub const CAMERA_TOPIC_PREFIX: &str = "/camera/";

// ---------------------------------------------------------------------------
// Topic builder
// ---------------------------------------------------------------------------

/// Build the canonical MCAP topic for a camera with the given `label`.
///
/// `camera_topic("front")` returns `"/camera/front"`. This is the single
/// source of truth — every Rust caller routes through this function and
/// the Python loader discovers cameras via the
/// [`CAMERA_TOPIC_PREFIX`] glob.
///
/// # Examples
///
/// ```
/// use clankers_record::schema::{camera_topic, CAMERA_TOPIC_PREFIX};
///
/// let topic = camera_topic("wrist");
/// assert_eq!(topic, "/camera/wrist");
/// assert!(topic.starts_with(CAMERA_TOPIC_PREFIX));
/// ```
#[must_use]
pub fn camera_topic(label: &str) -> String {
    format!("{CAMERA_TOPIC_PREFIX}{label}")
}

// ---------------------------------------------------------------------------
// Recorder schema factory
// ---------------------------------------------------------------------------

/// Build the recorder's full [`RecorderSchema`] manifest given a slice of
/// camera labels.
///
/// The returned manifest carries one [`FrameSchema`] per fixed channel
/// (in the order `joint_states`, `actions`, `reward`, `body_poses`)
/// followed by one entry per camera label in the input slice's order.
///
/// Fixed channels are JSON-encoded (matching the recorder hot path).
/// Camera channels are tagged [`FrameEncoding::RawBytes`] — the recorder
/// writes raw pixel data via `application/octet-stream` MCAP channels.
///
/// `message_type` strings (`"JointState"`, `"Action"`, `"Reward"`,
/// `"BodyPose"`, `"Image"`) match the type-name spelling used in
/// `crates/clankers-record/src/types.rs`.
///
/// The `version` field is set to
/// [`RecorderSchema::SCHEMA_VERSION`] (currently `1`).
///
/// # Examples
///
/// ```
/// use clankers_record::schema::recorder_schema;
///
/// let manifest = recorder_schema(&["front".to_string(), "wrist".to_string()]);
/// assert_eq!(manifest.channels.len(), 6);
/// assert_eq!(manifest.channels[0].channel, "/joint_states");
/// assert_eq!(manifest.channels[4].channel, "/camera/front");
/// assert_eq!(manifest.channels[5].channel, "/camera/wrist");
/// ```
#[must_use]
pub fn recorder_schema(labels: &[String]) -> RecorderSchema {
    let mut channels: Vec<FrameSchema> = Vec::with_capacity(4 + labels.len());

    channels.push(FrameSchema {
        channel: JOINT_STATES_TOPIC.to_string(),
        message_type: "JointState".to_string(),
        encoding: FrameEncoding::Json,
        version: FrameSchema::SCHEMA_VERSION,
    });
    channels.push(FrameSchema {
        channel: ACTIONS_TOPIC.to_string(),
        message_type: "Action".to_string(),
        encoding: FrameEncoding::Json,
        version: FrameSchema::SCHEMA_VERSION,
    });
    channels.push(FrameSchema {
        channel: REWARD_TOPIC.to_string(),
        message_type: "Reward".to_string(),
        encoding: FrameEncoding::Json,
        version: FrameSchema::SCHEMA_VERSION,
    });
    channels.push(FrameSchema {
        channel: BODY_POSES_TOPIC.to_string(),
        message_type: "BodyPose".to_string(),
        encoding: FrameEncoding::Json,
        version: FrameSchema::SCHEMA_VERSION,
    });

    for label in labels {
        channels.push(FrameSchema {
            channel: camera_topic(label),
            message_type: "Image".to_string(),
            encoding: FrameEncoding::RawBytes,
            version: FrameSchema::SCHEMA_VERSION,
        });
    }

    RecorderSchema {
        channels,
        version: RecorderSchema::SCHEMA_VERSION,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn camera_topic_concatenates_prefix_and_label() {
        assert_eq!(camera_topic("front"), "/camera/front");
        assert_eq!(camera_topic("wrist"), "/camera/wrist");
        let topic = camera_topic("rear_left");
        assert!(topic.starts_with(CAMERA_TOPIC_PREFIX));
        assert!(topic.ends_with("rear_left"));
    }

    #[test]
    fn topic_constants_match_recorder_doc_table() {
        // Pins the wire format. If any of these literals changes the
        // recording produced by an older recorder becomes unreadable;
        // bump `RecorderSchema::SCHEMA_VERSION` first.
        assert_eq!(JOINT_STATES_TOPIC, "/joint_states");
        assert_eq!(ACTIONS_TOPIC, "/actions");
        assert_eq!(REWARD_TOPIC, "/reward");
        assert_eq!(BODY_POSES_TOPIC, "/body_poses");
        assert_eq!(CAMERA_TOPIC_PREFIX, "/camera/");
    }

    #[test]
    fn recorder_schema_includes_all_fixed_channels() {
        let manifest = recorder_schema(&["front".to_string(), "wrist".to_string()]);
        assert_eq!(manifest.version, RecorderSchema::SCHEMA_VERSION);

        let names: HashSet<&str> = manifest
            .channels
            .iter()
            .map(|c| c.channel.as_str())
            .collect();
        let expected: HashSet<&str> = [
            "/joint_states",
            "/actions",
            "/reward",
            "/body_poses",
            "/camera/front",
            "/camera/wrist",
        ]
        .into_iter()
        .collect();
        assert_eq!(names, expected);

        // Fixed channels are JSON; camera channels are RawBytes.
        for frame in &manifest.channels {
            if frame.channel.starts_with(CAMERA_TOPIC_PREFIX) {
                assert_eq!(frame.encoding, FrameEncoding::RawBytes);
                assert_eq!(frame.message_type, "Image");
            } else {
                assert_eq!(frame.encoding, FrameEncoding::Json);
            }
            assert_eq!(frame.version, FrameSchema::SCHEMA_VERSION);
        }
    }

    #[test]
    fn recorder_schema_with_no_cameras_has_only_fixed_channels() {
        let manifest = recorder_schema(&[]);
        assert_eq!(manifest.channels.len(), 4);
        assert_eq!(manifest.channels[0].channel, JOINT_STATES_TOPIC);
        assert_eq!(manifest.channels[1].channel, ACTIONS_TOPIC);
        assert_eq!(manifest.channels[2].channel, REWARD_TOPIC);
        assert_eq!(manifest.channels[3].channel, BODY_POSES_TOPIC);
    }
}
