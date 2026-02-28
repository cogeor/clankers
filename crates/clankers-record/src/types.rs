//! Serializable frame types written as MCAP message payloads.
//!
//! Each frame type corresponds to one MCAP topic:
//! - `JointFrame`  → `/joints`
//! - `ActionFrame` → `/action`
//! - `RewardFrame` → `/reward`
//! - `ImageFrame`  → `/camera/{label}` (feature "camera")

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// JointFrame
// ---------------------------------------------------------------------------

/// Snapshot of all joint states at a single simulation step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)] // f32 fields prevent Eq
pub struct JointFrame {
    /// Simulation timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Joint names (one per joint, same order as the value vectors).
    pub names: Vec<String>,
    /// Joint positions (rad).
    pub positions: Vec<f32>,
    /// Joint velocities (rad/s).
    pub velocities: Vec<f32>,
    /// Joint torques (Nm) — zero-filled if torque data is unavailable.
    pub torques: Vec<f32>,
}

// ---------------------------------------------------------------------------
// ActionFrame
// ---------------------------------------------------------------------------

/// Continuous action command issued to the environment at one step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)] // f32 fields prevent Eq
pub struct ActionFrame {
    /// Simulation timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Flattened action values.
    pub data: Vec<f32>,
}

// ---------------------------------------------------------------------------
// RewardFrame
// ---------------------------------------------------------------------------

/// Scalar reward received at the end of a step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)] // f32 fields prevent Eq
pub struct RewardFrame {
    /// Simulation timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Reward value.
    pub reward: f32,
}

// ---------------------------------------------------------------------------
// ImageFrame
// ---------------------------------------------------------------------------

/// Raw pixel data from a camera sensor (optional feature "camera").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageFrame {
    /// Simulation timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Camera label identifying which camera produced this frame.
    pub label: String,
    /// Raw pixel bytes (format depends on the camera's [`PixelFormat`]).
    ///
    /// [`PixelFormat`]: clankers_render::config::PixelFormat
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_frame_roundtrip() {
        let frame = JointFrame {
            timestamp_ns: 1_000_000_000,
            names: vec!["shoulder".to_string(), "elbow".to_string()],
            positions: vec![0.1, -0.2],
            velocities: vec![0.05, 0.0],
            torques: vec![1.5, -0.3],
        };
        let json = serde_json::to_string(&frame).unwrap();
        let frame2: JointFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(frame, frame2);
    }

    #[test]
    fn action_frame_roundtrip() {
        let frame = ActionFrame {
            timestamp_ns: 500_000,
            data: vec![0.5, -0.5, 0.0],
        };
        let json = serde_json::to_string(&frame).unwrap();
        let frame2: ActionFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(frame, frame2);
    }

    #[test]
    fn reward_frame_roundtrip() {
        let frame = RewardFrame {
            timestamp_ns: 999,
            reward: 1.0,
        };
        let json = serde_json::to_string(&frame).unwrap();
        let frame2: RewardFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(frame, frame2);
    }

    #[test]
    fn image_frame_roundtrip() {
        let frame = ImageFrame {
            timestamp_ns: 42,
            width: 2,
            height: 1,
            label: "front".to_string(),
            data: vec![255, 0, 0, 0, 255, 0],
        };
        let json = serde_json::to_string(&frame).unwrap();
        let frame2: ImageFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(frame, frame2);
    }
}
