//! `clankers-record` — MCAP episode recorder for the Clankers robotics simulator.
//!
//! Add [`RecorderPlugin`] to your Bevy app and insert a [`RecordingConfig`]
//! resource to enable recording. The plugin writes an MCAP file containing
//! timestamped joint states, actions, and rewards.
//!
//! # Feature flags
//!
//! | Feature  | Description                                          |
//! |----------|------------------------------------------------------|
//! | `camera` | Enable recording camera frames from `clankers-render` |
//!
//! # Example
//!
//! ```no_run
//! use bevy::prelude::*;
//! use clankers_record::prelude::*;
//!
//! let mut app = App::new();
//! app.insert_resource(RecordingConfig {
//!     output_path: "my_episode.mcap".into(),
//!     ..RecordingConfig::default()
//! });
//! app.add_plugins(RecorderPlugin);
//! ```

pub mod plugin;
pub mod recorder;
pub mod types;

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::{
        plugin::RecorderPlugin,
        recorder::{PendingAction, PendingReward, RecordingConfig, Recorder},
        types::{ActionFrame, ImageFrame, JointFrame, RewardFrame},
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use bevy::prelude::*;

    /// Verify the plugin can be added to a minimal App without panicking.
    #[test]
    fn recorder_plugin_builds_without_panic() {
        use clankers_core::time::SimTime;
        let tmp = std::env::temp_dir().join("clankers_test_recorder.mcap");
        let mut app = App::new();
        app.add_plugins(bevy::MinimalPlugins);
        // Provide SimTime resource required by recording systems.
        app.insert_resource(SimTime::new());
        app.insert_resource(RecordingConfig {
            output_path: tmp,
            record_joints: false,
            record_actions: false,
            record_rewards: false,
        });
        app.add_plugins(RecorderPlugin);
        app.finish();
        app.cleanup();
        app.update();
    }

    /// Verify a round-trip write+read of a JointFrame via serde_json.
    #[test]
    fn joint_frame_write_read_roundtrip() {
        let original = JointFrame {
            timestamp_ns: 1_000_000_000,
            names: vec!["shoulder".to_string(), "elbow".to_string()],
            positions: vec![0.1_f32, -0.2_f32],
            velocities: vec![0.0_f32, 0.1_f32],
            torques: vec![2.5_f32, -1.0_f32],
        };

        let encoded = serde_json::to_vec(&original).expect("serialize");
        let decoded: JointFrame = serde_json::from_slice(&encoded).expect("deserialize");
        assert_eq!(original, decoded);
    }
}
