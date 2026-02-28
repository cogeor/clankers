//! MCAP-backed episode recorder resources and Bevy systems.
//!
//! # Architecture
//!
//! [`RecordingConfig`] is inserted as a Bevy [`Resource`] before the app
//! starts.  [`Recorder`] is inserted by [`RecorderPlugin`] on startup as a
//! **non-send resource** (because `mcap::Writer` is not `Send`).
//! The recording systems run in [`PostUpdate`] and append serialized frames
//! to the open MCAP file.
//!
//! Each data stream gets its own MCAP channel:
//!
//! | Topic             | Payload type    | Encoding           |
//! |-------------------|-----------------|-------------------|
//! | `/joints`         | [`JointFrame`]  | `application/json` |
//! | `/action`         | [`ActionFrame`] | `application/json` |
//! | `/reward`         | [`RewardFrame`] | `application/json` |
//! | `/camera/{label}` | [`ImageFrame`]  | `application/json` |

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use bevy::prelude::*;
use mcap::records::MessageHeader;
use mcap::write::Writer as McapWriter;

use clankers_actuator::components::{JointState, JointTorque};
use clankers_core::time::SimTime;

use crate::types::{ActionFrame, JointFrame, RewardFrame};

// ---------------------------------------------------------------------------
// RecordingConfig
// ---------------------------------------------------------------------------

/// Configuration resource that controls what gets recorded and where.
///
/// Insert this resource before the app starts.  If it is absent the
/// [`RecorderPlugin`] will use sensible defaults and write to
/// `episode_recording.mcap` in the current directory.
#[derive(Resource, Clone, Debug)]
pub struct RecordingConfig {
    /// Path for the output MCAP file.
    pub output_path: PathBuf,
    /// Whether to record joint state (`/joints` channel).
    pub record_joints: bool,
    /// Whether to record actions (`/action` channel).
    pub record_actions: bool,
    /// Whether to record rewards (`/reward` channel).
    pub record_rewards: bool,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from("episode_recording.mcap"),
            record_joints: true,
            record_actions: true,
            record_rewards: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ChannelIds
// ---------------------------------------------------------------------------

/// Internal resource storing pre-registered MCAP channel IDs.
#[derive(Resource, Debug, Default)]
pub struct ChannelIds {
    pub joints: Option<u16>,
    pub action: Option<u16>,
    pub reward: Option<u16>,
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

/// Non-send Bevy resource wrapping the open MCAP writer.
///
/// Registered as a **non-send resource** because `mcap::Writer` is not `Send`.
/// Recording systems must use [`NonSendMut<Recorder>`] to access it.
pub struct Recorder {
    pub(crate) writer: Option<McapWriter<BufWriter<File>>>,
    pub(crate) sequence: u32,
}

impl Recorder {
    /// Open a new MCAP file at the given path and return a ready Recorder.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or the MCAP header
    /// cannot be written.
    pub fn open(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let buf = BufWriter::new(file);
        let writer = McapWriter::new(buf)?;
        Ok(Self {
            writer: Some(writer),
            sequence: 0,
        })
    }

    /// Register the JSON schema used for all channels in this recorder.
    pub(crate) fn register_json_schema(
        writer: &mut McapWriter<BufWriter<File>>,
    ) -> Result<u16, Box<dyn std::error::Error>> {
        // Use an empty schema for JSON — schema data is optional for JSON encoding.
        let schema_id = writer.add_schema("json", "jsonschema", &[])?;
        Ok(schema_id)
    }

    /// Add a channel for the given topic using JSON encoding.
    pub(crate) fn add_json_channel(
        writer: &mut McapWriter<BufWriter<File>>,
        schema_id: u16,
        topic: &str,
    ) -> Result<u16, Box<dyn std::error::Error>> {
        let channel_id =
            writer.add_channel(schema_id, topic, "application/json", &BTreeMap::new())?;
        Ok(channel_id)
    }

    /// Write a raw JSON payload to a known channel.
    pub(crate) fn write_json(
        &mut self,
        channel_id: u16,
        timestamp_ns: u64,
        payload: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let seq = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        if let Some(ref mut w) = self.writer {
            w.write_to_known_channel(
                &MessageHeader {
                    channel_id,
                    sequence: seq,
                    log_time: timestamp_ns,
                    publish_time: timestamp_ns,
                },
                payload,
            )?;
        }
        Ok(())
    }

    /// Serialize and write a [`JointFrame`] to the given channel.
    pub fn write_joint_frame(
        &mut self,
        channel_id: u16,
        frame: &JointFrame,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ts = frame.timestamp_ns;
        let payload = serde_json::to_vec(frame)?;
        self.write_json(channel_id, ts, &payload)
    }

    /// Serialize and write an [`ActionFrame`] to the given channel.
    pub fn write_action_frame(
        &mut self,
        channel_id: u16,
        frame: &ActionFrame,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ts = frame.timestamp_ns;
        let payload = serde_json::to_vec(frame)?;
        self.write_json(channel_id, ts, &payload)
    }

    /// Serialize and write a [`RewardFrame`] to the given channel.
    pub fn write_reward_frame(
        &mut self,
        channel_id: u16,
        frame: &RewardFrame,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ts = frame.timestamp_ns;
        let payload = serde_json::to_vec(frame)?;
        self.write_json(channel_id, ts, &payload)
    }

    /// Finalize the MCAP file.  Must be called before drop for a valid file.
    ///
    /// # Errors
    ///
    /// Returns an error if the MCAP footer cannot be written.
    pub fn finish(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut w) = self.writer.take() {
            w.finish()?;
        }
        Ok(())
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // Best-effort finish; errors are silently ignored on drop.
        let _ = self.finish();
    }
}

// ---------------------------------------------------------------------------
// Pending reward resource
// ---------------------------------------------------------------------------

/// Scalar reward to be recorded this frame, set by external systems.
///
/// Insert and update this resource to drive [`record_reward_system`].
#[derive(Resource, Default, Debug, Clone)]
pub struct PendingReward(pub f32);

// ---------------------------------------------------------------------------
// Pending action resource
// ---------------------------------------------------------------------------

/// Action to be recorded this frame, set by external systems.
///
/// Insert and update this resource to drive [`record_action_system`].
#[derive(Resource, Default, Debug, Clone)]
pub struct PendingAction(pub Vec<f32>);

// ---------------------------------------------------------------------------
// Startup system: initialise Recorder and register channels
// ---------------------------------------------------------------------------

/// Startup system that opens the MCAP file and prepares the recorder.
pub fn setup_recorder(world: &mut World) {
    let config = world
        .get_resource::<RecordingConfig>()
        .cloned()
        .unwrap_or_default();

    let recorder = match Recorder::open(&config.output_path) {
        Ok(r) => r,
        Err(e) => {
            error!("clankers-record: failed to open MCAP file: {e}");
            return;
        }
    };

    world.insert_non_send_resource(recorder);
    world.insert_resource(ChannelIds::default());
    world.insert_resource(NeedsChannelSetup);
}

/// Marker resource — removed after channels have been registered.
#[derive(Resource, Default)]
pub struct NeedsChannelSetup;

/// One-shot system that registers MCAP channels on the first frame.
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are extracted by value
pub fn setup_channels(
    mut commands: Commands,
    recorder: Option<NonSendMut<Recorder>>,
    config: Res<RecordingConfig>,
    mut channel_ids: ResMut<ChannelIds>,
    needs_setup: Option<Res<NeedsChannelSetup>>,
) {
    if needs_setup.is_none() {
        return;
    }

    let Some(mut recorder) = recorder else {
        return;
    };

    if let Some(ref mut writer) = recorder.writer {
        match Recorder::register_json_schema(writer) {
            Ok(schema_id) => {
                if config.record_joints {
                    match Recorder::add_json_channel(writer, schema_id, "/joints") {
                        Ok(id) => channel_ids.joints = Some(id),
                        Err(e) => {
                            error!("clankers-record: failed to add /joints channel: {e}");
                        }
                    }
                }
                if config.record_actions {
                    match Recorder::add_json_channel(writer, schema_id, "/action") {
                        Ok(id) => channel_ids.action = Some(id),
                        Err(e) => {
                            error!("clankers-record: failed to add /action channel: {e}");
                        }
                    }
                }
                if config.record_rewards {
                    match Recorder::add_json_channel(writer, schema_id, "/reward") {
                        Ok(id) => channel_ids.reward = Some(id),
                        Err(e) => {
                            error!("clankers-record: failed to add /reward channel: {e}");
                        }
                    }
                }
            }
            Err(e) => error!("clankers-record: failed to register JSON schema: {e}"),
        }
    }

    commands.remove_resource::<NeedsChannelSetup>();
}

// ---------------------------------------------------------------------------
// Recording systems
// ---------------------------------------------------------------------------

/// `PostUpdate` system: collects all joint states and writes a [`JointFrame`].
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are extracted by value
pub fn record_joint_states_system(
    recorder: Option<NonSendMut<Recorder>>,
    channel_ids: Res<ChannelIds>,
    sim_time: Res<SimTime>,
    query: Query<(&Name, &JointState, Option<&JointTorque>)>,
) {
    let Some(channel_id) = channel_ids.joints else {
        return;
    };

    let Some(mut recorder) = recorder else {
        return;
    };

    let mut names = Vec::new();
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut torques = Vec::new();

    for (name, state, torque) in &query {
        names.push(name.to_string());
        positions.push(state.position);
        velocities.push(state.velocity);
        torques.push(torque.map_or(0.0, |t| t.value));
    }

    if names.is_empty() {
        return;
    }

    let frame = JointFrame {
        timestamp_ns: sim_time.nanos(),
        names,
        positions,
        velocities,
        torques,
    };

    if let Err(e) = recorder.write_joint_frame(channel_id, &frame) {
        error!("clankers-record: failed to write joint frame: {e}");
    }
}

/// `PostUpdate` system: writes the current [`PendingAction`] as an [`ActionFrame`].
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are extracted by value
pub fn record_action_system(
    recorder: Option<NonSendMut<Recorder>>,
    channel_ids: Res<ChannelIds>,
    sim_time: Res<SimTime>,
    pending: Res<PendingAction>,
) {
    let Some(channel_id) = channel_ids.action else {
        return;
    };

    let Some(mut recorder) = recorder else {
        return;
    };

    let frame = ActionFrame {
        timestamp_ns: sim_time.nanos(),
        data: pending.0.clone(),
    };

    if let Err(e) = recorder.write_action_frame(channel_id, &frame) {
        error!("clankers-record: failed to write action frame: {e}");
    }
}

/// `PostUpdate` system: writes the current [`PendingReward`] as a [`RewardFrame`].
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are extracted by value
pub fn record_reward_system(
    recorder: Option<NonSendMut<Recorder>>,
    channel_ids: Res<ChannelIds>,
    sim_time: Res<SimTime>,
    pending: Res<PendingReward>,
) {
    let Some(channel_id) = channel_ids.reward else {
        return;
    };

    let Some(mut recorder) = recorder else {
        return;
    };

    let frame = RewardFrame {
        timestamp_ns: sim_time.nanos(),
        reward: pending.0,
    };

    if let Err(e) = recorder.write_reward_frame(channel_id, &frame) {
        error!("clankers-record: failed to write reward frame: {e}");
    }
}

// ---------------------------------------------------------------------------
// Camera recording system (feature "camera")
// ---------------------------------------------------------------------------

#[cfg(feature = "camera")]
pub mod camera {
    //! Optional camera recording system.
    use super::*;
    use clankers_render::buffer::CameraFrameBuffers;
    use crate::types::ImageFrame;
    use std::collections::HashMap;

    /// Per-camera MCAP channel ID cache.
    #[derive(Resource, Default)]
    pub struct CameraChannelIds(pub HashMap<String, u16>);

    /// `PostUpdate` system: writes one [`ImageFrame`] per registered camera.
    #[allow(clippy::needless_pass_by_value)] // Bevy system parameters are extracted by value
    pub fn record_image_system(
        recorder: Option<NonSendMut<Recorder>>,
        mut cam_channels: ResMut<CameraChannelIds>,
        sim_time: Res<SimTime>,
        frame_buffers: Res<CameraFrameBuffers>,
    ) {
        let Some(mut recorder) = recorder else {
            return;
        };

        for (label, buf) in frame_buffers.iter() {
            // Lazily register a channel per camera label.
            let channel_id = if let Some(&id) = cam_channels.0.get(label) {
                id
            } else {
                // Register schema + channel on first use.
                if let Some(ref mut writer) = recorder.writer {
                    let schema_id = match writer.add_schema("json", "jsonschema", &[]) {
                        Ok(id) => id,
                        Err(e) => {
                            error!(
                                "clankers-record: failed to register schema for camera {label}: {e}"
                            );
                            continue;
                        }
                    };
                    let topic = format!("/camera/{label}");
                    match writer.add_channel(
                        schema_id,
                        &topic,
                        "application/json",
                        &BTreeMap::new(),
                    ) {
                        Ok(id) => {
                            cam_channels.0.insert(label.to_string(), id);
                            id
                        }
                        Err(e) => {
                            error!("clankers-record: failed to add channel {topic}: {e}");
                            continue;
                        }
                    }
                } else {
                    continue;
                }
            };

            let frame = ImageFrame {
                timestamp_ns: sim_time.nanos(),
                width: buf.width(),
                height: buf.height(),
                label: label.to_string(),
                data: buf.data().to_vec(),
            };

            let ts = frame.timestamp_ns;
            let payload = match serde_json::to_vec(&frame) {
                Ok(b) => b,
                Err(e) => {
                    error!("clankers-record: failed to serialize ImageFrame: {e}");
                    continue;
                }
            };

            let seq = recorder.sequence;
            recorder.sequence = recorder.sequence.wrapping_add(1);
            if let Some(ref mut w) = recorder.writer {
                if let Err(e) = w.write_to_known_channel(
                    &MessageHeader {
                        channel_id,
                        sequence: seq,
                        log_time: ts,
                        publish_time: ts,
                    },
                    &payload,
                ) {
                    error!("clankers-record: failed to write image frame: {e}");
                }
            }
        }
    }
}
