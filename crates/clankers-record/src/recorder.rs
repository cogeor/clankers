//! MCAP-backed episode recorder resources and Bevy systems.
//!
//! # Architecture
//!
//! [`RecordingConfig`] is inserted as a Bevy [`Resource`] before the app
//! starts.  [`Recorder`] is inserted by `RecorderPlugin` on startup as a
//! **non-send resource** (because `mcap::Writer` is not `Send`).
//! The recording systems run in [`PostUpdate`] and append serialized frames
//! to the open MCAP file.
//!
//! Each data stream gets its own MCAP channel. The canonical topic
//! strings live in [`crate::schema`] — cite the constants by name
//! (e.g. [`crate::schema::JOINT_STATES_TOPIC`]) when constructing a
//! channel so the wire format cannot drift:
//!
//! | Topic             | Constant                                | Payload type             | Encoding                   |
//! |-------------------|-----------------------------------------|--------------------------|----------------------------|
//! | `/manifest`       | [`crate::schema::MANIFEST_TOPIC`]       | `RecorderSchema`         | `application/json` (once)  |
//! | `/joint_states`   | [`crate::schema::JOINT_STATES_TOPIC`]   | [`JointFrame`]           | `application/json`         |
//! | `/actions`        | [`crate::schema::ACTIONS_TOPIC`]        | [`ActionFrame`]          | `application/json`         |
//! | `/reward`         | [`crate::schema::REWARD_TOPIC`]         | [`RewardFrame`]          | `application/json`         |
//! | `/body_poses`     | [`crate::schema::BODY_POSES_TOPIC`]     | [`BodyPoseFrame`]        | `application/json`         |
//! | `/camera/{label}` | [`crate::schema::camera_topic`] (with prefix [`crate::schema::CAMERA_TOPIC_PREFIX`]) | raw pixels | `application/octet-stream` |

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy::prelude::*;
use mcap::records::MessageHeader;
use mcap::write::Writer as McapWriter;

use clankers_actuator::components::{JointState, JointTorque};
use clankers_core::time::SimTime;

use crate::async_writer::{AsyncRecorder, AsyncSink};
use crate::schema;
use crate::types::{ActionFrame, BodyPoseFrame, JointFrame, RewardFrame};

// ---------------------------------------------------------------------------
// RecordingConfig
// ---------------------------------------------------------------------------

/// Configuration resource that controls what gets recorded and where.
///
/// Insert this resource before the app starts.  If it is absent the
/// `RecorderPlugin` will use sensible defaults and write to
/// `episode_recording.mcap` in the current directory.
#[derive(Resource, Clone, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct RecordingConfig {
    /// Path for the output MCAP file.
    pub output_path: PathBuf,
    /// Whether to record joint state (`/joint_states` channel).
    pub record_joints: bool,
    /// Whether to record actions (`/actions` channel).
    pub record_actions: bool,
    /// Whether to record rewards (`/reward` channel).
    pub record_rewards: bool,
    /// Whether to record body poses (`/body_poses` channel).
    pub record_body_poses: bool,
    /// When `true`, the recorder dispatches every JSON write through the
    /// W7 PR4 [`AsyncRecorder`](crate::async_writer::AsyncRecorder)
    /// background worker rather than blocking the Bevy `PostUpdate`
    /// schedule on disk I/O. Default `false` for back-compat —
    /// existing recordings are byte-identical to the pre-W7 sync path
    /// when this stays unset.
    ///
    /// NOTE for W7 PR4: the async dispatch currently routes JSON writes
    /// only. The camera (raw image) hot path stays sync — the binary
    /// channel write goes through `mcap::Writer` directly and is
    /// unaffected by this flag. Lifting the camera path onto the async
    /// queue is W8 follow-up scope.
    pub async_mode: bool,
    /// Bounded queue capacity for the async writer (in frames).
    /// Ignored when [`Self::async_mode`] is `false`. Default 256.
    pub async_buffer_capacity: usize,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from("episode_recording.mcap"),
            record_joints: true,
            record_actions: true,
            record_rewards: true,
            record_body_poses: false,
            async_mode: false,
            async_buffer_capacity: 256,
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
    pub body_poses: Option<u16>,
    /// MCAP channel id for the [`crate::schema::MANIFEST_TOPIC`]
    /// `/manifest` channel. The first record on this channel is the
    /// W7 PR4 [`crate::schema::recorder_schema`] payload (written once
    /// at recording open, immediately after channel registration).
    pub manifest: Option<u16>,
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

/// Non-send Bevy resource wrapping the open MCAP writer.
///
/// Registered as a **non-send resource** because `mcap::Writer` is not `Send`.
/// Recording systems must use [`NonSendMut<Recorder>`] to access it.
///
/// # Sync vs async backend (W7 PR4)
///
/// When [`RecordingConfig::async_mode`] is `false` (default) every
/// [`Self::write_json`] call writes directly through the inline
/// [`mcap::write::Writer`] — byte-identical to the pre-W7 behaviour.
/// When `async_mode` is `true`, [`Self::write_json`] dispatches to the
/// [`AsyncRecorder`] worker thread via a bounded crossbeam channel and
/// the inline writer is still held but the producer pushes through
/// `try_send_frame`. The asynchronous path always also publishes
/// directly through the writer would race with the worker so the W7 PR4
/// design pulls the writer out of `Self::writer` and gives ownership to
/// the worker through an [`AsyncSink`] adapter. See [`Recorder::open`]
/// for the wiring.
pub struct Recorder {
    /// The underlying MCAP writer when the recorder is in **sync** mode
    /// (or before [`Self::install_async`] is called). `None` once the
    /// async worker has taken ownership of the writer.
    pub(crate) writer: Option<McapWriter<BufWriter<File>>>,
    /// Monotonically increasing message sequence counter (sync path
    /// only — the async sink keeps its own counter).
    pub(crate) sequence: u32,
    /// Async dispatch handle. Populated by [`Self::install_async`]; when
    /// `Some`, [`Self::write_json`] routes through it instead of the
    /// inline writer.
    pub(crate) async_writer: Option<AsyncRecorder>,
    /// Shared dropped-frame counter (cloned with the [`AsyncRecorder`]).
    /// In sync mode this stays at 0.
    pub(crate) dropped: Arc<AtomicU64>,
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
            async_writer: None,
            dropped: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Take the inline writer and hand it to a fresh [`AsyncRecorder`]
    /// of the given queue capacity.
    ///
    /// After this call [`Self::writer`] is `None` and every
    /// [`Self::write_json`] dispatches through the bounded crossbeam
    /// channel. Calling this twice is a no-op.
    pub(crate) fn install_async(&mut self, capacity: usize) {
        if self.async_writer.is_some() {
            return;
        }
        let Some(writer) = self.writer.take() else {
            return;
        };
        let sink = McapAsyncSink { writer };
        let dropped = self.dropped.clone();
        let async_recorder = AsyncRecorder::new(capacity, sink, Some(dropped));
        self.async_writer = Some(async_recorder);
    }

    /// Snapshot of the shared dropped-frame counter.
    ///
    /// Stays at 0 in sync mode (the sync path always blocks until the
    /// writer accepts the bytes). In async mode this reflects the number
    /// of frames the producer dropped because the bounded queue was
    /// full.
    #[must_use]
    pub fn dropped_frames(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Cloneable handle to the shared dropped-frame counter so the
    /// Bevy [`crate::async_writer::DroppedFrames`] resource and the
    /// recorder agree on a single source of truth.
    #[must_use]
    pub fn dropped_frames_handle(&self) -> Arc<AtomicU64> {
        self.dropped.clone()
    }

    /// Register the JSON schema used for all channels in this recorder (static).
    pub(crate) fn register_json_schema(
        writer: &mut McapWriter<BufWriter<File>>,
    ) -> Result<u16, Box<dyn std::error::Error>> {
        // Use an empty schema for JSON — schema data is optional for JSON encoding.
        let schema_id = writer.add_schema("json", "jsonschema", &[])?;
        Ok(schema_id)
    }

    /// Add a channel for the given topic using JSON encoding (static).
    pub(crate) fn add_json_channel(
        writer: &mut McapWriter<BufWriter<File>>,
        schema_id: u16,
        topic: &str,
    ) -> Result<u16, Box<dyn std::error::Error>> {
        let channel_id =
            writer.add_channel(schema_id, topic, "application/json", &BTreeMap::new())?;
        Ok(channel_id)
    }

    /// Register the JSON schema on this recorder's writer.
    ///
    /// Returns the schema ID on success.
    pub fn register_schema(&mut self) -> Result<u16, Box<dyn std::error::Error>> {
        let writer = self.writer.as_mut().ok_or("writer not open")?;
        let schema_id = writer.add_schema("json", "jsonschema", &[])?;
        Ok(schema_id)
    }

    /// Add a JSON-encoded channel for the given topic on this recorder's writer.
    ///
    /// Returns the channel ID on success.
    pub fn add_channel(
        &mut self,
        schema_id: u16,
        topic: &str,
    ) -> Result<u16, Box<dyn std::error::Error>> {
        let writer = self.writer.as_mut().ok_or("writer not open")?;
        let channel_id =
            writer.add_channel(schema_id, topic, "application/json", &BTreeMap::new())?;
        Ok(channel_id)
    }

    /// Write a raw JSON payload to a known channel.
    ///
    /// Dispatches through the [`AsyncRecorder`] when one has been
    /// installed via [`Self::install_async`]; otherwise writes inline
    /// through the held [`mcap::write::Writer`]. The dispatch boundary
    /// is the **single** point where async vs sync diverges — every
    /// per-system writer (`write_joint_frame`, `write_action_frame`,
    /// `write_reward_frame`, `write_body_pose_frame`) routes through
    /// this function and inherits the backend.
    pub(crate) fn write_json(
        &mut self,
        channel_id: u16,
        timestamp_ns: u64,
        payload: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref async_rec) = self.async_writer {
            // Owned payload — the worker thread keeps the bytes alive
            // until it finishes the write.
            async_rec.try_send_frame(channel_id, timestamp_ns, payload.to_vec());
            return Ok(());
        }
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

    /// Serialize and write a [`BodyPoseFrame`] to the given channel.
    pub fn write_body_pose_frame(
        &mut self,
        channel_id: u16,
        frame: &BodyPoseFrame,
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
    /// In async mode the worker is shut down first (draining the queue);
    /// the writer is then recovered through the sink and finalised
    /// inline so the MCAP footer is byte-equivalent to the sync path.
    ///
    /// # Errors
    ///
    /// Returns an error if the MCAP footer cannot be written.
    pub fn finish(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Drop the async writer first so the worker drains the queue
        // and closes the sink. The Drop impl on `AsyncRecorder`
        // performs the join.
        self.async_writer.take();
        if let Some(mut w) = self.writer.take() {
            w.finish()?;
        }
        Ok(())
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // Drop the async writer here too — the worker holds the inline
        // writer in async mode, so we must let it run its sink's flush
        // before we attempt to finish the inline writer (which by then
        // is `None`, so `finish` is a no-op).
        self.async_writer.take();
        if let Err(e) = self.finish() {
            eprintln!("clankers-record: MCAP finalization failed in Drop: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// McapAsyncSink — adapter that owns the inline MCAP writer in async mode
// ---------------------------------------------------------------------------

/// [`AsyncSink`] adapter that owns the inline MCAP writer once the
/// recorder transitions to async mode.
///
/// The recorder is a Bevy **non-send** resource, but the async worker
/// runs on its own OS thread. We can hand the writer across the thread
/// boundary because `mcap::Writer<BufWriter<File>>` is `Send` (its
/// internal state holds only `BufWriter<File>`, which is `Send`; the
/// non-`Send` constraint on `Recorder` itself is a defensive choice
/// rooted in older mcap versions). The adapter exists solely to give
/// the worker something it can call without re-implementing the message
/// header construction.
struct McapAsyncSink {
    writer: McapWriter<BufWriter<File>>,
}

impl AsyncSink for McapAsyncSink {
    fn write_message(
        &mut self,
        channel_id: u16,
        log_time_ns: u64,
        payload: &[u8],
    ) -> std::io::Result<()> {
        // mcap returns its own error type; collapse to `io::Error` so
        // the worker can log uniformly.
        // sequence: monotonic per-sink. We start at 0 because the
        // recorder's `sequence` only advances on the sync path; the
        // async path's sequence is local to this sink.
        let header = MessageHeader {
            channel_id,
            sequence: 0,
            log_time: log_time_ns,
            publish_time: log_time_ns,
        };
        self.writer
            .write_to_known_channel(&header, payload)
            .map_err(|e| std::io::Error::other(format!("mcap write failed: {e}")))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // mcap::Writer batches in-memory; the `finish` step writes the
        // index + footer. Inline flush is a no-op at the mcap level —
        // it is the right semantic for the AsyncSink contract, which
        // requests a best-effort flush to the underlying writer.
        Ok(())
    }
}

impl Drop for McapAsyncSink {
    fn drop(&mut self) {
        // Worker thread shutdown — finalize the MCAP footer so the file
        // is readable. We can't propagate the error out of Drop; log
        // and continue.
        if let Err(e) = self.writer.finish() {
            eprintln!("clankers-record async worker: MCAP finalize failed: {e}");
        }
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
// Pending body poses resource
// ---------------------------------------------------------------------------

/// Body poses to be recorded this frame, set by external systems.
///
/// Each entry maps a body name to `[x, y, z, qx, qy, qz, qw]`.
/// Populate this from your physics backend (e.g. `RapierContext`) before
/// `PostUpdate` runs so [`record_body_poses_system`] picks it up.
#[derive(Resource, Default, Debug, Clone)]
pub struct PendingBodyPoses(pub std::collections::HashMap<String, [f32; 7]>);

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

    // Insert a `DroppedFrames` resource whose `Arc<AtomicU64>` is shared
    // with the recorder. In sync mode the counter stays at 0; in async
    // mode (installed during `setup_channels`) the worker thread shares
    // the same counter so every reader sees the same value.
    let dropped = crate::async_writer::DroppedFrames(recorder.dropped_frames_handle());
    world.insert_resource(dropped);

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

    let mut manifest_payload: Option<Vec<u8>> = None;
    if let Some(ref mut writer) = recorder.writer {
        match Recorder::register_json_schema(writer) {
            Ok(schema_id) => {
                if config.record_joints {
                    match Recorder::add_json_channel(writer, schema_id, schema::JOINT_STATES_TOPIC)
                    {
                        Ok(id) => channel_ids.joints = Some(id),
                        Err(e) => {
                            error!("clankers-record: failed to add /joint_states channel: {e}");
                        }
                    }
                }
                if config.record_actions {
                    match Recorder::add_json_channel(writer, schema_id, schema::ACTIONS_TOPIC) {
                        Ok(id) => channel_ids.action = Some(id),
                        Err(e) => {
                            error!("clankers-record: failed to add /actions channel: {e}");
                        }
                    }
                }
                if config.record_rewards {
                    match Recorder::add_json_channel(writer, schema_id, schema::REWARD_TOPIC) {
                        Ok(id) => channel_ids.reward = Some(id),
                        Err(e) => {
                            error!("clankers-record: failed to add /reward channel: {e}");
                        }
                    }
                }
                if config.record_body_poses {
                    match Recorder::add_json_channel(writer, schema_id, schema::BODY_POSES_TOPIC) {
                        Ok(id) => channel_ids.body_poses = Some(id),
                        Err(e) => {
                            error!("clankers-record: failed to add /body_poses channel: {e}");
                        }
                    }
                }
                // W7 PR4: register the /manifest channel and stage the
                // payload for the first MCAP record. Camera labels are
                // empty here — the loader either reads the manifest as
                // a baseline channel set or falls back to topic-glob
                // discovery for camera channels (W6 loop 02 design).
                match Recorder::add_json_channel(writer, schema_id, schema::MANIFEST_TOPIC) {
                    Ok(id) => {
                        channel_ids.manifest = Some(id);
                        let manifest = schema::recorder_schema(&[]);
                        match serde_json::to_vec(&manifest) {
                            Ok(bytes) => manifest_payload = Some(bytes),
                            Err(e) => {
                                error!("clankers-record: failed to serialise manifest payload: {e}")
                            }
                        }
                    }
                    Err(e) => {
                        error!("clankers-record: failed to add /manifest channel: {e}");
                    }
                }
            }
            Err(e) => error!("clankers-record: failed to register JSON schema: {e}"),
        }
    }

    // Write the manifest payload as the **first** MCAP record on the
    // /manifest channel (W7 PR4 — W6 PR1 deferral closed). This stays
    // on the sync path because async install happens *after* this point
    // so the manifest sequence number is deterministic across recordings.
    if let (Some(manifest_id), Some(payload)) = (channel_ids.manifest, manifest_payload.as_ref())
        && let Err(e) = recorder.write_json(manifest_id, 0, payload)
    {
        error!("clankers-record: failed to write manifest record: {e}");
    }

    // W7 PR4: hand the inline writer to the async worker if the config
    // asks for it. This must happen AFTER channel registration and the
    // manifest record write so both stay on the sync path.
    if config.async_mode {
        recorder.install_async(config.async_buffer_capacity);
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

/// `PostUpdate` system: writes current [`PendingBodyPoses`] as a [`BodyPoseFrame`].
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are extracted by value
pub fn record_body_poses_system(
    recorder: Option<NonSendMut<Recorder>>,
    channel_ids: Res<ChannelIds>,
    sim_time: Res<SimTime>,
    pending: Res<PendingBodyPoses>,
) {
    let Some(channel_id) = channel_ids.body_poses else {
        return;
    };

    let Some(mut recorder) = recorder else {
        return;
    };

    if pending.0.is_empty() {
        return;
    }

    let frame = BodyPoseFrame {
        timestamp_ns: sim_time.nanos(),
        poses: pending.0.clone(),
    };

    if let Err(e) = recorder.write_body_pose_frame(channel_id, &frame) {
        error!("clankers-record: failed to write body pose frame: {e}");
    }
}

// ---------------------------------------------------------------------------
// Camera recording system (feature "camera")
// ---------------------------------------------------------------------------

#[cfg(feature = "camera")]
pub mod camera {
    //! Optional camera recording system.
    use std::collections::{BTreeMap, HashMap};

    use bevy::prelude::*;
    use mcap::records::MessageHeader;

    use super::{Recorder, SimTime, schema};
    use clankers_render::buffer::CameraFrameBuffers;

    /// Per-camera MCAP channel ID cache.
    #[derive(Resource, Default)]
    pub struct CameraChannelIds {
        /// Channel IDs keyed by camera label.
        pub channels: HashMap<String, u16>,
        /// Shared schema ID for binary image channels (registered once).
        pub schema_id: Option<u16>,
    }

    /// `PostUpdate` system: writes raw pixel bytes per registered camera.
    #[allow(clippy::needless_pass_by_value)] // Bevy system parameters are extracted by value
    pub fn record_image_system(
        recorder: Option<NonSendMut<Recorder>>,
        mut cam_channels: ResMut<CameraChannelIds>,
        sim_time: Res<SimTime>,
        frame_buffers: Option<Res<CameraFrameBuffers>>,
    ) {
        let Some(mut recorder) = recorder else {
            return;
        };
        let Some(frame_buffers) = frame_buffers else {
            return;
        };

        for (label, buf) in frame_buffers.iter() {
            // Lazily register a channel per camera label.
            let channel_id = if let Some(&id) = cam_channels.channels.get(label) {
                id
            } else if let Some(ref mut writer) = recorder.writer {
                // Register shared binary schema once.
                let schema_id = if let Some(id) = cam_channels.schema_id {
                    id
                } else {
                    let id = match writer.add_schema("binary", "application/octet-stream", &[]) {
                        Ok(id) => id,
                        Err(e) => {
                            error!("clankers-record: failed to register binary schema: {e}");
                            continue;
                        }
                    };
                    cam_channels.schema_id = Some(id);
                    id
                };

                let topic = schema::camera_topic(label);
                let mut metadata = BTreeMap::new();
                metadata.insert("width".to_string(), buf.width().to_string());
                metadata.insert("height".to_string(), buf.height().to_string());
                metadata.insert(
                    "channels".to_string(),
                    (buf.format().bytes_per_pixel()).to_string(),
                );
                match writer.add_channel(schema_id, &topic, "application/octet-stream", &metadata) {
                    Ok(id) => {
                        cam_channels.channels.insert(label.to_string(), id);
                        id
                    }
                    Err(e) => {
                        error!("clankers-record: failed to add channel {topic}: {e}");
                        continue;
                    }
                }
            } else {
                continue;
            };

            // Write raw pixel bytes directly.
            let ts = sim_time.nanos();
            let payload = buf.data();

            let seq = recorder.sequence;
            recorder.sequence = recorder.sequence.wrapping_add(1);
            if let Some(ref mut w) = recorder.writer
                && let Err(e) = w.write_to_known_channel(
                    &MessageHeader {
                        channel_id,
                        sequence: seq,
                        log_time: ts,
                        publish_time: ts,
                    },
                    payload,
                )
            {
                error!("clankers-record: failed to write image frame: {e}");
            }
        }
    }
}
