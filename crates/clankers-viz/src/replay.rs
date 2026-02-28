//! MCAP replay: loads recorded episodes and plays them back in the visualizer.

use std::path::PathBuf;

use bevy::prelude::*;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_record::types::{ActionFrame, JointFrame, RewardFrame};

/// Pre-parsed episode data loaded from an MCAP file.
#[derive(Resource, Default)]
pub struct PlaybackIndex {
    /// Joint frames sorted by `timestamp_ns`.
    pub joint_frames: Vec<JointFrame>,
    /// Action frames sorted by `timestamp_ns`.
    pub action_frames: Vec<ActionFrame>,
    /// Reward frames sorted by `timestamp_ns`.
    pub reward_frames: Vec<RewardFrame>,
}

/// Playback state controlling the timeline cursor.
#[derive(Resource)]
pub struct PlaybackState {
    /// Current playback position in nanoseconds.
    pub cursor_ns: u64,
    /// Total duration of the recording in nanoseconds.
    pub duration_ns: u64,
    /// Whether playback is actively advancing.
    pub playing: bool,
    /// Playback speed multiplier (1.0 = realtime).
    pub speed: f32,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            cursor_ns: 0,
            duration_ns: 0,
            playing: false,
            speed: 1.0,
        }
    }
}

impl PlaybackState {
    /// Normalized position \[0.0, 1.0\].
    pub fn progress(&self) -> f32 {
        if self.duration_ns == 0 {
            return 0.0;
        }
        (self.cursor_ns as f64 / self.duration_ns as f64) as f32
    }

    /// Set cursor from normalized position \[0.0, 1.0\].
    pub fn seek(&mut self, t: f32) {
        self.cursor_ns = ((t.clamp(0.0, 1.0) as f64) * self.duration_ns as f64) as u64;
    }

    /// Current time in seconds.
    pub fn time_secs(&self) -> f64 {
        self.cursor_ns as f64 / 1_000_000_000.0
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.duration_ns as f64 / 1_000_000_000.0
    }
}

// ---------------------------------------------------------------------------
// MCAP Loading
// ---------------------------------------------------------------------------

/// Startup system: loads the MCAP file from [`VizConfig::replay_path`] into
/// [`PlaybackIndex`].
#[allow(clippy::needless_pass_by_value)]
pub fn load_replay_mcap(config: Res<crate::config::VizConfig>, mut commands: Commands) {
    let Some(ref path) = config.replay_path else {
        // No replay file configured -- insert empty defaults so systems don't panic.
        commands.insert_resource(PlaybackIndex::default());
        commands.insert_resource(PlaybackState::default());
        return;
    };

    match load_mcap_file(path) {
        Ok((index, state)) => {
            info!(
                "clankers-viz: loaded replay {} ({} joint frames, {:.1}s)",
                path.display(),
                index.joint_frames.len(),
                state.duration_secs()
            );
            commands.insert_resource(index);
            commands.insert_resource(state);
        }
        Err(e) => {
            error!(
                "clankers-viz: failed to load replay file {}: {e}",
                path.display()
            );
            commands.insert_resource(PlaybackIndex::default());
            commands.insert_resource(PlaybackState::default());
        }
    }
}

fn load_mcap_file(
    path: &PathBuf,
) -> Result<(PlaybackIndex, PlaybackState), Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;

    let mut joint_frames = Vec::new();
    let mut action_frames = Vec::new();
    let mut reward_frames = Vec::new();

    for message in mcap::MessageStream::new(&data)? {
        let message = message?;
        let topic = &message.channel.topic;
        let payload = &message.data;

        match topic.as_str() {
            "/joint_states" => {
                if let Ok(frame) = serde_json::from_slice::<JointFrame>(payload) {
                    joint_frames.push(frame);
                }
            }
            "/actions" => {
                if let Ok(frame) = serde_json::from_slice::<ActionFrame>(payload) {
                    action_frames.push(frame);
                }
            }
            "/reward" => {
                if let Ok(frame) = serde_json::from_slice::<RewardFrame>(payload) {
                    reward_frames.push(frame);
                }
            }
            _ => {} // Ignore camera/unknown channels for now
        }
    }

    // Sort by timestamp.
    joint_frames.sort_by_key(|f| f.timestamp_ns);
    action_frames.sort_by_key(|f| f.timestamp_ns);
    reward_frames.sort_by_key(|f| f.timestamp_ns);

    // Determine duration from the max timestamp across all channels.
    let mut min_ns = u64::MAX;
    let mut max_ns = 0_u64;

    if let Some(f) = joint_frames.first() {
        min_ns = min_ns.min(f.timestamp_ns);
    }
    if let Some(f) = joint_frames.last() {
        max_ns = max_ns.max(f.timestamp_ns);
    }
    if let Some(f) = action_frames.first() {
        min_ns = min_ns.min(f.timestamp_ns);
    }
    if let Some(f) = action_frames.last() {
        max_ns = max_ns.max(f.timestamp_ns);
    }
    if let Some(f) = reward_frames.first() {
        min_ns = min_ns.min(f.timestamp_ns);
    }
    if let Some(f) = reward_frames.last() {
        max_ns = max_ns.max(f.timestamp_ns);
    }

    if min_ns == u64::MAX {
        min_ns = 0;
    }
    let duration_ns = max_ns.saturating_sub(min_ns);

    // Rebase timestamps to start from 0.
    for f in &mut joint_frames {
        f.timestamp_ns = f.timestamp_ns.saturating_sub(min_ns);
    }
    for f in &mut action_frames {
        f.timestamp_ns = f.timestamp_ns.saturating_sub(min_ns);
    }
    for f in &mut reward_frames {
        f.timestamp_ns = f.timestamp_ns.saturating_sub(min_ns);
    }

    let index = PlaybackIndex {
        joint_frames,
        action_frames,
        reward_frames,
    };
    let state = PlaybackState {
        cursor_ns: 0,
        duration_ns,
        playing: false,
        speed: 1.0,
    };

    Ok((index, state))
}

// ---------------------------------------------------------------------------
// Runtime Systems
// ---------------------------------------------------------------------------

/// Advances the playback cursor based on wall-clock delta time and speed.
///
/// Only runs when [`VizMode::Replay`] and [`PlaybackState::playing`].
#[allow(clippy::needless_pass_by_value)]
pub fn replay_advance_system(
    time: Res<Time>,
    mode: Res<crate::mode::VizMode>,
    mut state: ResMut<PlaybackState>,
) {
    if *mode != crate::mode::VizMode::Replay || !state.playing {
        return;
    }

    let delta_ns = (time.delta_secs_f64() * state.speed as f64 * 1_000_000_000.0) as u64;
    state.cursor_ns = (state.cursor_ns + delta_ns).min(state.duration_ns);

    // Auto-pause at end.
    if state.cursor_ns >= state.duration_ns {
        state.playing = false;
    }
}

/// Applies joint positions from the nearest [`JointFrame`] to matching ECS
/// entities. Uses binary search on the sorted `joint_frames` by timestamp.
#[allow(clippy::needless_pass_by_value)]
pub fn replay_apply_joints_system(
    mode: Res<crate::mode::VizMode>,
    state: Res<PlaybackState>,
    index: Res<PlaybackIndex>,
    mut joints: Query<(&Name, &mut JointState, &mut JointCommand)>,
) {
    if *mode != crate::mode::VizMode::Replay {
        return;
    }

    if index.joint_frames.is_empty() {
        return;
    }

    // Binary search for the frame closest to (but not exceeding) cursor_ns.
    let cursor = state.cursor_ns;
    let idx = match index
        .joint_frames
        .binary_search_by_key(&cursor, |f| f.timestamp_ns)
    {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1),
    };

    let frame = &index.joint_frames[idx];

    // Apply to matching joint entities by name.
    for (name, mut joint_state, mut joint_cmd) in &mut joints {
        let name_str = name.as_str();
        if let Some(pos) = frame.names.iter().position(|n| n == name_str) {
            if let Some(&position) = frame.positions.get(pos) {
                joint_state.position = position;
                joint_cmd.value = position;
            }
            if let Some(&velocity) = frame.velocities.get(pos) {
                joint_state.velocity = velocity;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playback_state_progress() {
        let mut s = PlaybackState {
            cursor_ns: 500,
            duration_ns: 1000,
            playing: false,
            speed: 1.0,
        };
        assert!((s.progress() - 0.5).abs() < 0.001);
        s.seek(0.75);
        assert_eq!(s.cursor_ns, 750);
    }

    #[test]
    fn playback_state_zero_duration() {
        let s = PlaybackState::default();
        assert!((s.progress() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn playback_index_default_empty() {
        let idx = PlaybackIndex::default();
        assert!(idx.joint_frames.is_empty());
        assert!(idx.action_frames.is_empty());
        assert!(idx.reward_frames.is_empty());
    }

    #[test]
    fn playback_state_time_secs() {
        let s = PlaybackState {
            cursor_ns: 2_500_000_000,
            duration_ns: 10_000_000_000,
            playing: true,
            speed: 1.0,
        };
        assert!((s.time_secs() - 2.5).abs() < 0.001);
        assert!((s.duration_secs() - 10.0).abs() < 0.001);
    }

    #[test]
    fn playback_state_seek_clamps() {
        let mut s = PlaybackState {
            cursor_ns: 0,
            duration_ns: 1000,
            playing: false,
            speed: 1.0,
        };
        s.seek(-0.5);
        assert_eq!(s.cursor_ns, 0);
        s.seek(1.5);
        assert_eq!(s.cursor_ns, 1000);
    }
}
