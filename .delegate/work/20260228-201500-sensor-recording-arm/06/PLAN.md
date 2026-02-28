# Loop 06: clankers-record crate — MCAP writer, RecorderPlugin

## Goal
Create a new crate for offline episode recording to MCAP format.

## Changes

### 1. `Cargo.toml` (workspace root)
- Add `clankers-record` to workspace members
- Add `mcap` as workspace dependency

### 2. `crates/clankers-record/Cargo.toml` (NEW)
- Deps: bevy, mcap, serde, serde_json, clankers-core, clankers-actuator
- Optional dep: clankers-render (feature "camera")

### 3. `crates/clankers-record/src/types.rs` (NEW)
- JointFrame { timestamp_ns, names, positions, velocities, torques }
- ImageFrame { timestamp_ns, width, height, data: Vec<u8> }
- ActionFrame { timestamp_ns, data: Vec<f32> }
- RewardFrame { timestamp_ns, reward: f32 }
- All derive Serialize, Deserialize

### 4. `crates/clankers-record/src/recorder.rs` (NEW)
- RecordingConfig resource: output_path, channels to enable
- Recorder resource wrapping mcap::Writer<BufWriter<File>>
- record_joint_states_system: queries JointState + Name + SimTime
- record_action_system: writes Action
- record_reward_system: writes scalar reward
- record_image_system (feature "camera"): reads CameraFrameBuffers

### 5. `crates/clankers-record/src/plugin.rs` (NEW)
- RecorderPlugin registers all systems in PostUpdate

### 6. `crates/clankers-record/src/lib.rs` (NEW)
- Re-exports
