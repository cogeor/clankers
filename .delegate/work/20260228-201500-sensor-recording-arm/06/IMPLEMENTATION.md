# Implementation Log

## Task 1: Workspace Cargo.toml

Completed: 2026-02-28T20:30:00Z

### Changes

- `Cargo.toml`: Added `mcap = "0.24"` to `[workspace.dependencies]`. Added `clankers-record = { path = "crates/clankers-record" }` to the internal crates list. The workspace glob `members = ["apps/*", "crates/*", "examples"]` already covers the new crate directory.

### Verification

- [x] Workspace builds without errors after adding `mcap` dependency
- [x] `clankers-record` appears as a member through the existing `crates/*` glob

### Notes

- Used `mcap = "0.24"` (latest verified version 0.24.0). The mcap crate is not `Send` due to internal `Compressor` enum, which required using Bevy non-send resources instead of standard `Resource` trait.

---

## Task 2: crates/clankers-record/Cargo.toml

Completed: 2026-02-28T20:30:00Z

### Changes

- `crates/clankers-record/Cargo.toml`: Created new file with `bevy_log` feature enabled on bevy (needed because workspace bevy uses `default-features = false`), plus `mcap`, `serde`, `serde_json`, `clankers-core`, `clankers-actuator`. Optional `clankers-render` behind feature `camera`.

### Verification

- [x] Crate compiles as workspace member
- [x] Feature `camera` compiles conditionally

---

## Task 3: crates/clankers-record/src/types.rs

Completed: 2026-02-28T20:30:00Z

### Changes

- `crates/clankers-record/src/types.rs`: Created with four frame types:
  - `JointFrame` — timestamp_ns, names, positions, velocities, torques
  - `ActionFrame` — timestamp_ns, data: Vec<f32>
  - `RewardFrame` — timestamp_ns, reward: f32
  - `ImageFrame` — timestamp_ns, width, height, label, data: Vec<u8>
  - All derive `Serialize`, `Deserialize`, `Debug`, `Clone`, `PartialEq`
  - `ImageFrame` also derives `Eq` (all fields are `Eq`)
  - Added `#[allow(clippy::derive_partial_eq_without_eq)]` on f32-containing types

### Verification

- [x] All 4 serde roundtrip tests pass

---

## Task 4: crates/clankers-record/src/recorder.rs

Completed: 2026-02-28T20:30:00Z

### Changes

- `crates/clankers-record/src/recorder.rs`: Created with:
  - `RecordingConfig` resource (output_path, record_joints, record_actions, record_rewards)
  - `Recorder` non-send resource wrapping `mcap::Writer<BufWriter<File>>`
  - `ChannelIds` resource holding pre-registered MCAP channel IDs
  - `NeedsChannelSetup` marker resource for one-shot channel registration
  - `PendingReward` and `PendingAction` resources for external systems to inject data
  - `setup_recorder` exclusive world system (opens file, inserts non-send resource)
  - `setup_channels` system (registers JSON schema + 3 MCAP channels on first frame)
  - `record_joint_states_system` (queries Name, JointState, JointTorque)
  - `record_action_system` (reads PendingAction)
  - `record_reward_system` (reads PendingReward)
  - `camera::record_image_system` behind `#[cfg(feature = "camera")]`

### Deviation

The `mcap::Writer` is not `Send` (contains internal `Compressor` enum that is !Send). Since the workspace forbids `unsafe_code`, `unsafe impl Send` is not possible. The `Recorder` is therefore inserted as a non-send resource via `world.insert_non_send_resource()`, and all systems use `Option<NonSendMut<Recorder>>`. This implicitly constrains recording systems to the main thread, which is acceptable for file I/O.

### Verification

- [x] Compiles without errors
- [x] Clippy passes without warnings (used `#[allow(clippy::needless_pass_by_value)]` for Bevy system params, following the same pattern as `clankers-actuator`)

---

## Task 5: crates/clankers-record/src/plugin.rs

Completed: 2026-02-28T20:30:00Z

### Changes

- `crates/clankers-record/src/plugin.rs`: Created `RecorderPlugin` implementing `bevy::prelude::Plugin`:
  - Initializes `RecordingConfig`, `PendingReward`, `PendingAction` resources
  - Adds `setup_recorder` to `Startup` schedule (exclusive world system)
  - Adds `setup_channels`, `record_joint_states_system`, `record_action_system`, `record_reward_system` to `PostUpdate` in a chain
  - Under `camera` feature: initializes `CameraChannelIds` and adds `record_image_system` to `PostUpdate`

### Verification

- [x] Plugin adds to `App` without panic (verified by `recorder_plugin_builds_without_panic` test)

---

## Task 6: crates/clankers-record/src/lib.rs

Completed: 2026-02-28T20:30:00Z

### Changes

- `crates/clankers-record/src/lib.rs`: Created module declarations (`plugin`, `recorder`, `types`) and `prelude` module re-exporting `RecorderPlugin`, `RecordingConfig`, `Recorder`, `PendingAction`, `PendingReward`, and all frame types. Added integration tests:
  - `recorder_plugin_builds_without_panic`: adds `RecorderPlugin` to a `MinimalPlugins` app
  - `joint_frame_write_read_roundtrip`: serde round-trip of `JointFrame`

### Verification

- [x] 6/6 unit tests pass
- [x] 2/2 doc-tests pass
- [x] Entire workspace builds successfully: `Finished dev profile [unoptimized + debuginfo] target(s) in 5m 50s`
- [x] `cargo clippy -p clankers-record` produces zero warnings

---
