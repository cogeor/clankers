# Implementation Log

## Task 1: Add `KeyboardTeleopMap::for_joint_count` helper

Completed: 2026-02-24T21:30:00Z

### Changes

- `crates/clankers-viz/src/input.rs`: Added `KeyboardTeleopMap::for_joint_count(n: usize)` method that creates bindings for exactly `n` joints (clamped to 0..=6), reusing the same key pairs as the default (Q/A, W/S, E/D, R/F, T/G, Y/H). Refactored `six_joint_default()` to delegate to `for_joint_count(6)`, eliminating the duplicated key-pair table.

### Verification

- [x] `cargo check -p clankers-viz` compiles with no errors or warnings
- [x] Existing tests still pass

### Notes

None.

---

## Task 2: Add `sync_teleop_to_robot` system

Completed: 2026-02-24T21:32:00Z

### Changes

- `crates/clankers-viz/src/systems.rs`: Added imports for `RobotGroup`, `RobotId`, `KeyboardTeleopMap`, and `SelectedRobotId`. Added `sync_teleop_to_robot` system that uses `Local<Option<Option<RobotId>>>` to detect changes in `SelectedRobotId`, rebuilds `KeyboardTeleopMap` and `TeleopConfig` targeting the selected robot's joints, and clears `TeleopCommander`. When `SelectedRobotId` is `None`, maps all joints from all robots sorted by `RobotId::index()`.

### Verification

- [x] `cargo check -p clankers-viz` compiles with no errors or warnings
- [x] System correctly rebuilds on robot switch (tested)
- [x] System maps all joints when no robot selected (tested)
- [x] System clears commander on switch (tested)
- [x] System preserves `TeleopConfig.enabled` across rebuilds (tested)
- [x] System is a no-op when selection has not changed (tested)

### Notes

None.

---

## Task 3: Schedule `sync_teleop_to_robot` in plugin

Completed: 2026-02-24T21:33:00Z

### Changes

- `crates/clankers-viz/src/plugin.rs`: Replaced the standalone `keyboard_teleop_system` scheduling with a chained tuple `(systems::sync_teleop_to_robot, input::keyboard_teleop_system).chain()` in `ClankersSet::Decide`, ensuring sync runs before keyboard input each frame, both before `apply_teleop_commands`.

### Verification

- [x] `cargo check -p clankers-viz` compiles with no errors or warnings
- [x] No scheduling ambiguity warnings

### Notes

None.

---

## Task 4: Add unit tests

Completed: 2026-02-24T21:34:00Z

### Changes

- `crates/clankers-viz/src/input.rs`: Added 4 tests to existing test module: `for_joint_count_zero`, `for_joint_count_three`, `for_joint_count_clamps_at_six`, `six_joint_default_equals_for_joint_count_six`.
- `crates/clankers-viz/src/systems.rs`: Added 5 tests to existing test module with helper functions `spawn_joint` and `build_sync_test_app`: `sync_rebuilds_on_robot_switch`, `sync_none_uses_all_joints`, `sync_clears_commander_on_switch`, `sync_preserves_enabled_flag`, `sync_no_change_is_noop`.

### Verification

- [x] `cargo test -p clankers-viz`: 21 passed, 0 failed (was 12, added 9 new tests)
- [x] `cargo test` (full workspace): all tests pass, no regressions

### Notes

Fixed unused import warning for `RobotId` in the test module (only `RobotGroup` was needed directly since `RobotId` comes from `RobotGroup` methods).

---
