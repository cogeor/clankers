# Test Results

Tested: 2026-02-24T22:10:00Z
Status: PASS

## Task Verification

- [x] Task 1 (`for_joint_count` helper): `KeyboardTeleopMap::for_joint_count` is present at line 42 of `input.rs` with the correct signature `pub fn for_joint_count(n: usize) -> Self`. It clamps to 0..=6 using `n.min(pairs.len())`. `six_joint_default` at line 74 delegates to `Self::for_joint_count(6)`, eliminating duplicated key pairs.
- [x] Task 2 (`sync_teleop_to_robot` system): System is present at line 285 of `systems.rs` with the correct signature using `Res<SelectedRobotId>`, `Option<Res<RobotGroup>>`, `ResMut<TeleopConfig>`, `ResMut<KeyboardTeleopMap>`, `ResMut<TeleopCommander>`, and `Local<Option<Option<RobotId>>>`. Uses early-return when selection unchanged. Rebuilds map and config, clears commander, preserves `enabled` flag.
- [x] Task 3 (plugin scheduling): `plugin.rs` lines 72-81 schedule `(systems::sync_teleop_to_robot, input::keyboard_teleop_system).chain()` in `ClankersSet::Decide` before `apply_teleop_commands`, ensuring sync runs before keyboard input.
- [x] Task 4 (unit tests): 4 new tests in `input.rs` (`for_joint_count_zero`, `for_joint_count_three`, `for_joint_count_clamps_at_six`, `six_joint_default_equals_for_joint_count_six`) and 5 new tests in `systems.rs` (`sync_rebuilds_on_robot_switch`, `sync_none_uses_all_joints`, `sync_clears_commander_on_switch`, `sync_preserves_enabled_flag`, `sync_no_change_is_noop`) with `spawn_joint` and `build_sync_test_app` helpers.

## Acceptance Criteria

- [x] `KeyboardTeleopMap::for_joint_count(n)` creates correct bindings for 0..=6 joints and clamps at 6: Verified in source and confirmed by `for_joint_count_zero`, `for_joint_count_three`, `for_joint_count_clamps_at_six` tests passing
- [x] `sync_teleop_to_robot` rebuilds `TeleopConfig` and `KeyboardTeleopMap` when `SelectedRobotId` changes: Confirmed by `sync_rebuilds_on_robot_switch` test passing (2 bindings and 2 mappings for a 2-joint robot)
- [x] `sync_teleop_to_robot` clears `TeleopCommander` on robot switch: Confirmed by `sync_clears_commander_on_switch` test passing (channel_count == 0 after switch)
- [x] `sync_teleop_to_robot` preserves `TeleopConfig.enabled` across rebuilds: Confirmed by `sync_preserves_enabled_flag` test passing (enabled stays false)
- [x] `sync_teleop_to_robot` does nothing when selection has not changed (no spurious clears): Confirmed by `sync_no_change_is_noop` test passing (commander value 0.7 preserved)
- [x] When `SelectedRobotId` is `None`, all joints from all robots are mapped (sorted by `RobotId`): Confirmed by `sync_none_uses_all_joints` test passing (3 bindings and 3 mappings for 2+1 joints)
- [x] System is scheduled before `keyboard_teleop_system` via `.chain()` in `ClankersSet::Decide`: Verified in `plugin.rs` lines 74-78 using `.chain()` tuple ordering
- [x] `cargo check -p clankers-viz` compiles with no errors or warnings: PASS
- [x] `cargo test -p clankers-viz` passes all tests (existing + new): 21 passed, 0 failed (+ 1 doctest)
- [x] `cargo test` workspace-wide passes with no regressions: 874 passed, 0 failed across 43 test suites

## Build & Tests

- Build: OK (`cargo check -p clankers-viz` finished with no errors or warnings)
- Tests: 21/21 (clankers-viz), 874/874 (workspace)

## Scope Check

- [x] Single logical purpose: All changes are confined to 3 files within `crates/clankers-viz/src/` (input.rs, systems.rs, plugin.rs), adding teleop rebinding when the selected robot changes
- [x] No unrelated modules touched
- [x] No unrelated refactoring mixed in (the only refactoring is `six_joint_default` delegating to `for_joint_count`, which is directly related)

---

Ready for Commit: yes
Commit Message: feat(viz): rebuild teleop bindings when selected robot changes
