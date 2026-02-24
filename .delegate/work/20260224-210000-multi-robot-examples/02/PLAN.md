# Loop 02: Rebuild KeyboardTeleopMap and TeleopConfig when selected robot changes

## Overview

When the user switches the selected robot in the UI (via `SelectedRobotId`), the teleop system must remap keyboard bindings and joint-entity mappings to target the new robot's joints. This loop adds:

1. A helper method on `KeyboardTeleopMap` to create bindings for an arbitrary joint count (up to 6).
2. A new `sync_teleop_to_robot` system that watches `SelectedRobotId` for changes and rebuilds `KeyboardTeleopMap`, `TeleopConfig`, and clears `TeleopCommander`.
3. Scheduling of the new system in `ClankersVizPlugin`.
4. Unit tests for all new logic.

## Tasks

### Task 1: Add `KeyboardTeleopMap::for_joint_count` helper

**Goal:** Add a constructor that creates bindings for exactly `n` joints (clamped to 0..=6), reusing the same key pairs as `six_joint_default`. This avoids duplicating the key-pair table.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `crates/clankers-viz/src/input.rs` |

**Steps:**
1. In the `impl KeyboardTeleopMap` block (after `six_joint_default`, around line 66), add a new public method:
   ```rust
   /// Create a mapping for exactly `n` joints (clamped to 0..=6).
   ///
   /// Uses the same key pairs as [`six_joint_default`](Self::six_joint_default):
   /// Q/A, W/S, E/D, R/F, T/G, Y/H -- truncated to the first `n`.
   #[must_use]
   pub fn for_joint_count(n: usize) -> Self {
       let pairs = [
           (KeyCode::KeyQ, KeyCode::KeyA),
           (KeyCode::KeyW, KeyCode::KeyS),
           (KeyCode::KeyE, KeyCode::KeyD),
           (KeyCode::KeyR, KeyCode::KeyF),
           (KeyCode::KeyT, KeyCode::KeyG),
           (KeyCode::KeyY, KeyCode::KeyH),
       ];

       let count = n.min(pairs.len());
       let bindings = pairs[..count]
           .iter()
           .enumerate()
           .map(|(i, &(pos, neg))| KeyboardJointBinding {
               channel: format!("joint_{i}"),
               key_positive: pos,
               key_negative: neg,
           })
           .collect();

       Self {
           bindings,
           increment: 0.05,
       }
   }
   ```
2. Optionally refactor `six_joint_default` to delegate to `for_joint_count(6)` to eliminate the duplicated key-pair array:
   ```rust
   pub fn six_joint_default() -> Self {
       Self::for_joint_count(6)
   }
   ```

**Verify:** `cargo check -p clankers-viz` compiles. Existing tests in `input::tests` still pass (`cargo test -p clankers-viz`).

---

### Task 2: Add `sync_teleop_to_robot` system

**Goal:** Create a new Bevy system that detects when `SelectedRobotId` changes and rebuilds the teleop resources to target the selected robot's joints.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `crates/clankers-viz/src/systems.rs` |

**Steps:**
1. Add the following imports at the top of `systems.rs` (some may already be present):
   ```rust
   use clankers_core::types::{RobotGroup, RobotId};
   use crate::input::KeyboardTeleopMap;
   use crate::SelectedRobotId;
   ```
2. Add the new system function after `mode_transition_system` (before the tests module):
   ```rust
   /// Rebuilds [`KeyboardTeleopMap`] and [`TeleopConfig`] when the selected
   /// robot changes, so keyboard teleop targets the correct joints.
   ///
   /// When `SelectedRobotId` is `None`, maps ALL joints from ALL robots
   /// (sorted by `RobotId` index) for backwards compatibility with
   /// single-robot scenes.
   #[allow(clippy::needless_pass_by_value)]
   pub fn sync_teleop_to_robot(
       selected: Res<SelectedRobotId>,
       robot_group: Option<Res<RobotGroup>>,
       mut teleop_config: ResMut<TeleopConfig>,
       mut teleop_map: ResMut<KeyboardTeleopMap>,
       mut commander: ResMut<TeleopCommander>,
       mut last_selected: Local<Option<Option<RobotId>>>,
   ) {
       // Determine current selection.
       let current = selected.0;

       // Check if this is the first run or if the selection changed.
       if *last_selected == Some(current) {
           return;
       }
       *last_selected = Some(current);

       // Collect the joint entities for the target robot(s).
       let joints: Vec<Entity> = match (current, robot_group.as_deref()) {
           // A specific robot is selected and the group exists.
           (Some(id), Some(group)) => {
               group
                   .get(id)
                   .map(|info| info.joints.clone())
                   .unwrap_or_default()
           }
           // No robot selected -- gather all joints, sorted by RobotId.
           (None, Some(group)) => {
               let mut robots: Vec<_> = group.iter().collect();
               robots.sort_by_key(|(id, _)| id.index());
               robots.iter().flat_map(|(_, info)| info.joints.iter().copied()).collect()
           }
           // No RobotGroup resource at all -- nothing to map.
           (_, None) => Vec::new(),
       };

       // Rebuild KeyboardTeleopMap for the joint count.
       *teleop_map = KeyboardTeleopMap::for_joint_count(joints.len());

       // Rebuild TeleopConfig mappings.
       let enabled = teleop_config.enabled;
       let mut new_config = TeleopConfig::new();
       for (i, &entity) in joints.iter().enumerate() {
           new_config = new_config.with_mapping(
               format!("joint_{i}"),
               clankers_teleop::config::JointMapping::new(entity),
           );
       }
       new_config.enabled = enabled;
       *teleop_config = new_config;

       // Clear stale commander values from previous robot.
       commander.clear();
   }
   ```

**Key design decisions:**
- `Local<Option<Option<RobotId>>>`: Outer `Option` is `None` on first run (forces initial sync). Inner `Option<RobotId>` mirrors `SelectedRobotId.0`.
- Preserves `TeleopConfig.enabled` across rebuilds so mode gating is not disrupted.
- Sorts robots by `RobotId::index()` in the "all robots" fallback for deterministic ordering (matching the UI's sort in `robot_section`).

**Verify:** `cargo check -p clankers-viz` compiles.

---

### Task 3: Schedule `sync_teleop_to_robot` in plugin

**Goal:** Add the new system to `ClankersVizPlugin::build` so it runs before `keyboard_teleop_system` each frame.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `crates/clankers-viz/src/plugin.rs` |

**Steps:**
1. In `ClankersVizPlugin::build`, add `sync_teleop_to_robot` to the Update schedule. It must run:
   - In `ClankersSet::Decide` (same set as `keyboard_teleop_system`)
   - Before `keyboard_teleop_system`
   - Before `apply_teleop_commands`

   Replace the existing keyboard input scheduling block (lines 72-77):
   ```rust
   // Keyboard input: in Decide, before teleop apply.
   app.add_systems(
       Update,
       input::keyboard_teleop_system
           .in_set(ClankersSet::Decide)
           .before(clankers_teleop::systems::apply_teleop_commands),
   );
   ```
   With:
   ```rust
   // Teleop sync + keyboard input: in Decide, before teleop apply.
   app.add_systems(
       Update,
       (
           systems::sync_teleop_to_robot,
           input::keyboard_teleop_system,
       )
           .chain()
           .in_set(ClankersSet::Decide)
           .before(clankers_teleop::systems::apply_teleop_commands),
   );
   ```
   Using `.chain()` guarantees `sync_teleop_to_robot` runs before `keyboard_teleop_system`.

**Verify:** `cargo check -p clankers-viz` compiles. No scheduling ambiguity warnings.

---

### Task 4: Add unit tests

**Goal:** Add tests covering the new `for_joint_count` helper and the `sync_teleop_to_robot` system.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `crates/clankers-viz/src/input.rs` |
| MODIFY | `crates/clankers-viz/src/systems.rs` |

**Steps:**

1. In `crates/clankers-viz/src/input.rs`, add tests to the existing `#[cfg(test)] mod tests` block:
   ```rust
   #[test]
   fn for_joint_count_zero() {
       let map = KeyboardTeleopMap::for_joint_count(0);
       assert_eq!(map.bindings.len(), 0);
   }

   #[test]
   fn for_joint_count_three() {
       let map = KeyboardTeleopMap::for_joint_count(3);
       assert_eq!(map.bindings.len(), 3);
       assert_eq!(map.bindings[0].channel, "joint_0");
       assert_eq!(map.bindings[2].channel, "joint_2");
       assert_eq!(map.bindings[0].key_positive, KeyCode::KeyQ);
       assert_eq!(map.bindings[2].key_positive, KeyCode::KeyE);
   }

   #[test]
   fn for_joint_count_clamps_at_six() {
       let map = KeyboardTeleopMap::for_joint_count(10);
       assert_eq!(map.bindings.len(), 6);
   }

   #[test]
   fn six_joint_default_equals_for_joint_count_six() {
       let a = KeyboardTeleopMap::six_joint_default();
       let b = KeyboardTeleopMap::for_joint_count(6);
       assert_eq!(a.bindings.len(), b.bindings.len());
       for (ba, bb) in a.bindings.iter().zip(b.bindings.iter()) {
           assert_eq!(ba.channel, bb.channel);
           assert_eq!(ba.key_positive, bb.key_positive);
           assert_eq!(ba.key_negative, bb.key_negative);
       }
   }
   ```

2. In `crates/clankers-viz/src/systems.rs`, add tests to the existing `#[cfg(test)] mod tests` block. The tests need a helper that spawns joint entities, registers them in a `RobotGroup`, and inserts the resources:
   ```rust
   use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
   use clankers_core::types::{RobotGroup, RobotId};
   use crate::input::KeyboardTeleopMap;
   use crate::SelectedRobotId;

   fn spawn_joint(world: &mut World) -> Entity {
       world
           .spawn((
               Actuator::default(),
               JointCommand::default(),
               JointState::default(),
               JointTorque::default(),
           ))
           .id()
   }

   fn build_sync_test_app() -> App {
       let mut app = App::new();
       app.add_plugins(clankers_core::ClankersCorePlugin);
       app.add_plugins(clankers_teleop::ClankersTeleopPlugin);
       app.init_resource::<VizConfig>();
       app.init_resource::<VizMode>();
       app.init_resource::<VizSimGate>();
       app.init_resource::<KeyboardTeleopMap>();
       app.init_resource::<SelectedRobotId>();
       app.init_resource::<RobotGroup>();
       app.add_systems(Update, sync_teleop_to_robot);
       app.finish();
       app.cleanup();
       app
   }

   #[test]
   fn sync_rebuilds_on_robot_switch() {
       let mut app = build_sync_test_app();

       // Register two robots with different joint counts.
       let j0 = spawn_joint(app.world_mut());
       let j1 = spawn_joint(app.world_mut());
       let j2 = spawn_joint(app.world_mut());

       let mut group = app.world_mut().resource_mut::<RobotGroup>();
       let id_a = group.allocate("arm".to_string(), vec![j0, j1]);
       let _id_b = group.allocate("gripper".to_string(), vec![j2]);

       // Select robot A.
       app.world_mut().resource_mut::<SelectedRobotId>().0 = Some(id_a);
       app.update();

       // Should have 2 bindings and 2 mappings.
       assert_eq!(app.world().resource::<KeyboardTeleopMap>().bindings.len(), 2);
       assert_eq!(app.world().resource::<TeleopConfig>().mappings.len(), 2);
   }

   #[test]
   fn sync_none_uses_all_joints() {
       let mut app = build_sync_test_app();

       let j0 = spawn_joint(app.world_mut());
       let j1 = spawn_joint(app.world_mut());
       let j2 = spawn_joint(app.world_mut());

       let mut group = app.world_mut().resource_mut::<RobotGroup>();
       group.allocate("arm".to_string(), vec![j0, j1]);
       group.allocate("gripper".to_string(), vec![j2]);

       // No selection => all joints.
       app.world_mut().resource_mut::<SelectedRobotId>().0 = None;
       app.update();

       assert_eq!(app.world().resource::<KeyboardTeleopMap>().bindings.len(), 3);
       assert_eq!(app.world().resource::<TeleopConfig>().mappings.len(), 3);
   }

   #[test]
   fn sync_clears_commander_on_switch() {
       let mut app = build_sync_test_app();

       let j0 = spawn_joint(app.world_mut());
       let j1 = spawn_joint(app.world_mut());

       let mut group = app.world_mut().resource_mut::<RobotGroup>();
       let id_a = group.allocate("arm".to_string(), vec![j0]);
       let id_b = group.allocate("leg".to_string(), vec![j1]);

       // Select A, run, set a commander value.
       app.world_mut().resource_mut::<SelectedRobotId>().0 = Some(id_a);
       app.update();
       app.world_mut().resource_mut::<TeleopCommander>().set("joint_0", 0.5);

       // Switch to B.
       app.world_mut().resource_mut::<SelectedRobotId>().0 = Some(id_b);
       app.update();

       assert_eq!(app.world().resource::<TeleopCommander>().channel_count(), 0);
   }

   #[test]
   fn sync_preserves_enabled_flag() {
       let mut app = build_sync_test_app();

       let j0 = spawn_joint(app.world_mut());
       let mut group = app.world_mut().resource_mut::<RobotGroup>();
       let id_a = group.allocate("arm".to_string(), vec![j0]);

       // Disable teleop, then trigger sync.
       app.world_mut().resource_mut::<TeleopConfig>().enabled = false;
       app.world_mut().resource_mut::<SelectedRobotId>().0 = Some(id_a);
       app.update();

       assert!(!app.world().resource::<TeleopConfig>().enabled);
   }

   #[test]
   fn sync_no_change_is_noop() {
       let mut app = build_sync_test_app();

       let j0 = spawn_joint(app.world_mut());
       let mut group = app.world_mut().resource_mut::<RobotGroup>();
       let id_a = group.allocate("arm".to_string(), vec![j0]);

       app.world_mut().resource_mut::<SelectedRobotId>().0 = Some(id_a);
       app.update();
       // Set a commander value.
       app.world_mut().resource_mut::<TeleopCommander>().set("joint_0", 0.7);

       // Run again with same selection -- commander should NOT be cleared.
       app.update();
       assert!((app.world().resource::<TeleopCommander>().get("joint_0") - 0.7).abs() < f32::EPSILON);
   }
   ```

**Verify:** `cargo test -p clankers-viz` -- all existing tests pass plus the new ones. `cargo test` workspace-wide has no regressions.

## Acceptance Criteria

- [ ] `KeyboardTeleopMap::for_joint_count(n)` creates correct bindings for 0..=6 joints and clamps at 6
- [ ] `sync_teleop_to_robot` rebuilds `TeleopConfig` and `KeyboardTeleopMap` when `SelectedRobotId` changes
- [ ] `sync_teleop_to_robot` clears `TeleopCommander` on robot switch
- [ ] `sync_teleop_to_robot` preserves `TeleopConfig.enabled` across rebuilds
- [ ] `sync_teleop_to_robot` does nothing when selection has not changed (no spurious clears)
- [ ] When `SelectedRobotId` is `None`, all joints from all robots are mapped (sorted by `RobotId`)
- [ ] System is scheduled before `keyboard_teleop_system` via `.chain()` in `ClankersSet::Decide`
- [ ] `cargo check -p clankers-viz` compiles with no errors or warnings
- [ ] `cargo test -p clankers-viz` passes all tests (existing + new)
- [ ] `cargo test` workspace-wide passes with no regressions
