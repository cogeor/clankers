# Loop 01: Add SelectedRobotId resource and robot selector UI with filtered joint display

## Overview

This loop adds multi-robot awareness to the `clankers-viz` UI panel. Currently
the side panel queries **all** joint entities and renders them in a flat list
with labels J0, J1, J2... without any concept of which robot owns which joint.

After this loop:
- A `SelectedRobotId` resource tracks which robot the user is inspecting.
- When multiple robots exist in the scene, a row of selector buttons appears in
  the side panel.
- The joints section filters by the selected robot and uses per-robot joint
  indices.
- Single-robot scenes are unaffected (the selector hides itself and all joints
  show as before).

## Tasks

### Task 1: Define SelectedRobotId resource

**Goal:** Create a new Bevy resource that stores the currently selected robot.
`None` means "show all robots" (backwards-compatible default for single-robot
scenes).

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `crates/clankers-viz/src/lib.rs` |
| MODIFY | `crates/clankers-viz/src/plugin.rs` |

**Steps:**
1. In `crates/clankers-viz/src/lib.rs`, add the resource definition after the
   existing `pub use` block:

   ```rust
   use clankers_core::types::RobotId;

   /// Resource tracking which robot is selected in the viz UI.
   /// `None` means show all robots (useful for single-robot scenes).
   #[derive(Resource, Default, Clone, Debug)]
   pub struct SelectedRobotId(pub Option<RobotId>);
   ```

   Also add a `pub use` for it (it is already in the crate root, so just the
   `use bevy::prelude::*;` import for `Resource` and the `use` for `RobotId`
   are needed). Because `lib.rs` currently does not import `bevy::prelude`, the
   struct may be better placed here with a targeted `use bevy::prelude::Resource;`
   import, or alternatively defined in a small new module. Keeping it in `lib.rs`
   is simplest since it is a single type with no methods.

2. In `crates/clankers-viz/src/plugin.rs`, inside `ClankersVizPlugin::build`,
   add `.init_resource::<crate::SelectedRobotId>()` alongside the existing
   resource inits (line 40 area). This ensures the resource is always present
   even if no robot has been spawned yet.

**Verify:** `cargo check -p clankers-viz` compiles with no errors.

### Task 2: Add robot selector section to the egui side panel

**Goal:** When the `RobotGroup` resource exists and contains more than one
robot, render a horizontal row of selectable buttons between the mode section
and the controls section so the user can pick which robot's joints to inspect.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `crates/clankers-viz/src/ui.rs` |

**Steps:**
1. Add imports at the top of `ui.rs`:
   ```rust
   use clankers_core::types::{RobotGroup, RobotId};
   use crate::SelectedRobotId;
   ```

2. Extend the `side_panel_system` function signature with two new parameters:
   ```rust
   robot_group: Option<Res<RobotGroup>>,
   mut selected_robot: ResMut<SelectedRobotId>,
   ```
   The `#[allow(clippy::too_many_arguments)]` attribute is already present on
   this function.

3. Inside the `SidePanel` closure, between the `mode_section(...)` call
   (line 47) and the `controls_section(...)` call (line 50), insert:
   ```rust
   robot_section(ui, robot_group.as_deref(), &mut selected_robot);
   ui.separator();
   ```

4. Define a new private function `robot_section`:
   ```rust
   fn robot_section(
       ui: &mut egui::Ui,
       robot_group: Option<&RobotGroup>,
       selected: &mut SelectedRobotId,
   ) {
       let Some(group) = robot_group else { return };
       if group.len() <= 1 {
           return;
       }

       ui.label("Robot");
       ui.horizontal(|ui| {
           // Sort by RobotId index for stable ordering.
           let mut robots: Vec<_> = group.iter().collect();
           robots.sort_by_key(|(id, _)| id.index());

           for (id, info) in &robots {
               let is_selected = selected.0 == Some(*id);
               let button = egui::Button::new(&info.name).selected(is_selected);
               if ui.add(button).clicked() {
                   selected.0 = if is_selected { None } else { Some(*id) };
               }
           }
       });
   }
   ```
   Clicking a selected robot deselects it (sets to `None` = show all), clicking
   an unselected robot selects it. This gives a toggle-style UX.

**Verify:** `cargo check -p clankers-viz` compiles. Visually confirm with
a multi-robot scene that the buttons appear and highlight correctly.

### Task 3: Filter joints by selected robot

**Goal:** When a robot is selected, only show that robot's joints in the joints
section. When no robot is selected (single-robot or deselected), show all joints
as before.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `crates/clankers-viz/src/ui.rs` |

**Steps:**
1. Change the joint query in `side_panel_system` from:
   ```rust
   joints: Query<(Entity, &JointCommand, &JointState, &JointTorque)>,
   ```
   to:
   ```rust
   joints: Query<(Entity, &JointCommand, &JointState, &JointTorque, Option<&RobotId>)>,
   ```

2. Update the call to `joints_section` to also pass the selected robot:
   ```rust
   joints_section(ui, &joints, &mut commander, *mode, &selected_robot);
   ```

3. Update the `joints_section` function signature:
   ```rust
   fn joints_section(
       ui: &mut egui::Ui,
       joints: &Query<(Entity, &JointCommand, &JointState, &JointTorque, Option<&RobotId>)>,
       commander: &mut ResMut<TeleopCommander>,
       mode: VizMode,
       selected: &SelectedRobotId,
   ) {
   ```

4. In the rendering loop, filter and re-index joints:
   ```rust
   // Collect and filter joints by selected robot.
   let filtered: Vec<_> = joints
       .iter()
       .filter(|(_, _, _, _, rid)| match selected.0 {
           Some(sel) => rid.map_or(false, |r| *r == sel),
           None => true,
       })
       .collect();

   if filtered.is_empty() {
       ui.label("No joints spawned.");
       return;
   }
   ```

5. Replace the existing `for (i, (_entity, cmd, state, torque)) in joints.iter().enumerate()`
   loop with iteration over `filtered`, using the filtered index for display:
   ```rust
   for (i, (_entity, cmd, state, torque, _rid)) in filtered.iter().enumerate() {
       ui.label(format!("J{i}"));
       // ... rest of joint rendering unchanged, but use `i` for channel name
   ```
   The teleop channel name `format!("joint_{i}")` uses the filtered index,
   which is correct: for a selected 6-DOF robot its joints are channels 0..5
   regardless of global entity ordering.

**Verify:** `cargo check -p clankers-viz` compiles. In a multi-robot scene,
selecting a robot should show only its joints. Deselecting shows all.

### Task 4: Add unit tests

**Goal:** Add tests verifying `SelectedRobotId` default behavior and that the
crate compiles and passes its existing test suite.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `crates/clankers-viz/src/lib.rs` |

**Steps:**
1. Add a `#[cfg(test)]` module in `crates/clankers-viz/src/lib.rs`:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn selected_robot_id_default_is_none() {
           let selected = SelectedRobotId::default();
           assert!(selected.0.is_none());
       }

       #[test]
       fn selected_robot_id_clone() {
           let a = SelectedRobotId(Some(clankers_core::types::RobotId(2)));
           let b = a.clone();
           assert_eq!(a.0, b.0);
       }
   }
   ```

2. Run `cargo test -p clankers-viz` and confirm all tests pass (including the
   existing tests in `systems.rs` and `input.rs`).

3. Run `cargo test` for the full workspace and confirm no regressions.

**Verify:** `cargo test -p clankers-viz` exits 0. `cargo test` exits 0.

## Acceptance Criteria

- [ ] `SelectedRobotId` resource is defined, defaults to `None`, and is
      initialized by `ClankersVizPlugin`
- [ ] Robot selector section appears in the side panel only when `RobotGroup`
      has more than one robot
- [ ] Clicking a robot button sets `SelectedRobotId`; clicking again deselects
- [ ] Joints section filters by selected robot when one is chosen
- [ ] Joints section shows all joints when no robot is selected (backwards
      compatible)
- [ ] Joint labels use filtered index (J0, J1, ...) relative to the displayed
      set
- [ ] `SelectedRobotId` is re-exported from `crates/clankers-viz/src/lib.rs`
- [ ] `cargo check -p clankers-viz` passes
- [ ] `cargo test -p clankers-viz` passes
- [ ] `cargo test` (full workspace) passes
