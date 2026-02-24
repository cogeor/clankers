# Task: Multi-Robot GUI Support and Example Preparation

## Context

The codebase has multi-robot infrastructure (SceneBuilder, RobotGroup, RobotId, robot-scoped
sensors, MultiRobotActionApplicator trait) but the GUI/viz system doesn't support selecting
or switching between robots. The existing `multi_robot.rs` example is headless only.

Training currently uses batch_size = n_envs (1 robot per env). The infrastructure for
n_robots per env exists (SceneBuilder, robot-scoped sensors) but GymEnv only uses single
ActionApplicator.

## Requirements

1. **GUI robot selection**: Add `SelectedRobotId` resource, robot selector in egui panel,
   filter joint display by selected robot, dynamic keyboard rebinding per robot.

2. **Multi-robot viz example**: Create `multi_robot_viz.rs` that visualizes multiple robots
   with the robot selection GUI, demonstrating the full multi-robot viz pipeline.

3. **Multi-robot training investigation**: Document what changes are needed in GymEnv
   for n_envs x n_bots training. Wire MultiRobotActionApplicator into GymEnv if feasible,
   or document the path forward.

## Acceptance Criteria

- GUI panel shows robot selector when multiple robots are in scene
- Joint display filtered to selected robot
- Keyboard teleop rebinds when switching robots
- Multi-robot viz example demonstrates 2+ robots with switching
- All existing tests pass
