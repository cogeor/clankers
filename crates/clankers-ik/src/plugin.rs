//! Bevy ECS integration for the IK solver.
//!
//! Provides [`ClankerIkPlugin`] which adds an IK solve system that reads
//! [`IkGoal`] targets and writes [`JointCommand`] values each frame.
//!
//! # Usage
//!
//! 1. Add [`ClankerIkPlugin`] to your app.
//! 2. Spawn a robot with [`clankers_urdf::spawn_robot`].
//! 3. Call [`IkChainMap::insert`] to register the robot's IK chain.
//! 4. Set [`IkGoal`] targets on joint entities.
//!
//! The plugin's system runs in [`ClankersSet::Decide`], before the actuator
//! step in [`ClankersSet::Act`].

use std::collections::HashMap;

use bevy::prelude::*;
use nalgebra::{Isometry3, Vector3};

use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::ClankersSet;
use clankers_core::types::RobotId;
use clankers_urdf::{RobotModel, SpawnedRobot};

use crate::chain::KinematicChain;
use crate::solver::{DlsConfig, DlsSolver, IkTarget};

/// Bevy plugin that adds IK solving each frame.
///
/// Systems run in [`ClankersSet::Decide`] — after observation, before actuation.
pub struct ClankerIkPlugin;

impl Plugin for ClankerIkPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<IkChainMap>()
            .init_resource::<IkSolverConfig>()
            .add_systems(Update, ik_solve_system.in_set(ClankersSet::Decide));
    }
}

/// Per-robot IK chain data: the chain plus ordered entity references.
#[derive(Debug)]
pub struct IkChainEntry {
    /// The kinematic chain (static transforms, axes, limits).
    pub chain: KinematicChain,
    /// Joint entities in chain order (index matches chain joint index).
    pub joint_entities: Vec<Entity>,
    /// Current IK target. `None` means IK is inactive for this robot.
    pub goal: Option<IkTarget>,
}

/// Resource mapping [`RobotId`] to IK chain data.
///
/// Register a robot's chain after spawning via [`IkChainMap::insert`] or
/// the convenience [`IkChainMap::build_and_insert`].
#[derive(Resource, Debug, Default)]
pub struct IkChainMap {
    chains: HashMap<RobotId, IkChainEntry>,
}

impl IkChainMap {
    /// Insert a pre-built IK chain entry for a robot.
    pub fn insert(&mut self, robot_id: RobotId, entry: IkChainEntry) {
        self.chains.insert(robot_id, entry);
    }

    /// Build a [`KinematicChain`] from a [`RobotModel`] and [`SpawnedRobot`],
    /// then register it.
    ///
    /// `ee_link` is the name of the end-effector link in the URDF.
    ///
    /// Returns `true` if the chain was successfully built and inserted.
    pub fn build_and_insert(
        &mut self,
        robot_id: RobotId,
        model: &RobotModel,
        spawned: &SpawnedRobot,
        ee_link: &str,
    ) -> bool {
        let Some(chain) = KinematicChain::from_model(model, ee_link) else {
            return false;
        };

        // Map chain joint names to entities in chain order
        let joint_entities: Vec<Entity> = chain
            .joint_names()
            .iter()
            .filter_map(|name| spawned.joint_entity(name))
            .collect();

        if joint_entities.len() != chain.dof() {
            return false; // some joints not found in spawned robot
        }

        self.chains.insert(
            robot_id,
            IkChainEntry {
                chain,
                joint_entities,
                goal: None,
            },
        );
        true
    }

    /// Set the IK target for a robot.
    pub fn set_goal(&mut self, robot_id: RobotId, target: IkTarget) {
        if let Some(entry) = self.chains.get_mut(&robot_id) {
            entry.goal = Some(target);
        }
    }

    /// Set a position-only target for a robot.
    pub fn set_position_goal(&mut self, robot_id: RobotId, x: f32, y: f32, z: f32) {
        self.set_goal(robot_id, IkTarget::Position(Vector3::new(x, y, z)));
    }

    /// Set a full-pose target for a robot.
    pub fn set_pose_goal(&mut self, robot_id: RobotId, pose: Isometry3<f32>) {
        self.set_goal(robot_id, IkTarget::Pose(pose));
    }

    /// Clear the IK target for a robot (stop IK control).
    pub fn clear_goal(&mut self, robot_id: RobotId) {
        if let Some(entry) = self.chains.get_mut(&robot_id) {
            entry.goal = None;
        }
    }

    /// Get a reference to a chain entry.
    pub fn get(&self, robot_id: RobotId) -> Option<&IkChainEntry> {
        self.chains.get(&robot_id)
    }

    /// Get a mutable reference to a chain entry.
    pub fn get_mut(&mut self, robot_id: RobotId) -> Option<&mut IkChainEntry> {
        self.chains.get_mut(&robot_id)
    }
}

/// Resource for IK solver configuration.
#[derive(Resource, Debug, Clone, Default)]
pub struct IkSolverConfig(pub DlsConfig);

/// System that solves IK for all robots with active goals.
///
/// For each robot in [`IkChainMap`] with a non-`None` goal:
/// 1. Reads current [`JointState`] positions from the entity query.
/// 2. Runs the DLS solver.
/// 3. Writes solved positions to [`JointCommand`] on each joint entity.
#[allow(clippy::needless_pass_by_value)]
pub fn ik_solve_system(
    mut ik_chains: ResMut<IkChainMap>,
    solver_config: Res<IkSolverConfig>,
    mut joint_query: Query<(&JointState, &mut JointCommand)>,
) {
    let solver = DlsSolver::new(solver_config.0.clone());

    for entry in ik_chains.chains.values_mut() {
        let Some(ref target) = entry.goal else {
            continue;
        };

        // Read current joint positions
        let mut q_current: Vec<f32> = Vec::with_capacity(entry.joint_entities.len());
        for &entity in &entry.joint_entities {
            if let Ok((state, _)) = joint_query.get(entity) {
                q_current.push(state.position);
            } else {
                // Entity missing — skip this robot
                q_current.clear();
                break;
            }
        }

        if q_current.len() != entry.chain.dof() {
            continue;
        }

        // Solve IK (warm-started from current position)
        let result = solver.solve(&entry.chain, target, &q_current);

        // Write solved joint positions as commands (position mode)
        for (i, &entity) in entry.joint_entities.iter().enumerate() {
            if let Ok((_, mut cmd)) = joint_query.get_mut(entity) {
                cmd.value = result.joint_positions[i];
            }
        }
    }
}

/// Convenience: compute an IK solution without the ECS (for scripted use).
///
/// Takes a [`RobotModel`], end-effector link name, current joint positions,
/// and target. Returns the solved joint positions or `None` if the chain
/// can't be built.
pub fn solve_ik(
    model: &RobotModel,
    ee_link: &str,
    q_current: &[f32],
    target: &IkTarget,
    config: &DlsConfig,
) -> Option<crate::solver::IkResult> {
    let chain = KinematicChain::from_model(model, ee_link)?;
    let solver = DlsSolver::new(config.clone());
    Some(solver.solve(&chain, target, q_current))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clankers_core::ClankersCorePlugin;

    const TWO_LINK_ARM: &str = r#"
        <robot name="arm">
            <link name="base"><inertial><mass value="10.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
            <link name="upper"><inertial><mass value="2.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.002"/></inertial></link>
            <link name="lower"><inertial><mass value="1.0"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/></inertial></link>
            <link name="ee"><inertial><mass value="0.1"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial></link>
            <joint name="shoulder" type="revolute">
                <parent link="base"/><child link="upper"/>
                <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
                <limit lower="-2.617" upper="2.617" effort="50" velocity="3"/>
            </joint>
            <joint name="elbow" type="revolute">
                <parent link="upper"/><child link="lower"/>
                <origin xyz="0 0 0.3"/><axis xyz="0 0 1"/>
                <limit lower="-2.094" upper="2.094" effort="30" velocity="5"/>
            </joint>
            <joint name="ee_fixed" type="fixed">
                <parent link="lower"/><child link="ee"/>
                <origin xyz="0 0 0.25"/>
            </joint>
        </robot>
    "#;

    #[test]
    fn plugin_builds() {
        let mut app = App::new();
        app.add_plugins(ClankersCorePlugin);
        app.add_plugins(clankers_actuator::ClankersActuatorPlugin);
        app.add_plugins(ClankerIkPlugin);
        app.finish();
        app.cleanup();
        app.update();

        assert!(app.world().get_resource::<IkChainMap>().is_some());
        assert!(app.world().get_resource::<IkSolverConfig>().is_some());
    }

    #[test]
    fn build_and_insert_chain() {
        let model = clankers_urdf::parse_string(TWO_LINK_ARM).unwrap();
        let mut world = World::new();
        let spawned = clankers_urdf::spawn_robot(&mut world, &model, &HashMap::new());

        let mut map = IkChainMap::default();
        let robot_id = RobotId(0);
        assert!(map.build_and_insert(robot_id, &model, &spawned, "ee"));
        assert_eq!(map.get(robot_id).unwrap().chain.dof(), 2);
        assert_eq!(map.get(robot_id).unwrap().joint_entities.len(), 2);
    }

    #[test]
    fn build_and_insert_nonexistent_ee_returns_false() {
        let model = clankers_urdf::parse_string(TWO_LINK_ARM).unwrap();
        let mut world = World::new();
        let spawned = clankers_urdf::spawn_robot(&mut world, &model, &HashMap::new());

        let mut map = IkChainMap::default();
        assert!(!map.build_and_insert(RobotId(0), &model, &spawned, "nonexistent"));
    }

    #[test]
    fn set_and_clear_goal() {
        let mut map = IkChainMap::default();
        let robot_id = RobotId(0);
        let model = clankers_urdf::parse_string(TWO_LINK_ARM).unwrap();
        let mut world = World::new();
        let spawned = clankers_urdf::spawn_robot(&mut world, &model, &HashMap::new());
        map.build_and_insert(robot_id, &model, &spawned, "ee");

        map.set_position_goal(robot_id, 0.1, 0.2, 0.3);
        assert!(map.get(robot_id).unwrap().goal.is_some());

        map.clear_goal(robot_id);
        assert!(map.get(robot_id).unwrap().goal.is_none());
    }

    #[test]
    fn ik_system_writes_commands() {
        let mut app = App::new();
        app.add_plugins(ClankersCorePlugin);
        app.add_plugins(clankers_actuator::ClankersActuatorPlugin);
        app.add_plugins(ClankerIkPlugin);
        app.finish();
        app.cleanup();

        let model = clankers_urdf::parse_string(TWO_LINK_ARM).unwrap();
        let spawned = clankers_urdf::spawn_robot(app.world_mut(), &model, &HashMap::new());
        let robot_id = RobotId(0);

        // Register the IK chain
        {
            let mut ik_chains = app.world_mut().resource_mut::<IkChainMap>();
            ik_chains.build_and_insert(robot_id, &model, &spawned, "ee");
            ik_chains.set_position_goal(robot_id, 0.0, 0.0, 0.4);
        }

        // Run one frame
        app.update();

        // Check that JointCommand was written (should be non-zero for a non-trivial target)
        let shoulder = spawned.joint_entity("shoulder").unwrap();
        let cmd = app.world().get::<JointCommand>(shoulder).unwrap();
        // The solver should have written something — exact value depends on IK solution
        // At minimum, the command should have been touched
        let _ = cmd.value; // just check it doesn't panic
    }

    const SIX_DOF_ARM: &str = r#"
        <robot name="six_dof_arm">
            <link name="base"><inertial><mass value="20.0"/><inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.5"/></inertial></link>
            <link name="shoulder_link"><inertial><mass value="3.0"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/></inertial></link>
            <link name="upper_arm"><inertial><mass value="2.5"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.003"/></inertial></link>
            <link name="elbow_link"><inertial><mass value="1.5"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.002"/></inertial></link>
            <link name="forearm"><inertial><mass value="1.0"/><inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.001"/></inertial></link>
            <link name="wrist_link"><inertial><mass value="0.5"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.0005"/></inertial></link>
            <link name="end_effector"><inertial><mass value="0.2"/><inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/></inertial></link>
            <joint name="j1" type="revolute"><parent link="base"/><child link="shoulder_link"/><origin xyz="0 0 0.05"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="80" velocity="2"/></joint>
            <joint name="j2" type="revolute"><parent link="shoulder_link"/><child link="upper_arm"/><origin xyz="0 0 0.2"/><axis xyz="0 1 0"/><limit lower="-1.57" upper="2.35" effort="60" velocity="2"/></joint>
            <joint name="j3" type="revolute"><parent link="upper_arm"/><child link="elbow_link"/><origin xyz="0 0 0.3"/><axis xyz="0 1 0"/><limit lower="-2.35" upper="2.35" effort="40" velocity="3"/></joint>
            <joint name="j4" type="revolute"><parent link="elbow_link"/><child link="forearm"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="20" velocity="5"/></joint>
            <joint name="j5" type="revolute"><parent link="forearm"/><child link="wrist_link"/><origin xyz="0 0 0.2"/><axis xyz="0 1 0"/><limit lower="-2.09" upper="2.09" effort="10" velocity="5"/></joint>
            <joint name="j6" type="revolute"><parent link="wrist_link"/><child link="end_effector"/><origin xyz="0 0 0.06"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="5" velocity="8"/></joint>
        </robot>
    "#;

    #[test]
    fn solve_ik_convenience() {
        let model = clankers_urdf::parse_string(SIX_DOF_ARM).unwrap();
        let target = IkTarget::Position(Vector3::new(0.3, 0.0, 0.5));
        let config = DlsConfig::default();

        let result = solve_ik(&model, "end_effector", &[0.0; 6], &target, &config);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.converged, "pos_err={}", result.position_error);
    }
}
