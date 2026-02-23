//! Scene builder for constructing a fully configured Bevy [`App`].
//!
//! [`SceneBuilder`] provides a fluent API for composing a simulation:
//! robots from URDF, episode settings, stats tracking, and all core plugins.
//!
//! # Example
//!
//! ```no_run
//! use clankers_sim::SceneBuilder;
//!
//! let app = SceneBuilder::new()
//!     .with_max_episode_steps(500)
//!     .build();
//! ```

use std::collections::HashMap;

use bevy::prelude::*;
use clankers_core::config::SimConfig;
use clankers_env::episode::EpisodeConfig;
use clankers_urdf::spawner::SpawnedRobot;
use clankers_urdf::types::RobotModel;

use crate::ClankersSimPlugin;

// ---------------------------------------------------------------------------
// SpawnedScene
// ---------------------------------------------------------------------------

/// Result of building a scene — the Bevy app plus spawned robot handles.
pub struct SpawnedScene {
    /// The fully configured Bevy application.
    pub app: App,
    /// Spawned robots, keyed by the robot model name.
    pub robots: HashMap<String, SpawnedRobot>,
}

// ---------------------------------------------------------------------------
// RobotEntry
// ---------------------------------------------------------------------------

/// Internal representation of a robot to spawn.
struct RobotEntry {
    model: RobotModel,
    initial_positions: HashMap<String, f32>,
}

// ---------------------------------------------------------------------------
// SceneBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a complete Clankers simulation.
///
/// Configures plugins, robots, episode settings, and returns a ready-to-run
/// [`SpawnedScene`].
pub struct SceneBuilder {
    sim_config: Option<SimConfig>,
    episode_config: Option<EpisodeConfig>,
    robots: Vec<RobotEntry>,
}

impl Default for SceneBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneBuilder {
    /// Create a new scene builder with default settings.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sim_config: None,
            episode_config: None,
            robots: Vec::new(),
        }
    }

    /// Set the simulation configuration.
    #[must_use]
    pub const fn with_sim_config(mut self, config: SimConfig) -> Self {
        self.sim_config = Some(config);
        self
    }

    /// Set the episode configuration.
    #[must_use]
    pub const fn with_episode_config(mut self, config: EpisodeConfig) -> Self {
        self.episode_config = Some(config);
        self
    }

    /// Set the maximum episode steps (convenience for common case).
    #[must_use]
    pub fn with_max_episode_steps(mut self, max_steps: u32) -> Self {
        self.episode_config = Some(
            self.episode_config
                .unwrap_or_default()
                .with_max_steps(max_steps),
        );
        self
    }

    /// Add a robot from a parsed [`RobotModel`] with optional initial joint positions.
    #[must_use]
    pub fn with_robot(
        mut self,
        model: RobotModel,
        initial_positions: HashMap<String, f32>,
    ) -> Self {
        self.robots.push(RobotEntry {
            model,
            initial_positions,
        });
        self
    }

    /// Add a robot from a URDF XML string.
    ///
    /// # Errors
    ///
    /// Returns [`clankers_urdf::UrdfError`] if parsing fails.
    pub fn with_robot_urdf(
        self,
        urdf_xml: &str,
        initial_positions: HashMap<String, f32>,
    ) -> Result<Self, clankers_urdf::UrdfError> {
        let model = clankers_urdf::parse_string(urdf_xml)?;
        Ok(self.with_robot(model, initial_positions))
    }

    /// Build the Bevy [`App`] with all plugins and spawned entities.
    ///
    /// Returns a [`SpawnedScene`] containing the app and robot handles.
    #[must_use]
    pub fn build(self) -> SpawnedScene {
        let mut app = App::new();
        app.add_plugins(ClankersSimPlugin);

        // Apply custom SimConfig if provided.
        if let Some(config) = self.sim_config {
            *app.world_mut().resource_mut::<SimConfig>() = config;
        }

        // Apply custom EpisodeConfig if provided.
        if let Some(config) = self.episode_config {
            *app.world_mut().resource_mut::<EpisodeConfig>() = config;
        }

        // Finalize plugin setup before spawning entities.
        app.finish();
        app.cleanup();

        // Spawn robots.
        let mut robots = HashMap::new();
        for entry in self.robots {
            let spawned =
                clankers_urdf::spawn_robot(app.world_mut(), &entry.model, &entry.initial_positions);
            robots.insert(spawned.name.clone(), spawned);
        }

        SpawnedScene { app, robots }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EpisodeStats;
    use clankers_env::episode::Episode;

    const TWO_JOINT_URDF: &str = r#"
        <robot name="test_bot">
            <link name="base"/>
            <link name="link1"/>
            <link name="link2"/>
            <joint name="joint1" type="revolute">
                <parent link="base"/>
                <child link="link1"/>
                <axis xyz="0 0 1"/>
                <limit lower="-1.57" upper="1.57" effort="50" velocity="3"/>
            </joint>
            <joint name="joint2" type="revolute">
                <parent link="link1"/>
                <child link="link2"/>
                <axis xyz="0 1 0"/>
                <limit lower="-2.0" upper="2.0" effort="30" velocity="5"/>
            </joint>
        </robot>
    "#;

    #[test]
    fn build_empty_scene() {
        let scene = SceneBuilder::new().build();
        assert!(scene.robots.is_empty());
        assert!(scene.app.world().get_resource::<EpisodeStats>().is_some());
    }

    #[test]
    fn build_with_max_steps() {
        let scene = SceneBuilder::new().with_max_episode_steps(200).build();
        let config = scene.app.world().resource::<EpisodeConfig>();
        assert_eq!(config.max_episode_steps, 200);
    }

    #[test]
    fn build_with_sim_config() {
        let config = SimConfig {
            physics_dt: 0.005,
            control_dt: 0.05,
            ..SimConfig::default()
        };
        let scene = SceneBuilder::new().with_sim_config(config).build();
        let cfg = scene.app.world().resource::<SimConfig>();
        assert!((cfg.physics_dt - 0.005).abs() < f64::EPSILON);
        assert!((cfg.control_dt - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn build_with_robot_urdf() {
        let scene = SceneBuilder::new()
            .with_max_episode_steps(100)
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        assert_eq!(scene.robots.len(), 1);
        let bot = &scene.robots["test_bot"];
        assert_eq!(bot.joint_count(), 2);
        assert!(bot.joint_entity("joint1").is_some());
        assert!(bot.joint_entity("joint2").is_some());
    }

    #[test]
    fn build_with_initial_positions() {
        let mut positions = HashMap::new();
        positions.insert("joint1".into(), 0.5);

        let scene = SceneBuilder::new()
            .with_robot_urdf(TWO_JOINT_URDF, positions)
            .unwrap()
            .build();

        let bot = &scene.robots["test_bot"];
        let entity = bot.joint_entity("joint1").unwrap();
        let state = scene
            .app
            .world()
            .get::<clankers_actuator::components::JointState>(entity)
            .unwrap();
        assert!((state.position - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn build_with_robot_model() {
        let model = clankers_urdf::parse_string(TWO_JOINT_URDF).unwrap();
        let scene = SceneBuilder::new()
            .with_robot(model, HashMap::new())
            .build();

        assert_eq!(scene.robots.len(), 1);
        assert_eq!(scene.robots["test_bot"].joint_count(), 2);
    }

    #[test]
    fn built_scene_can_run_episode() {
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(5)
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        scene.app.world_mut().resource_mut::<Episode>().reset(None);

        for _ in 0..5 {
            scene.app.update();
        }

        let stats = scene.app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 1);
        assert_eq!(stats.total_steps, 5);
    }

    #[test]
    fn multiple_robots_in_scene() {
        let urdf2 = r#"
            <robot name="second_bot">
                <link name="base"/>
                <link name="arm"/>
                <joint name="arm_joint" type="revolute">
                    <parent link="base"/>
                    <child link="arm"/>
                    <axis xyz="0 0 1"/>
                    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
                </joint>
            </robot>
        "#;

        let scene = SceneBuilder::new()
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .with_robot_urdf(urdf2, HashMap::new())
            .unwrap()
            .build();

        assert_eq!(scene.robots.len(), 2);
        assert_eq!(scene.robots["test_bot"].joint_count(), 2);
        assert_eq!(scene.robots["second_bot"].joint_count(), 1);
    }

    #[test]
    fn invalid_urdf_returns_error() {
        let result = SceneBuilder::new().with_robot_urdf("<not-urdf/>", HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn default_builder_is_same_as_new() {
        let scene = SceneBuilder::default().build();
        assert!(scene.robots.is_empty());
    }
}
