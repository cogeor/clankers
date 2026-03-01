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
use clankers_core::config::{ObjectConfig, Shape, SimConfig};
use clankers_core::types::RobotGroup;
use clankers_env::episode::EpisodeConfig;
use clankers_physics::rapier::bridge::register_robot;
use clankers_physics::rapier::{RapierBackend, RapierBackendFixed, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_urdf::spawner::SpawnedRobot;
use clankers_urdf::types::RobotModel;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyHandle};

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
    /// Physics rigid-body handles for objects added via [`SceneBuilder::with_object`],
    /// keyed by object name. Empty when physics is not enabled.
    pub object_bodies: HashMap<String, RigidBodyHandle>,
}

// ---------------------------------------------------------------------------
// RobotEntry
// ---------------------------------------------------------------------------

/// Internal representation of a robot to spawn.
struct RobotEntry {
    model: RobotModel,
    initial_positions: HashMap<String, f32>,
}

/// Physics backend configuration for auto-registration.
struct PhysicsConfig {
    /// When true, robot base links are fixed in place.
    fixed_base: bool,
    /// When true, physics runs on `FixedUpdate` instead of `Update`.
    use_fixed_update: bool,
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
    physics: Option<PhysicsConfig>,
    objects: Vec<ObjectConfig>,
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
            physics: None,
            objects: Vec::new(),
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

    /// Enable Rapier physics and auto-register all robots in [`build()`](Self::build).
    ///
    /// Physics systems run on the `Update` schedule. For visualization binaries
    /// that need `FixedUpdate`, use [`with_physics_fixed_update`](Self::with_physics_fixed_update).
    ///
    /// When `fixed_base` is true, each robot's root link is pinned in world
    /// space (typical for table-mounted arms).
    #[must_use]
    pub const fn with_physics(mut self, fixed_base: bool) -> Self {
        self.physics = Some(PhysicsConfig {
            fixed_base,
            use_fixed_update: false,
        });
        self
    }

    /// Enable Rapier physics on the `FixedUpdate` schedule.
    ///
    /// Same as [`with_physics`](Self::with_physics) but registers the step
    /// system on `FixedUpdate` so simulation and rendering are decoupled.
    #[must_use]
    pub const fn with_physics_fixed_update(mut self, fixed_base: bool) -> Self {
        self.physics = Some(PhysicsConfig {
            fixed_base,
            use_fixed_update: true,
        });
        self
    }

    /// Add a free-floating physics object to the scene.
    ///
    /// Objects are created during [`build()`](Self::build) as Rapier rigid
    /// bodies with colliders. Requires [`with_physics`](Self::with_physics)
    /// to be called first; objects are silently ignored if physics is disabled.
    ///
    /// The resulting [`SpawnedScene::object_bodies`] map contains the Rapier
    /// `RigidBodyHandle` for each object, keyed by name.
    #[must_use]
    pub fn with_object(mut self, config: ObjectConfig) -> Self {
        self.objects.push(config);
        self
    }

    /// Build the Bevy [`App`] with all plugins and spawned entities.
    ///
    /// Each robot is assigned a [`RobotId`](clankers_core::types::RobotId) via
    /// the [`RobotGroup`] resource, and all its joint entities are tagged with
    /// that ID for multi-robot queries.
    ///
    /// When physics is enabled via [`with_physics`](Self::with_physics),
    /// each robot is automatically registered with the Rapier backend and
    /// objects added via [`with_object`](Self::with_object) are materialised
    /// as rigid bodies with colliders.
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

        // Initialize RobotGroup resource.
        let mut robot_group = RobotGroup::default();

        // Spawn robots into the world with RobotId tagging.
        // We keep the model alongside the spawned data for physics registration.
        let mut robots = HashMap::new();
        let mut robot_models: Vec<(RobotModel, String)> = Vec::new();
        for entry in self.robots {
            let robot_id = robot_group.allocate(
                entry.model.name.clone(),
                Vec::new(), // joints filled after spawn
            );
            let spawned = clankers_urdf::spawn_robot_with_id(
                app.world_mut(),
                &entry.model,
                &entry.initial_positions,
                robot_id,
            );
            // Update RobotGroup with actual joint entities.
            let joint_entities: Vec<Entity> = spawned.joints.values().copied().collect();
            let info = robot_group.get_mut(robot_id).expect("just allocated");
            info.joints = joint_entities;
            robot_models.push((entry.model, spawned.name.clone()));
            robots.insert(spawned.name.clone(), spawned);
        }

        app.world_mut().insert_resource(robot_group);

        // --- Physics auto-registration ---
        let mut object_bodies = HashMap::new();
        if let Some(physics_config) = self.physics {
            // Add the physics plugin (creates RapierContext resource).
            if physics_config.use_fixed_update {
                app.add_plugins(ClankersPhysicsPlugin::new(RapierBackendFixed));
            } else {
                app.add_plugins(ClankersPhysicsPlugin::new(RapierBackend));
            }

            let world = app.world_mut();
            let mut ctx = world.remove_resource::<RapierContext>().unwrap();

            // Register each robot with the Rapier physics backend.
            for (model, name) in &robot_models {
                let spawned = &robots[name];
                register_robot(&mut ctx, model, spawned, world, physics_config.fixed_base);
            }

            // Create rigid bodies and colliders for scene objects.
            for obj in &self.objects {
                let body_builder = if obj.is_static {
                    RigidBodyBuilder::fixed()
                } else {
                    RigidBodyBuilder::dynamic().can_sleep(false)
                };
                let body = ctx.rigid_body_set.insert(
                    body_builder
                        .translation(Vec3::new(obj.position[0], obj.position[1], obj.position[2]))
                        .build(),
                );

                let collider = shape_to_collider(&obj.shape)
                    .density(obj.mass)
                    .friction(obj.friction)
                    .restitution(obj.restitution)
                    .sensor(obj.is_sensor)
                    .build();
                ctx.collider_set
                    .insert_with_parent(collider, body, &mut ctx.rigid_body_set);

                // Track named handle for callers.
                ctx.body_handles.insert(obj.name.clone(), body);
                object_bodies.insert(obj.name.clone(), body);
            }

            // Re-snapshot so reset covers robots AND objects.
            ctx.snapshot_initial_state();

            world.insert_resource(ctx);
        }

        SpawnedScene {
            app,
            robots,
            object_bodies,
        }
    }
}

// ---------------------------------------------------------------------------
// Shape -> Rapier collider
// ---------------------------------------------------------------------------

/// Convert a [`Shape`] config into a Rapier [`ColliderBuilder`].
fn shape_to_collider(shape: &Shape) -> ColliderBuilder {
    match shape {
        Shape::Sphere(radius) => ColliderBuilder::ball(*radius),
        Shape::Box(half_extents) => {
            ColliderBuilder::cuboid(half_extents[0], half_extents[1], half_extents[2])
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => ColliderBuilder::cylinder(*half_height, *radius),
        Shape::Capsule {
            radius,
            half_height,
        } => ColliderBuilder::capsule_y(*half_height, *radius),
        // Mesh shapes are not yet supported; fall back to a small sphere
        // so the build does not panic.
        Shape::ConvexMesh(_) | Shape::TriMesh(_) => {
            eprintln!("warning: mesh shapes are not yet supported in with_object(); falling back to unit sphere collider");
            ColliderBuilder::ball(0.01)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EpisodeStats;
    use clankers_core::types::RobotId;
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

    #[test]
    fn build_populates_robot_group() {
        let scene = SceneBuilder::new()
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        let group = scene.app.world().resource::<RobotGroup>();
        assert_eq!(group.len(), 1);
        let info = group.get(RobotId(0)).unwrap();
        assert_eq!(info.name(), "test_bot");
        assert_eq!(info.joint_count(), 2);
    }

    #[test]
    fn multi_robot_scene_assigns_distinct_ids() {
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

        let group = scene.app.world().resource::<RobotGroup>();
        assert_eq!(group.len(), 2);

        // Verify entities are tagged with RobotId
        let bot = &scene.robots["test_bot"];
        let entity = bot.joint_entity("joint1").unwrap();
        let id = scene.app.world().get::<RobotId>(entity).unwrap();
        assert_eq!(id.index(), 0);

        let bot2 = &scene.robots["second_bot"];
        let entity2 = bot2.joint_entity("arm_joint").unwrap();
        let id2 = scene.app.world().get::<RobotId>(entity2).unwrap();
        assert_eq!(id2.index(), 1);
    }

    #[test]
    fn empty_scene_has_empty_robot_group() {
        let scene = SceneBuilder::new().build();
        let group = scene.app.world().resource::<RobotGroup>();
        assert!(group.is_empty());
    }
}
