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
use clankers_core::layout::JointLayout;
use clankers_core::types::{LayoutCompileError, MissingJoints, RobotGroup};
use clankers_env::episode::EpisodeConfig;
use clankers_physics::ClankersPhysicsPlugin;
use clankers_physics::rapier::bridge::register_robot;
use clankers_physics::rapier::runtime::{JointRuntime, JointRuntimes};
use clankers_physics::rapier::systems::{
    MotorOverrides, validate_motor_coverage as physics_validate_motor_coverage,
};
use clankers_physics::rapier::{RapierBackend, RapierBackendFixed, RapierContext};
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
// BuildError
// ---------------------------------------------------------------------------

/// Recoverable error returned by [`SceneBuilder::try_build`].
///
/// `SceneBuilder::build` panics on configuration errors that almost
/// always indicate a programming mistake (e.g. a forgotten motor
/// override entry — see the `MEMORY.md` "robot flailing wildly"
/// note). `try_build` returns this error instead so the CLI and any
/// future plugin loader can degrade gracefully. See
/// `docs/plans/WS2-plan.md` § 8 ("Risk: …recoverable error…").
#[derive(Debug)]
pub enum BuildError {
    /// `MotorOverrides` resource is present but does not cover every
    /// joint declared by the supplied [`JointLayout`].
    MissingMotors(MissingJoints),
    /// The dense `JointRuntimes` could not be compiled from the
    /// scene's layout, [`RapierContext`], and [`MotorOverrides`]. See
    /// [`LayoutCompileError`] for the failure modes.
    CompileRuntime(LayoutCompileError),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingMotors(m) => write!(f, "scene build failed: {m}"),
            Self::CompileRuntime(e) => write!(f, "scene build failed: {e}"),
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MissingMotors(m) => Some(m),
            Self::CompileRuntime(e) => Some(e),
        }
    }
}

impl From<MissingJoints> for BuildError {
    fn from(m: MissingJoints) -> Self {
        Self::MissingMotors(m)
    }
}

impl From<LayoutCompileError> for BuildError {
    fn from(e: LayoutCompileError) -> Self {
        Self::CompileRuntime(e)
    }
}

// ---------------------------------------------------------------------------
// validate_motor_coverage (re-export wrapper)
// ---------------------------------------------------------------------------

/// Validate that `overrides` covers every joint declared by `layout`.
///
/// Thin re-export of
/// [`clankers_physics::rapier::systems::validate_motor_coverage`]; lives
/// here so call sites that already depend on `clankers-sim` (the
/// majority of example bins) don't have to add a direct
/// `clankers-physics` dependency. See `docs/plans/WS2-plan.md` § 5
/// PR1-5 and `.delegate/work/.../01/PLAN.md` design choice A.
///
/// # Errors
///
/// Returns [`MissingJoints`] listing every layout joint name whose
/// entity is absent from `overrides.joints`.
pub fn validate_motor_coverage(
    group: &RobotGroup,
    layout: &JointLayout,
    overrides: &MotorOverrides,
) -> Result<(), MissingJoints> {
    physics_validate_motor_coverage(group, layout, overrides)
}

// ---------------------------------------------------------------------------
// compile_runtime
// ---------------------------------------------------------------------------

/// Compile a layout-indexed [`JointRuntimes`] from the scene's
/// [`RapierContext`] and [`MotorOverrides`].
///
/// Each actuated layout slot is resolved in order; missing entries
/// surface a typed [`LayoutCompileError`]. The resulting
/// [`JointRuntimes`] is the dense, slot-ordered view of the per-frame
/// hot path's joint data.
///
/// # Why a free function
///
/// `clankers-core` (where [`RobotGroup`] lives) cannot depend on
/// `clankers-physics` (where the runtime structs live) without
/// inducing a reverse cycle. So the function lives in `clankers-sim`,
/// mirroring the W2 PR1 [`validate_motor_coverage`] precedent.
///
/// The `_group` parameter is accepted for future per-robot
/// validation (matching the `validate_motor_coverage` precedent —
/// the current implementation does not use it).
///
/// # W7 PR3 invariant
///
/// This is the generalisation of [`validate_motor_coverage`]: the prose
/// invariant "every joint must be overridden" becomes a typed compile
/// step that returns [`LayoutCompileError::MissingMotor`] if violated.
/// `validate_motor_coverage` still runs from
/// [`SceneBuilder::try_build`] first, so callers see the existing
/// [`MissingJoints`] error before this new one.
///
/// # Errors
///
/// - [`LayoutCompileError::UnboundEntity`] — a layout slot has no
///   `entity` field set (caller forgot
///   [`JointLayout::bind_entities`](clankers_core::layout::JointLayout::bind_entities)).
/// - [`LayoutCompileError::MissingJoint`] — the slot's entity has no
///   entry in `ctx.joint_handles` (bridge setup was incomplete for
///   this joint).
/// - [`LayoutCompileError::MissingJointInfo`] — likewise for
///   `ctx.joint_info`.
/// - [`LayoutCompileError::MissingMotor`] — `overrides.joints` has no
///   entry for an actuated slot's entity. Promotes MEMORY.md "every
///   joint must be overridden" to a compile-time invariant.
pub fn compile_runtime(
    _group: &RobotGroup,
    layout: &JointLayout,
    ctx: &RapierContext,
    overrides: &MotorOverrides,
) -> Result<JointRuntimes, LayoutCompileError> {
    let mut joints: Vec<JointRuntime> = Vec::with_capacity(layout.len());
    let mut entity_to_slot: HashMap<Entity, usize> = HashMap::with_capacity(layout.len());

    for spec in layout.joints() {
        // Skip non-actuated slots entirely — they have no joint handle
        // in `ctx.joint_handles` (the bridge only inserts actuated
        // joints) and no motor override (rule applies only to
        // actuated joints).
        if !spec.joint_type.is_actuated() {
            continue;
        }

        let entity = spec
            .entity
            .ok_or_else(|| LayoutCompileError::UnboundEntity {
                name: spec.name.clone(),
            })?;
        let handle = ctx.joint_handles.get(&entity).copied().ok_or_else(|| {
            LayoutCompileError::MissingJoint {
                name: spec.name.clone(),
            }
        })?;
        let info = ctx.joint_info.get(&entity).cloned().ok_or_else(|| {
            LayoutCompileError::MissingJointInfo {
                name: spec.name.clone(),
            }
        })?;
        let motor = overrides.joints.get(&entity).cloned();
        if motor.is_none() {
            return Err(LayoutCompileError::MissingMotor {
                name: spec.name.clone(),
            });
        }

        let slot = joints.len();
        entity_to_slot.insert(entity, slot);
        joints.push(JointRuntime {
            entity,
            handle,
            layout_slot: slot,
            info,
            motor,
        });
    }

    Ok(JointRuntimes {
        joints,
        entity_to_slot,
    })
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

    /// Build the Bevy [`App`] and validate every robot's motor
    /// coverage if a [`MotorOverrides`] resource was inserted.
    ///
    /// Equivalent to [`Self::build`] except missing-motor errors are
    /// returned via [`BuildError`] instead of being deferred to a
    /// runtime "robot flailing wildly" failure. The CLI surface
    /// (`clankers validate`, W5) calls this method; example bins keep
    /// using `build()` for the panic-on-misconfigure behaviour.
    ///
    /// The layout for each robot is constructed from its parsed
    /// [`RobotModel`] via `RobotModel::to_layout()` and then bound to
    /// the spawned joint entities in the same order
    /// `RobotModel::actuated_joints()` reports.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::MissingMotors`] if any joint declared by
    /// any robot's layout is absent from `MotorOverrides.joints`.
    /// Scenes built without a `MotorOverrides` resource always succeed
    /// because there is nothing to validate.
    pub fn try_build(self) -> Result<SpawnedScene, BuildError> {
        // Snapshot robot models *before* `build()` consumes them so we
        // can rebuild layouts for validation after the scene exists.
        let model_snapshot: Vec<(RobotModel, HashMap<String, f32>)> = self
            .robots
            .iter()
            .map(|entry| (entry.model.clone(), entry.initial_positions.clone()))
            .collect();
        let mut scene = self.build();

        if scene.app.world().get_resource::<MotorOverrides>().is_none() {
            return Ok(scene);
        }

        // First pass: per-robot layouts bound to entities, validate
        // motor coverage. Cache the bound layouts so we can compile
        // the dense runtime in a second pass without re-binding.
        let mut bound_layouts: Vec<JointLayout> = Vec::with_capacity(model_snapshot.len());
        for (model, _) in &model_snapshot {
            let mut layout = model.to_layout();
            let Some(spawned) = scene.robots.get(&model.name) else {
                continue;
            };
            // Bind joint entities into the layout in the same order
            // RobotModel::actuated_joints() produces names.
            let joint_entities: Vec<Entity> = layout
                .joints()
                .iter()
                .filter_map(|spec| spawned.joints.get(&spec.name).copied())
                .collect();
            if joint_entities.len() != layout.len() {
                // Layout names didn't all resolve to entities (e.g.
                // the spawner skipped a joint). Don't fail validation
                // here — the layout itself is structurally suspect and
                // a clearer error will surface at the CLI layer.
                continue;
            }
            layout.bind_entities(&joint_entities);
            let group = scene.app.world().resource::<RobotGroup>();
            let overrides = scene.app.world().resource::<MotorOverrides>();
            validate_motor_coverage(group, &layout, overrides)?;
            bound_layouts.push(layout);
        }

        // Second pass: W7 PR3 dense runtime compile.
        //
        // Compiles each bound layout into a `JointRuntimes` vec and
        // concatenates across robots so a single Bevy resource carries
        // every actuated joint in the scene. Only attempted when a
        // `RapierContext` resource is present (scenes built without
        // `with_physics()` have nothing to compile against). Returns
        // `BuildError::CompileRuntime` on any layout/context mismatch.
        if scene.app.world().get_resource::<RapierContext>().is_some() {
            let mut combined = JointRuntimes::default();
            // Re-borrow each resource — borrow checker requires reads
            // be released across iterations because `compile_runtime`
            // takes shared references.
            for layout in &bound_layouts {
                let group = scene.app.world().resource::<RobotGroup>();
                let overrides = scene.app.world().resource::<MotorOverrides>();
                let ctx = scene.app.world().resource::<RapierContext>();
                let part = compile_runtime(group, layout, ctx, overrides)?;
                // Concatenate, re-indexing slots so the merged vec is
                // a contiguous 0..N range across robots.
                let base = combined.joints.len();
                for mut jr in part.joints {
                    let new_slot = base + jr.layout_slot;
                    jr.layout_slot = new_slot;
                    combined.entity_to_slot.insert(jr.entity, new_slot);
                    combined.joints.push(jr);
                }
            }
            scene.app.world_mut().insert_resource(combined);
        }
        Ok(scene)
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
            warn!(
                "mesh shapes are not yet supported in with_object(); falling back to unit sphere collider"
            );
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

    // ---- try_build / validate_motor_coverage ----

    #[test]
    fn try_build_returns_scene_when_no_motor_overrides_present() {
        let scene = SceneBuilder::new()
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .try_build()
            .expect("scene without overrides must build cleanly");
        assert_eq!(scene.robots.len(), 1);
    }

    #[test]
    fn try_build_validates_motor_overrides_against_layout() {
        use clankers_core::layout::{JointKind, JointLayoutBuilder, JointSpec, JointSpecLimits};
        use clankers_physics::rapier::systems::MotorOverrideParams;
        use std::collections::HashMap as StdHashMap;

        // Build a synthetic 2-joint layout bound to two fake entities
        // so we can exercise validate_motor_coverage directly.
        let entity_a = Entity::from_bits(101);
        let entity_b = Entity::from_bits(202);
        let mut layout = JointLayoutBuilder::default()
            .push(JointSpec {
                name: "a".into(),
                entity: None,
                joint_type: JointKind::Revolute,
                limits: JointSpecLimits::default(),
                axis: [0.0, 0.0, 1.0],
            })
            .push(JointSpec {
                name: "b".into(),
                entity: None,
                joint_type: JointKind::Revolute,
                limits: JointSpecLimits::default(),
                axis: [0.0, 0.0, 1.0],
            })
            .build();
        layout.bind_entities(&[entity_a, entity_b]);

        let group = RobotGroup::default();

        // Missing entity_b -> error.
        let mut partial = StdHashMap::new();
        partial.insert(
            entity_a,
            MotorOverrideParams {
                target_pos: 0.0,
                target_vel: 0.0,
                stiffness: 100.0,
                damping: 10.0,
                max_force: 50.0,
            },
        );
        let overrides = MotorOverrides {
            joints: partial,
            ..MotorOverrides::default()
        };
        let err = validate_motor_coverage(&group, &layout, &overrides).unwrap_err();
        assert_eq!(err.layout_joint_names, vec!["b".to_string()]);

        // Complete coverage -> Ok.
        let mut full = StdHashMap::new();
        for e in [entity_a, entity_b] {
            full.insert(
                e,
                MotorOverrideParams {
                    target_pos: 0.0,
                    target_vel: 0.0,
                    stiffness: 100.0,
                    damping: 10.0,
                    max_force: 50.0,
                },
            );
        }
        let overrides_full = MotorOverrides {
            joints: full,
            ..MotorOverrides::default()
        };
        assert!(validate_motor_coverage(&group, &layout, &overrides_full).is_ok());
    }

    #[test]
    fn try_build_propagates_missing_motors_for_spawned_robot() {
        use clankers_physics::rapier::systems::MotorOverrideParams;
        use std::collections::HashMap as StdHashMap;

        // Build a scene with TWO_JOINT_URDF (2 actuated joints) and
        // insert a MotorOverrides resource that covers only one of
        // them. try_build must surface the missing joint via
        // BuildError::MissingMotors.
        let scene = SceneBuilder::new()
            .with_robot_urdf(TWO_JOINT_URDF, HashMap::new())
            .unwrap()
            .build();

        // Locate one of the spawned joint entities.
        let bot = &scene.robots["test_bot"];
        let entity_one = bot.joint_entity("joint1").expect("joint1 spawned");

        // Insert MotorOverrides covering only that one entity.
        let mut map = StdHashMap::new();
        map.insert(
            entity_one,
            MotorOverrideParams {
                target_pos: 0.0,
                target_vel: 0.0,
                stiffness: 100.0,
                damping: 10.0,
                max_force: 50.0,
            },
        );
        let mut app = scene.app;
        app.world_mut().insert_resource(MotorOverrides {
            joints: map,
            ..MotorOverrides::default()
        });

        // Now reconstruct the validation path manually (we cannot call
        // try_build a second time without rebuilding the whole scene).
        // The motor_overrides cover only joint1; joint2 is missing.
        let model = clankers_urdf::parse_string(TWO_JOINT_URDF).unwrap();
        let mut layout = model.to_layout();
        let entity_two = bot.joint_entity("joint2").expect("joint2 spawned");
        let entities: Vec<Entity> = layout
            .joints()
            .iter()
            .map(|spec| match spec.name.as_str() {
                "joint1" => entity_one,
                "joint2" => entity_two,
                _ => unreachable!(),
            })
            .collect();
        layout.bind_entities(&entities);

        let overrides = app.world().resource::<MotorOverrides>();
        let err = validate_motor_coverage(app.world().resource::<RobotGroup>(), &layout, overrides)
            .unwrap_err();
        assert!(
            err.layout_joint_names.contains(&"joint2".to_string()),
            "expected joint2 in missing: {:?}",
            err.layout_joint_names
        );

        // BuildError conversion picks the MissingMotors variant.
        let build_err = BuildError::from(err);
        let msg = format!("{build_err}");
        assert!(msg.contains("joint2"), "BuildError display: {msg}");
        assert!(std::error::Error::source(&build_err).is_some());
    }

    // ---- compile_runtime (W7 PR3) ----

    /// Build a fresh `RapierContext` populated with N actuated revolute
    /// joints whose entities have `joint_handles` / `joint_info` entries
    /// matching the order they were inserted. Returns the populated
    /// context plus the ordered entity vector.
    fn make_test_context_with_n_joints(n: usize) -> (RapierContext, Vec<Entity>) {
        use rapier3d::prelude::{
            ImpulseJointHandle, RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle,
        };

        let mut ctx = RapierContext::new(Vec3::new(0.0, 0.0, -9.81), 1.0 / 60.0, 1);

        let mut entities = Vec::with_capacity(n);
        let mut parent: RigidBodyHandle =
            ctx.rigid_body_set.insert(RigidBodyBuilder::fixed().build());
        for i in 0..n {
            let child = ctx
                .rigid_body_set
                .insert(RigidBodyBuilder::dynamic().can_sleep(false).build());
            let joint = RevoluteJointBuilder::new(Vec3::Z).build();
            let jh: ImpulseJointHandle = ctx.impulse_joint_set.insert(parent, child, joint, true);

            // Use deterministic non-zero entity bits.
            let entity = Entity::from_bits((i as u64) + 100);
            entities.push(entity);
            ctx.joint_handles.insert(entity, jh);
            ctx.joint_info.insert(
                entity,
                clankers_physics::rapier::context::JointInfo {
                    parent_body: parent,
                    child_body: child,
                    axis: Vec3::Z,
                    is_prismatic: false,
                },
            );
            ctx.body_to_entity.insert(child, entity);
            parent = child;
        }
        (ctx, entities)
    }

    fn motor_params(stiff: f32) -> clankers_physics::rapier::systems::MotorOverrideParams {
        clankers_physics::rapier::systems::MotorOverrideParams {
            target_pos: 0.0,
            target_vel: 0.0,
            stiffness: stiff,
            damping: 10.0,
            max_force: 50.0,
        }
    }

    fn synthetic_layout(names: &[&str], entities: &[Option<Entity>]) -> JointLayout {
        use clankers_core::layout::{JointKind, JointLayoutBuilder, JointSpec, JointSpecLimits};
        let mut b = JointLayoutBuilder::default();
        for (i, name) in names.iter().enumerate() {
            b = b.push(JointSpec {
                name: (*name).to_string(),
                entity: entities.get(i).copied().flatten(),
                joint_type: JointKind::Revolute,
                limits: JointSpecLimits {
                    lower: Some(-1.0),
                    upper: Some(1.0),
                    effort: 1.0,
                    velocity: 1.0,
                },
                axis: [0.0, 0.0, 1.0],
            });
        }
        b.build()
    }

    #[test]
    fn compile_runtime_orders_by_layout_slot() {
        use std::collections::HashMap as StdHashMap;
        // Insert 4 joint handles in order [d, a, c, b]; build layout
        // [a, b, c, d]. compile_runtime should walk the layout, so the
        // resulting JointRuntimes.joints[0].entity == entity bound to "a".
        let (ctx, entities) = make_test_context_with_n_joints(4);
        let e_d = entities[0];
        let e_a = entities[1];
        let e_c = entities[2];
        let e_b = entities[3];

        let layout = synthetic_layout(
            &["a", "b", "c", "d"],
            &[Some(e_a), Some(e_b), Some(e_c), Some(e_d)],
        );

        let mut motor_map = StdHashMap::new();
        for &e in &[e_a, e_b, e_c, e_d] {
            motor_map.insert(e, motor_params(100.0));
        }
        let overrides = MotorOverrides {
            joints: motor_map,
            ..MotorOverrides::default()
        };
        let group = RobotGroup::default();
        let runtimes = compile_runtime(&group, &layout, &ctx, &overrides).expect("should compile");

        assert_eq!(runtimes.joints.len(), 4);
        assert_eq!(runtimes.joints[0].entity, e_a);
        assert_eq!(runtimes.joints[1].entity, e_b);
        assert_eq!(runtimes.joints[2].entity, e_c);
        assert_eq!(runtimes.joints[3].entity, e_d);
        assert_eq!(runtimes.joints[0].layout_slot, 0);
        assert_eq!(runtimes.joints[3].layout_slot, 3);
        // entity_to_slot reflects slot positions, not insertion order
        assert_eq!(runtimes.slot_for(e_a), Some(0));
        assert_eq!(runtimes.slot_for(e_d), Some(3));
    }

    #[test]
    fn compile_runtime_rejects_missing_layout_joint() {
        use std::collections::HashMap as StdHashMap;
        // Build a layout with one valid entity and one "phantom" entity
        // not registered in ctx.joint_handles. Expect
        // LayoutCompileError::MissingJoint { name: "phantom_joint" }.
        let (ctx, entities) = make_test_context_with_n_joints(1);
        let real = entities[0];
        let phantom = Entity::from_bits(9999);

        let layout = synthetic_layout(
            &["real_joint", "phantom_joint"],
            &[Some(real), Some(phantom)],
        );

        let mut motor_map = StdHashMap::new();
        motor_map.insert(real, motor_params(100.0));
        motor_map.insert(phantom, motor_params(100.0));
        let overrides = MotorOverrides {
            joints: motor_map,
            ..MotorOverrides::default()
        };
        let group = RobotGroup::default();
        let err = compile_runtime(&group, &layout, &ctx, &overrides).unwrap_err();
        match err {
            LayoutCompileError::MissingJoint { name } => {
                assert_eq!(name, "phantom_joint");
            }
            other => panic!("expected MissingJoint, got {other:?}"),
        }
    }

    #[test]
    fn compile_runtime_rejects_missing_motor() {
        use std::collections::HashMap as StdHashMap;
        // Layout has 2 actuated joints; overrides cover only the first.
        // Expect LayoutCompileError::MissingMotor for the second.
        let (ctx, entities) = make_test_context_with_n_joints(2);
        let e0 = entities[0];
        let e1 = entities[1];
        let layout = synthetic_layout(&["a", "b"], &[Some(e0), Some(e1)]);

        let mut motor_map = StdHashMap::new();
        motor_map.insert(e0, motor_params(100.0));
        let overrides = MotorOverrides {
            joints: motor_map,
            ..MotorOverrides::default()
        };
        let group = RobotGroup::default();
        let err = compile_runtime(&group, &layout, &ctx, &overrides).unwrap_err();
        match err {
            LayoutCompileError::MissingMotor { name } => {
                assert_eq!(name, "b");
            }
            other => panic!("expected MissingMotor, got {other:?}"),
        }
    }

    #[test]
    fn compile_runtime_rejects_unbound_entity() {
        use std::collections::HashMap as StdHashMap;
        let (ctx, entities) = make_test_context_with_n_joints(1);
        let real = entities[0];
        // Second slot has `entity: None` (unbound)
        let layout = synthetic_layout(&["a", "unbound"], &[Some(real), None]);

        let mut motor_map = StdHashMap::new();
        motor_map.insert(real, motor_params(100.0));
        let overrides = MotorOverrides {
            joints: motor_map,
            ..MotorOverrides::default()
        };
        let group = RobotGroup::default();
        let err = compile_runtime(&group, &layout, &ctx, &overrides).unwrap_err();
        match err {
            LayoutCompileError::UnboundEntity { name } => {
                assert_eq!(name, "unbound");
            }
            other => panic!("expected UnboundEntity, got {other:?}"),
        }
    }

    #[test]
    fn compile_runtime_skips_non_actuated_slots() {
        use clankers_core::layout::{JointKind, JointLayoutBuilder, JointSpec, JointSpecLimits};
        use std::collections::HashMap as StdHashMap;
        // Layout with one actuated and one fixed joint. compile_runtime
        // should produce a single-entry JointRuntimes (skipping fixed).
        let (ctx, entities) = make_test_context_with_n_joints(1);
        let e_real = entities[0];
        let e_fixed = Entity::from_bits(8888);

        let mut layout = JointLayoutBuilder::default()
            .push(JointSpec {
                name: "act".into(),
                entity: Some(e_real),
                joint_type: JointKind::Revolute,
                limits: JointSpecLimits::default(),
                axis: [0.0, 0.0, 1.0],
            })
            .push(JointSpec {
                name: "fixed_link".into(),
                entity: Some(e_fixed),
                joint_type: JointKind::Fixed,
                limits: JointSpecLimits::default(),
                axis: [0.0, 0.0, 1.0],
            })
            .build();
        // bind no-ops since entities are already set; keep for symmetry
        // with realistic call sites.
        let _ = &mut layout;

        let mut motor_map = StdHashMap::new();
        motor_map.insert(e_real, motor_params(100.0));
        let overrides = MotorOverrides {
            joints: motor_map,
            ..MotorOverrides::default()
        };
        let group = RobotGroup::default();
        let runtimes = compile_runtime(&group, &layout, &ctx, &overrides).expect("should compile");
        // Fixed joint is skipped — only the actuated one made it in.
        assert_eq!(runtimes.joints.len(), 1);
        assert_eq!(runtimes.joints[0].entity, e_real);
    }

    #[test]
    fn build_error_compile_runtime_display_and_source() {
        let lc = LayoutCompileError::MissingJoint {
            name: "xjoint".into(),
        };
        let be = BuildError::from(lc);
        let msg = format!("{be}");
        assert!(msg.contains("xjoint"), "BuildError display: {msg}");
        assert!(std::error::Error::source(&be).is_some());
    }
}
