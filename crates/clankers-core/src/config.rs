use std::collections::HashMap;
use std::path::PathBuf;

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

// ---------------------------------------------------------------------------
// Serde default functions
// ---------------------------------------------------------------------------

const fn default_physics_dt() -> f64 {
    0.001
}
const fn default_control_dt() -> f64 {
    0.02
}
const fn default_max_episode_steps() -> u32 {
    1000
}
const fn default_gravity() -> [f32; 3] {
    [0.0, 0.0, -9.81]
}
const fn default_render_resolution() -> [u32; 2] {
    [64, 64]
}
const fn default_orientation() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}
const fn default_true() -> bool {
    true
}
const fn default_color() -> [f32; 4] {
    [0.5, 0.5, 0.5, 1.0]
}
const fn default_mass() -> f32 {
    1.0
}
const fn default_friction() -> f32 {
    0.5
}
fn default_action_type() -> String {
    "velocity".into()
}
const fn default_action_limits() -> [f32; 2] {
    [-1.0, 1.0]
}

// ---------------------------------------------------------------------------
// SimConfig
// ---------------------------------------------------------------------------

/// Main simulation configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Resource)]
pub struct SimConfig {
    /// Physics timestep in seconds (default: 0.001 = 1000 Hz).
    #[serde(default = "default_physics_dt")]
    pub physics_dt: f64,

    /// Control timestep in seconds (default: 0.02 = 50 Hz).
    /// Must be >= `physics_dt`. The ratio `control_dt` / `physics_dt` gives substeps.
    #[serde(default = "default_control_dt")]
    pub control_dt: f64,

    /// Maximum steps per episode (default: 1000).
    #[serde(default = "default_max_episode_steps")]
    pub max_episode_steps: u32,

    /// Master random seed.
    #[serde(default)]
    pub seed: u64,

    /// Enable deterministic mode (slower but reproducible).
    #[serde(default)]
    pub deterministic_mode: bool,

    /// Gravity vector [x, y, z] in m/s^2.
    #[serde(default = "default_gravity")]
    pub gravity: [f32; 3],

    /// Render resolution [width, height] (default: [64, 64]).
    #[serde(default = "default_render_resolution")]
    pub render_resolution: [u32; 2],
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            physics_dt: default_physics_dt(),
            control_dt: default_control_dt(),
            max_episode_steps: default_max_episode_steps(),
            seed: 0,
            deterministic_mode: false,
            gravity: default_gravity(),
            render_resolution: default_render_resolution(),
        }
    }
}

impl SimConfig {
    /// Validate configuration. Returns Err on invalid values.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.physics_dt <= 0.0 {
            return Err(ConfigError::InvalidPhysicsDt(self.physics_dt));
        }
        if self.control_dt < self.physics_dt {
            return Err(ConfigError::ControlDtLessThanPhysicsDt);
        }
        Ok(())
    }

    /// Number of physics substeps per control step.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn substeps(&self) -> usize {
        (self.control_dt / self.physics_dt).round() as usize
    }

    /// Physics rate in Hz.
    pub fn physics_hz(&self) -> f64 {
        1.0 / self.physics_dt
    }

    /// Control rate in Hz.
    pub fn control_hz(&self) -> f64 {
        1.0 / self.control_dt
    }

    /// Load from TOML file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }
}

// ---------------------------------------------------------------------------
// JointLimitsConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLimitsConfig {
    pub per_joint: HashMap<String, [f32; 2]>,
    pub default: Option<[f32; 2]>,
}

// ---------------------------------------------------------------------------
// RobotConfig
// ---------------------------------------------------------------------------

/// Configuration for a robot instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotConfig {
    pub name: String,
    pub urdf_path: PathBuf,
    #[serde(default)]
    pub base_position: [f32; 3],
    #[serde(default = "default_orientation")]
    pub base_orientation: [f32; 4],
    #[serde(default = "default_true")]
    pub fixed_base: bool,
    #[serde(default)]
    pub initial_joint_positions: HashMap<String, f32>,
    #[serde(default)]
    pub position_limits: Option<JointLimitsConfig>,
    #[serde(default)]
    pub velocity_limits: Option<JointLimitsConfig>,
    #[serde(default)]
    pub effort_limits: Option<JointLimitsConfig>,
}

impl Default for RobotConfig {
    fn default() -> Self {
        Self {
            name: "robot".into(),
            urdf_path: PathBuf::new(),
            base_position: [0.0; 3],
            base_orientation: default_orientation(),
            fixed_base: true,
            initial_joint_positions: HashMap::default(),
            position_limits: None,
            velocity_limits: None,
            effort_limits: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Shape {
    Sphere(f32),
    Box([f32; 3]),
    Cylinder { radius: f32, half_height: f32 },
    Capsule { radius: f32, half_height: f32 },
    ConvexMesh(PathBuf),
    TriMesh(PathBuf),
}

// ---------------------------------------------------------------------------
// ObjectConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectConfig {
    pub name: String,
    pub shape: Shape,
    #[serde(default)]
    pub position: [f32; 3],
    #[serde(default = "default_orientation")]
    pub orientation: [f32; 4],
    #[serde(default = "default_color")]
    pub color: [f32; 4],
    #[serde(default = "default_true")]
    pub is_static: bool,
    #[serde(default = "default_mass")]
    pub mass: f32,
    #[serde(default = "default_friction")]
    pub friction: f32,
    #[serde(default)]
    pub restitution: f32,
    #[serde(default)]
    pub is_sensor: bool,
}

impl Default for ObjectConfig {
    fn default() -> Self {
        Self {
            name: "object".into(),
            shape: Shape::Box([0.1, 0.1, 0.1]),
            position: [0.0; 3],
            orientation: default_orientation(),
            color: default_color(),
            is_static: true,
            mass: default_mass(),
            friction: default_friction(),
            restitution: 0.0,
            is_sensor: false,
        }
    }
}

// ---------------------------------------------------------------------------
// SceneMeta
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SceneMeta {
    pub name: String,
    pub version: String,
    pub description: String,
}

// ---------------------------------------------------------------------------
// SensorConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConfig {
    pub sensor_type: String,
    pub robot: String,
    pub link: Option<String>,
    #[serde(default)]
    pub params: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// ObservationConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObservationConfig {
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
    #[serde(default)]
    pub normalize: bool,
}

// ---------------------------------------------------------------------------
// ActionConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionConfig {
    #[serde(default = "default_action_type")]
    pub action_type: String,
    #[serde(default)]
    pub joints: Vec<String>,
    #[serde(default = "default_action_limits")]
    pub limits: [f32; 2],
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            action_type: default_action_type(),
            joints: Vec::new(),
            limits: default_action_limits(),
        }
    }
}

// ---------------------------------------------------------------------------
// RewardConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RewardConfig {
    #[serde(default)]
    pub reward_type: String,
    #[serde(default)]
    pub weights: HashMap<String, f32>,
}

// ---------------------------------------------------------------------------
// TaskConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    pub env_type: String,
    pub robot: String,
    #[serde(default)]
    pub params: HashMap<String, String>,
    #[serde(default)]
    pub observation: ObservationConfig,
    #[serde(default)]
    pub action: ActionConfig,
    #[serde(default)]
    pub reward: RewardConfig,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            env_type: "custom".into(),
            robot: "robot".into(),
            params: HashMap::default(),
            observation: ObservationConfig::default(),
            action: ActionConfig::default(),
            reward: RewardConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// SceneConfig
// ---------------------------------------------------------------------------

/// Complete scene configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneConfig {
    #[serde(default)]
    pub meta: SceneMeta,
    #[serde(default)]
    pub simulation: SimConfig,
    #[serde(default)]
    pub robots: Vec<RobotConfig>,
    #[serde(default)]
    pub objects: Vec<ObjectConfig>,
    #[serde(default)]
    pub sensors: Vec<SensorConfig>,
    pub task: TaskConfig,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SimConfig defaults ----

    #[test]
    fn sim_config_default_values() {
        let cfg = SimConfig::default();
        assert!((cfg.physics_dt - 0.001).abs() < f64::EPSILON);
        assert!((cfg.control_dt - 0.02).abs() < f64::EPSILON);
        assert_eq!(cfg.max_episode_steps, 1000);
        assert_eq!(cfg.seed, 0);
        assert!(!cfg.deterministic_mode);
        assert!((cfg.gravity[0] - 0.0).abs() < f32::EPSILON);
        assert!((cfg.gravity[1] - 0.0).abs() < f32::EPSILON);
        assert!((cfg.gravity[2] - (-9.81)).abs() < f32::EPSILON);
        assert_eq!(cfg.render_resolution, [64, 64]);
    }

    // ---- SimConfig validate ----

    #[test]
    fn sim_config_validate_ok() {
        let cfg = SimConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn sim_config_validate_invalid_physics_dt_zero() {
        let cfg = SimConfig {
            physics_dt: 0.0,
            ..SimConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::InvalidPhysicsDt(_)));
    }

    #[test]
    fn sim_config_validate_invalid_physics_dt_negative() {
        let cfg = SimConfig {
            physics_dt: -0.001,
            ..SimConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::InvalidPhysicsDt(_)));
    }

    #[test]
    fn sim_config_validate_control_dt_less_than_physics_dt() {
        let cfg = SimConfig {
            physics_dt: 0.01,
            control_dt: 0.005,
            ..SimConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::ControlDtLessThanPhysicsDt));
    }

    #[test]
    fn sim_config_validate_control_dt_equal_physics_dt() {
        let cfg = SimConfig {
            physics_dt: 0.01,
            control_dt: 0.01,
            ..SimConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    // ---- SimConfig computed methods ----

    #[test]
    fn sim_config_substeps() {
        let cfg = SimConfig::default();
        // 0.02 / 0.001 = 20
        assert_eq!(cfg.substeps(), 20);
    }

    #[test]
    fn sim_config_substeps_custom() {
        let cfg = SimConfig {
            physics_dt: 0.005,
            control_dt: 0.02,
            ..SimConfig::default()
        };
        // 0.02 / 0.005 = 4
        assert_eq!(cfg.substeps(), 4);
    }

    #[test]
    fn sim_config_physics_hz() {
        let cfg = SimConfig::default();
        assert!((cfg.physics_hz() - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sim_config_control_hz() {
        let cfg = SimConfig::default();
        assert!((cfg.control_hz() - 50.0).abs() < f64::EPSILON);
    }

    // ---- SimConfig TOML deserialization ----

    #[test]
    fn sim_config_toml_deserialization() {
        let toml_str = r"
            physics_dt = 0.002
            control_dt = 0.04
            max_episode_steps = 500
            seed = 42
            deterministic_mode = true
            gravity = [0.0, 0.0, -10.0]
            render_resolution = [128, 128]
        ";
        let cfg: SimConfig = toml::from_str(toml_str).unwrap();
        assert!((cfg.physics_dt - 0.002).abs() < f64::EPSILON);
        assert!((cfg.control_dt - 0.04).abs() < f64::EPSILON);
        assert_eq!(cfg.max_episode_steps, 500);
        assert_eq!(cfg.seed, 42);
        assert!(cfg.deterministic_mode);
        assert!((cfg.gravity[2] - (-10.0)).abs() < f32::EPSILON);
        assert_eq!(cfg.render_resolution, [128, 128]);
    }

    #[test]
    fn sim_config_toml_defaults() {
        let toml_str = "";
        let cfg: SimConfig = toml::from_str(toml_str).unwrap();
        assert!((cfg.physics_dt - 0.001).abs() < f64::EPSILON);
        assert!((cfg.control_dt - 0.02).abs() < f64::EPSILON);
        assert_eq!(cfg.max_episode_steps, 1000);
        assert_eq!(cfg.seed, 0);
        assert!(!cfg.deterministic_mode);
    }

    // ---- SimConfig from_file ----

    #[test]
    fn sim_config_from_file() {
        let dir = std::env::temp_dir().join("clankers_test_sim_config");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_sim.toml");
        std::fs::write(
            &path,
            r"
            physics_dt = 0.005
            control_dt = 0.02
            max_episode_steps = 200
            seed = 7
        ",
        )
        .unwrap();

        let cfg = SimConfig::from_file(&path).unwrap();
        assert!((cfg.physics_dt - 0.005).abs() < f64::EPSILON);
        assert!((cfg.control_dt - 0.02).abs() < f64::EPSILON);
        assert_eq!(cfg.max_episode_steps, 200);
        assert_eq!(cfg.seed, 7);

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn sim_config_from_file_invalid() {
        let dir = std::env::temp_dir().join("clankers_test_sim_config_invalid");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_invalid.toml");
        std::fs::write(
            &path,
            r"
            physics_dt = -1.0
            control_dt = 0.02
        ",
        )
        .unwrap();

        let result = SimConfig::from_file(&path);
        assert!(result.is_err());

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn sim_config_from_file_not_found() {
        let result = SimConfig::from_file("/nonexistent/path/config.toml");
        assert!(result.is_err());
    }

    // ---- RobotConfig ----

    #[test]
    fn robot_config_default_values() {
        let cfg = RobotConfig::default();
        assert_eq!(cfg.name, "robot");
        assert_eq!(cfg.urdf_path, PathBuf::new());
        for v in &cfg.base_position {
            assert!(v.abs() < f32::EPSILON);
        }
        for (i, expected) in [0.0, 0.0, 0.0, 1.0].iter().enumerate() {
            assert!((cfg.base_orientation[i] - expected).abs() < f32::EPSILON);
        }
        assert!(cfg.fixed_base);
        assert!(cfg.initial_joint_positions.is_empty());
        assert!(cfg.position_limits.is_none());
        assert!(cfg.velocity_limits.is_none());
        assert!(cfg.effort_limits.is_none());
    }

    #[test]
    fn robot_config_toml_deserialization() {
        let toml_str = r#"
            name = "panda"
            urdf_path = "robots/panda.urdf"
            base_position = [0.0, 0.5, 0.0]
            base_orientation = [0.0, 0.0, 0.707, 0.707]
            fixed_base = true

            [initial_joint_positions]
            joint1 = 0.0
            joint2 = -0.785
        "#;
        let cfg: RobotConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.name, "panda");
        assert_eq!(cfg.urdf_path, PathBuf::from("robots/panda.urdf"));
        assert!((cfg.base_position[1] - 0.5).abs() < f32::EPSILON);
        assert!((cfg.base_orientation[2] - 0.707).abs() < f32::EPSILON);
        assert!(cfg.fixed_base);
        assert_eq!(cfg.initial_joint_positions.len(), 2);
        assert!((cfg.initial_joint_positions["joint2"] - (-0.785)).abs() < f32::EPSILON);
    }

    #[test]
    fn robot_config_toml_defaults_applied() {
        let toml_str = r#"
            name = "test_robot"
            urdf_path = "robot.urdf"
        "#;
        let cfg: RobotConfig = toml::from_str(toml_str).unwrap();
        for v in &cfg.base_position {
            assert!(v.abs() < f32::EPSILON);
        }
        for (i, expected) in [0.0, 0.0, 0.0, 1.0].iter().enumerate() {
            assert!((cfg.base_orientation[i] - expected).abs() < f32::EPSILON);
        }
        assert!(cfg.fixed_base);
        assert!(cfg.initial_joint_positions.is_empty());
    }

    // ---- ObjectConfig ----

    #[test]
    fn object_config_default_values() {
        let cfg = ObjectConfig::default();
        assert_eq!(cfg.name, "object");
        for v in &cfg.position {
            assert!(v.abs() < f32::EPSILON);
        }
        for (i, expected) in [0.0, 0.0, 0.0, 1.0].iter().enumerate() {
            assert!((cfg.orientation[i] - expected).abs() < f32::EPSILON);
        }
        for (i, expected) in [0.5, 0.5, 0.5, 1.0].iter().enumerate() {
            assert!((cfg.color[i] - expected).abs() < f32::EPSILON);
        }
        assert!(cfg.is_static);
        assert!((cfg.mass - 1.0).abs() < f32::EPSILON);
        assert!((cfg.friction - 0.5).abs() < f32::EPSILON);
        assert!((cfg.restitution - 0.0).abs() < f32::EPSILON);
        assert!(!cfg.is_sensor);
    }

    // ---- Shape serde round-trip ----

    #[test]
    fn shape_sphere_serde_roundtrip() {
        let shape = Shape::Sphere(0.5);
        let json = serde_json::to_string(&shape).unwrap();
        let shape2: Shape = serde_json::from_str(&json).unwrap();
        if let Shape::Sphere(r) = shape2 {
            assert!((r - 0.5).abs() < f32::EPSILON);
        } else {
            panic!("Expected Shape::Sphere");
        }
    }

    #[test]
    fn shape_box_serde_roundtrip() {
        let shape = Shape::Box([1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&shape).unwrap();
        let shape2: Shape = serde_json::from_str(&json).unwrap();
        if let Shape::Box(dims) = shape2 {
            for (i, expected) in [1.0, 2.0, 3.0].iter().enumerate() {
                assert!((dims[i] - expected).abs() < f32::EPSILON);
            }
        } else {
            panic!("Expected Shape::Box");
        }
    }

    #[test]
    fn shape_cylinder_serde_roundtrip() {
        let shape = Shape::Cylinder {
            radius: 0.1,
            half_height: 0.5,
        };
        let json = serde_json::to_string(&shape).unwrap();
        let shape2: Shape = serde_json::from_str(&json).unwrap();
        if let Shape::Cylinder {
            radius,
            half_height,
        } = shape2
        {
            assert!((radius - 0.1).abs() < f32::EPSILON);
            assert!((half_height - 0.5).abs() < f32::EPSILON);
        } else {
            panic!("Expected Shape::Cylinder");
        }
    }

    #[test]
    fn shape_capsule_serde_roundtrip() {
        let shape = Shape::Capsule {
            radius: 0.05,
            half_height: 0.3,
        };
        let json = serde_json::to_string(&shape).unwrap();
        let shape2: Shape = serde_json::from_str(&json).unwrap();
        if let Shape::Capsule {
            radius,
            half_height,
        } = shape2
        {
            assert!((radius - 0.05).abs() < f32::EPSILON);
            assert!((half_height - 0.3).abs() < f32::EPSILON);
        } else {
            panic!("Expected Shape::Capsule");
        }
    }

    #[test]
    fn shape_convex_mesh_serde_roundtrip() {
        let shape = Shape::ConvexMesh(PathBuf::from("meshes/convex.obj"));
        let json = serde_json::to_string(&shape).unwrap();
        let shape2: Shape = serde_json::from_str(&json).unwrap();
        if let Shape::ConvexMesh(p) = shape2 {
            assert_eq!(p, PathBuf::from("meshes/convex.obj"));
        } else {
            panic!("Expected Shape::ConvexMesh");
        }
    }

    #[test]
    fn shape_tri_mesh_serde_roundtrip() {
        let shape = Shape::TriMesh(PathBuf::from("meshes/terrain.obj"));
        let json = serde_json::to_string(&shape).unwrap();
        let shape2: Shape = serde_json::from_str(&json).unwrap();
        if let Shape::TriMesh(p) = shape2 {
            assert_eq!(p, PathBuf::from("meshes/terrain.obj"));
        } else {
            panic!("Expected Shape::TriMesh");
        }
    }

    // ---- JointLimitsConfig ----

    #[test]
    fn joint_limits_config_deserialization() {
        let toml_str = r"
            default = [-3.0, 3.0]

            [per_joint]
            joint1 = [-1.0, 1.0]
            joint2 = [-2.0, 2.0]
        ";
        let cfg: JointLimitsConfig = toml::from_str(toml_str).unwrap();
        let default = cfg.default.unwrap();
        assert!((default[0] - (-3.0)).abs() < f32::EPSILON);
        assert!((default[1] - 3.0).abs() < f32::EPSILON);
        assert_eq!(cfg.per_joint.len(), 2);
        assert!((cfg.per_joint["joint1"][0] - (-1.0)).abs() < f32::EPSILON);
        assert!((cfg.per_joint["joint1"][1] - 1.0).abs() < f32::EPSILON);
        assert!((cfg.per_joint["joint2"][0] - (-2.0)).abs() < f32::EPSILON);
        assert!((cfg.per_joint["joint2"][1] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn joint_limits_config_no_default() {
        let toml_str = r"
            [per_joint]
            joint1 = [-1.0, 1.0]
        ";
        let cfg: JointLimitsConfig = toml::from_str(toml_str).unwrap();
        assert!(cfg.default.is_none());
        assert_eq!(cfg.per_joint.len(), 1);
    }

    // ---- SceneConfig ----

    #[test]
    fn scene_config_full_toml_deserialization() {
        let toml_str = r#"
            [meta]
            name = "reach_task"
            version = "1.0"
            description = "A simple reaching task"

            [simulation]
            physics_dt = 0.002
            control_dt = 0.02
            max_episode_steps = 500
            seed = 42

            [[robots]]
            name = "panda"
            urdf_path = "robots/panda.urdf"
            base_position = [0.0, 0.0, 0.0]
            fixed_base = true

            [robots.initial_joint_positions]
            joint1 = 0.0
            joint2 = -0.785

            [[objects]]
            name = "target"
            position = [0.5, 0.0, 0.5]
            is_static = true
            mass = 0.1
            friction = 0.3

            [objects.shape]
            sphere = 0.05

            [[sensors]]
            sensor_type = "joint_state"
            robot = "panda"

            [task]
            env_type = "reach"
            robot = "panda"

            [task.observation]
            include = ["joint_positions", "joint_velocities", "target_position"]
            normalize = true

            [task.action]
            action_type = "velocity"
            joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
            limits = [-1.0, 1.0]

            [task.reward]
            reward_type = "distance"

            [task.reward.weights]
            distance = -1.0
            action_penalty = -0.01
        "#;
        let scene: SceneConfig = toml::from_str(toml_str).unwrap();

        // Meta
        assert_eq!(scene.meta.name, "reach_task");
        assert_eq!(scene.meta.version, "1.0");
        assert_eq!(scene.meta.description, "A simple reaching task");

        // Simulation
        assert!((scene.simulation.physics_dt - 0.002).abs() < f64::EPSILON);
        assert!((scene.simulation.control_dt - 0.02).abs() < f64::EPSILON);
        assert_eq!(scene.simulation.max_episode_steps, 500);
        assert_eq!(scene.simulation.seed, 42);

        // Robots
        assert_eq!(scene.robots.len(), 1);
        assert_eq!(scene.robots[0].name, "panda");
        assert_eq!(
            scene.robots[0].urdf_path,
            PathBuf::from("robots/panda.urdf")
        );
        assert!(scene.robots[0].fixed_base);
        assert_eq!(scene.robots[0].initial_joint_positions.len(), 2);

        // Objects
        assert_eq!(scene.objects.len(), 1);
        assert_eq!(scene.objects[0].name, "target");
        assert!((scene.objects[0].position[0] - 0.5).abs() < f32::EPSILON);
        assert!(scene.objects[0].is_static);
        if let Shape::Sphere(r) = scene.objects[0].shape {
            assert!((r - 0.05).abs() < f32::EPSILON);
        } else {
            panic!("Expected Shape::Sphere for target object");
        }

        // Sensors
        assert_eq!(scene.sensors.len(), 1);
        assert_eq!(scene.sensors[0].sensor_type, "joint_state");
        assert_eq!(scene.sensors[0].robot, "panda");

        // Task
        assert_eq!(scene.task.env_type, "reach");
        assert_eq!(scene.task.robot, "panda");
        assert_eq!(scene.task.observation.include.len(), 3);
        assert!(scene.task.observation.normalize);
        assert_eq!(scene.task.action.action_type, "velocity");
        assert_eq!(scene.task.action.joints.len(), 7);
        assert_eq!(scene.task.reward.reward_type, "distance");
        assert_eq!(scene.task.reward.weights.len(), 2);
        assert!((scene.task.reward.weights["distance"] - (-1.0)).abs() < f32::EPSILON);
        assert!((scene.task.reward.weights["action_penalty"] - (-0.01)).abs() < f32::EPSILON);
    }

    // ---- TaskConfig defaults ----

    #[test]
    fn task_config_default_values() {
        let cfg = TaskConfig::default();
        assert_eq!(cfg.env_type, "custom");
        assert_eq!(cfg.robot, "robot");
        assert!(cfg.params.is_empty());
        assert!(cfg.observation.include.is_empty());
        assert!(cfg.observation.exclude.is_empty());
        assert!(!cfg.observation.normalize);
        assert_eq!(cfg.action.action_type, "velocity");
        assert!(cfg.action.joints.is_empty());
        assert!((cfg.action.limits[0] - (-1.0)).abs() < f32::EPSILON);
        assert!((cfg.action.limits[1] - 1.0).abs() < f32::EPSILON);
        assert!(cfg.reward.reward_type.is_empty());
        assert!(cfg.reward.weights.is_empty());
    }

    // ---- ActionConfig defaults ----

    #[test]
    fn action_config_default_values() {
        let cfg = ActionConfig::default();
        assert_eq!(cfg.action_type, "velocity");
        assert!(cfg.joints.is_empty());
        assert!((cfg.limits[0] - (-1.0)).abs() < f32::EPSILON);
        assert!((cfg.limits[1] - 1.0).abs() < f32::EPSILON);
    }

    // ---- SceneMeta defaults ----

    #[test]
    fn scene_meta_default_values() {
        let meta = SceneMeta::default();
        assert!(meta.name.is_empty());
        assert!(meta.version.is_empty());
        assert!(meta.description.is_empty());
    }
}
