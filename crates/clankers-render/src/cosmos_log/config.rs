//! Configuration types for the Cosmos logging plugin.

use std::path::PathBuf;

use bevy::math::{Quat, Vec3};
use bevy::prelude::Resource;

/// Cosmos-native framerate (16 fps).
pub const COSMOS_FPS: u32 = 16;

/// Cosmos 480p resolution (fits in 24 GB VRAM).
pub const COSMOS_480P: (u32, u32) = (854, 480);

/// Configuration resource for [`CosmosLogPlugin`][super::CosmosLogPlugin].
///
/// Insert this resource before adding the plugin. Each [`CameraSpec`] in
/// `cameras` produces a subdirectory with RGB, depth, and segmentation PNGs.
///
/// # Example
///
/// ```
/// use clankers_render::cosmos_log::{CosmosLogConfig, CameraSpec, CameraPlacement};
/// use bevy::math::Vec3;
///
/// let config = CosmosLogConfig {
///     cameras: vec![CameraSpec {
///         label: "main".into(),
///         resolution: (854, 480),
///         fov_deg: 70.0,
///         placement: CameraPlacement::Fixed {
///             position: Vec3::new(0.0, 1.5, 2.0),
///             target: Vec3::ZERO,
///         },
///     }],
///     ..CosmosLogConfig::default()
/// };
/// ```
#[derive(Resource, Clone, Debug)]
pub struct CosmosLogConfig {
    /// Root output directory (default: `"data/"`).
    pub output_root: PathBuf,

    /// Optional run name override. When `None`, a timestamp-based name is
    /// generated automatically.
    pub run_name: Option<String>,

    /// Video framerate for metadata (default: 16, Cosmos native).
    pub fps: u32,

    /// Maximum depth in metres for normalization (default: 10.0).
    pub depth_max_m: f32,

    /// Segmentation classes that map to white (transform region) in the
    /// binary mask output. Other classes map to black (keep).
    ///
    /// Default: `["robot", "obstacle", "table"]`.
    pub seg_transform_classes: Vec<String>,

    /// Cameras to record. Each spec produces a subdirectory with
    /// `rgb_NNNNN.png`, `depth_NNNNN.png`, `seg_NNNNN.png`.
    pub cameras: Vec<CameraSpec>,
}

impl Default for CosmosLogConfig {
    fn default() -> Self {
        Self {
            output_root: PathBuf::from("data"),
            run_name: None,
            fps: COSMOS_FPS,
            depth_max_m: 10.0,
            seg_transform_classes: vec!["robot".into(), "obstacle".into(), "table".into()],
            cameras: Vec::new(),
        }
    }
}

impl CosmosLogConfig {
    /// Set the run name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.run_name = Some(name.into());
        self
    }
}

/// Specification for a single logging camera.
///
/// Each camera spec spawns three offscreen render targets (RGB, depth,
/// segmentation) at the specified resolution and placement.
#[derive(Clone, Debug)]
pub struct CameraSpec {
    /// Label used as the subdirectory name and buffer key.
    pub label: String,

    /// Render resolution `(width, height)`.
    pub resolution: (u32, u32),

    /// Vertical field of view in degrees.
    pub fov_deg: f32,

    /// Camera placement strategy.
    pub placement: CameraPlacement,
}

/// How a cosmos log camera is positioned in the scene.
#[derive(Clone, Debug)]
pub enum CameraPlacement {
    /// Fixed position looking at a target point.
    Fixed {
        /// Camera world position.
        position: Vec3,
        /// Look-at target.
        target: Vec3,
    },
    /// Follow a named link (e.g., end-effector) with an offset.
    FollowLink {
        /// Name of the link entity to track (matched against `LinkVisual`).
        link_name: String,
        /// Offset from the link origin in the link's local frame.
        offset: Vec3,
        /// Camera orientation relative to the link.
        orientation: Quat,
    },
}
