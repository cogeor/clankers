//! Cosmos-Transfer2.5 logging plugin for uniform multi-camera frame export.
//!
//! [`CosmosLogPlugin`] spawns offscreen camera triplets (RGB + depth +
//! segmentation) per [`CameraSpec`] and writes PNG frames to a timestamped
//! `data/` directory every frame. Designed for replay binaries where exact
//! framerate is controlled by the app.
//!
//! # Usage
//!
//! ```no_run
//! use bevy::prelude::*;
//! use bevy::math::Vec3;
//! use clankers_render::cosmos_log::*;
//!
//! App::new()
//!     .add_plugins(MinimalPlugins)
//!     .insert_resource(CosmosLogConfig {
//!         cameras: vec![CameraSpec {
//!             label: "main".into(),
//!             resolution: (854, 480),
//!             fov_deg: 70.0,
//!             placement: CameraPlacement::Fixed {
//!                 position: Vec3::new(0.0, 1.5, 2.0),
//!                 target: Vec3::ZERO,
//!             },
//!         }],
//!         ..CosmosLogConfig::default()
//!     })
//!     .add_plugins(CosmosLogPlugin)
//!     .run();
//! ```

pub mod cameras;
pub mod config;
pub mod metadata;
pub mod writer;

pub use cameras::{CosmosLogCamera, CosmosModality};
pub use config::{CameraPlacement, CameraSpec, CosmosLogConfig, COSMOS_480P, COSMOS_FPS};

use bevy::prelude::*;

use crate::ClankersRenderPlugin;
use crate::depth_material::{DepthMaterial, DepthMaterialHandle, DepthMaterialPlugin};
use crate::readback::ImageCopyPlugin;
use crate::segmentation::{ClankersSegmentationPlugin, SegmentationFrameBuffers};

/// Bevy plugin for Cosmos-ready multi-camera frame logging.
///
/// Requires [`CosmosLogConfig`] to be inserted as a resource before the
/// plugin is added. Automatically adds dependency plugins if not already
/// present.
pub struct CosmosLogPlugin;

impl Plugin for CosmosLogPlugin {
    fn build(&self, app: &mut App) {
        // Auto-add dependency plugins.
        if !app.is_plugin_added::<ClankersRenderPlugin>() {
            app.add_plugins(ClankersRenderPlugin);
        }
        if !app.is_plugin_added::<ImageCopyPlugin>() {
            app.add_plugins(ImageCopyPlugin);
        }
        if !app.is_plugin_added::<DepthMaterialPlugin>() {
            app.add_plugins(DepthMaterialPlugin);
        }
        if !app.is_plugin_added::<ClankersSegmentationPlugin>() {
            app.add_plugins(ClankersSegmentationPlugin);
        }

        // Initialize keyed buffer resources.
        app.init_resource::<SegmentationFrameBuffers>();

        // Create and insert the shared depth material.
        let config = app.world().resource::<CosmosLogConfig>();
        let max_depth = config.depth_max_m;
        let depth_mat = app
            .world_mut()
            .resource_mut::<Assets<DepthMaterial>>()
            .add(DepthMaterial::new(max_depth));
        app.insert_resource(DepthMaterialHandle(depth_mat));

        // Startup: create output directory, spawn cameras, init seg colors.
        app.add_systems(
            Startup,
            (
                writer::create_run_directory,
                writer::init_seg_transform_colors,
                cameras::spawn_cosmos_cameras,
            )
                .chain(),
        );

        // PostUpdate: write frames and sync follow cameras.
        app.add_systems(
            PostUpdate,
            (
                cameras::sync_cosmos_follow_cameras,
                writer::write_cosmos_frames,
            )
                .chain(),
        );

        // Write metadata.json on app exit.
        app.add_systems(Last, metadata::write_cosmos_metadata_on_exit);
    }
}
