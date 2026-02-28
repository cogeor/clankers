//! Headless rendering and GPU-to-CPU image transfer for visual observations.
//!
//! This crate provides the data structures and Bevy plugin for capturing
//! rendered frames into CPU-side buffers:
//!
//! - [`RenderConfig`] — resolution and pixel format settings
//! - [`FrameBuffer`] — stores a single frame of raw pixel data per camera
//! - [`CameraFrameBuffers`] — resource mapping camera labels to frame buffers
//! - [`CameraConfig`] — per-camera projection parameters (includes `label`)
//! - [`DepthFrameBuffer`] — resource holding raw `f32` depth values per pixel
//! - [`ClankersRenderPlugin`] — Bevy plugin that initialises resources
//!
//! When the `gpu` feature is enabled, additional types become available:
//!
//! - [`camera::SimCamera`] — marker component for sensor cameras
//! - [`camera::spawn_camera_sensor`] — helper to create offscreen cameras
//! - [`readback::ImageCopyPlugin`] — copies GPU frames into [`CameraFrameBuffers`]
//! - [`depth::ClankersDepthPlugin`] — copies GPU depth frames into [`DepthFrameBuffer`]
//! - [`depth::DepthCamera`] — marker component for depth-capture cameras
//! - [`depth::spawn_depth_camera_sensor`] — helper to create depth cameras
//!
//! The design is rendering-backend-agnostic. In headless mode the frame buffers
//! exist but are only written to when external code explicitly calls
//! [`FrameBuffer::write_frame`] or [`DepthFrameBuffer::write_depth_frame`].
//! When the `gpu` feature is active and the appropriate plugin is added to the
//! app, the readback systems transfer GPU output into the buffers each frame.
//!
//! # Example
//!
//! ```no_run
//! use bevy::prelude::*;
//! use clankers_render::prelude::*;
//!
//! App::new()
//!     .add_plugins(clankers_core::ClankersCorePlugin)
//!     .add_plugins(ClankersRenderPlugin)
//!     .insert_resource(RenderConfig::new(512, 512))
//!     .run();
//! ```

pub mod buffer;
pub mod camera;
pub mod config;
pub mod depth;
pub mod readback;
pub mod sensor;

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use buffer::{CameraFrameBuffers, DepthFrameBuffer, FrameBuffer};
pub use config::{CameraConfig, RenderConfig};
pub use sensor::{DepthSensor, ImageSensor};

// ---------------------------------------------------------------------------
// ClankersRenderPlugin
// ---------------------------------------------------------------------------

/// Bevy plugin that initialises render resources.
///
/// Registers [`RenderConfig`], [`FrameBuffer`], and [`CameraFrameBuffers`].
/// The single legacy frame buffer is created from the current render config at
/// startup (for backward compatibility). Per-camera buffers are registered by
/// [`camera::spawn_camera_sensor`] or added manually.
///
/// When the `gpu` feature is active, add [`readback::ImageCopyPlugin`]
/// separately to enable live GPU readback.
pub struct ClankersRenderPlugin;

impl Plugin for ClankersRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderConfig>();
        app.init_resource::<CameraFrameBuffers>();

        // Build the legacy single-camera frame buffer from the render config
        // at startup.
        app.add_systems(Startup, init_frame_buffer);
    }
}

/// Startup system that creates the [`FrameBuffer`] from [`RenderConfig`].
#[allow(clippy::needless_pass_by_value)]
fn init_frame_buffer(mut commands: Commands, config: Res<RenderConfig>) {
    commands.insert_resource(FrameBuffer::from_config(&config));
}

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::{
        ClankersRenderPlugin,
        buffer::{CameraFrameBuffers, DepthFrameBuffer, FrameBuffer},
        config::{CameraConfig, PixelFormat, RenderConfig},
        sensor::{DepthSensor, ImageSensor},
    };

    #[cfg(feature = "gpu")]
    pub use crate::{
        camera::SimCamera,
        depth::{ClankersDepthPlugin, DepthCamera, spawn_depth_camera_sensor},
        readback::ImageCopyPlugin,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PixelFormat;

    #[test]
    fn plugin_builds_without_panic() {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(ClankersRenderPlugin);
        app.finish();
        app.cleanup();
        app.update();

        assert!(app.world().get_resource::<RenderConfig>().is_some());
        assert!(app.world().get_resource::<FrameBuffer>().is_some());
        assert!(app.world().get_resource::<CameraFrameBuffers>().is_some());
    }

    #[test]
    fn plugin_creates_buffer_matching_config() {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.insert_resource(RenderConfig::new(64, 32).with_format(PixelFormat::Rgba8));
        app.add_plugins(ClankersRenderPlugin);
        app.finish();
        app.cleanup();
        app.update();

        let buf = app.world().resource::<FrameBuffer>();
        assert_eq!(buf.width(), 64);
        assert_eq!(buf.height(), 32);
        assert_eq!(buf.format(), PixelFormat::Rgba8);
        assert_eq!(buf.byte_count(), 64 * 32 * 4);
    }

    #[test]
    fn plugin_default_config() {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(ClankersRenderPlugin);
        app.finish();
        app.cleanup();
        app.update();

        let config = app.world().resource::<RenderConfig>();
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
    }

    #[test]
    fn frame_buffer_writable_after_init() {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.insert_resource(RenderConfig::new(2, 1));
        app.add_plugins(ClankersRenderPlugin);
        app.finish();
        app.cleanup();
        app.update();

        let mut buf = app.world_mut().resource_mut::<FrameBuffer>();
        buf.write_frame(vec![255, 0, 0, 0, 255, 0]);
        assert_eq!(buf.frame_counter(), 1);
        assert_eq!(buf.pixel(0, 0), &[255, 0, 0]);
    }

    #[test]
    fn camera_frame_buffers_initialised_empty() {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(ClankersRenderPlugin);
        app.finish();
        app.cleanup();
        app.update();

        let bufs = app.world().resource::<CameraFrameBuffers>();
        assert!(bufs.is_empty());
    }
}
