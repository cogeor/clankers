//! Headless rendering and GPU-to-CPU image transfer for visual observations.
//!
//! This crate provides the data structures and Bevy plugin for capturing
//! rendered frames into a CPU-side buffer:
//!
//! - [`RenderConfig`] — resolution and pixel format settings
//! - [`FrameBuffer`] — stores a single frame of raw pixel data
//! - [`CameraConfig`] — per-camera projection parameters
//! - [`ClankersRenderPlugin`] — Bevy plugin that initialises resources
//!
//! The design is rendering-backend-agnostic. In headless mode (Bevy without
//! rendering features) the frame buffer exists but is only written to when
//! external code explicitly calls [`FrameBuffer::write_frame`]. When a
//! rendering backend is active, a capture system can transfer GPU output
//! into the frame buffer each frame.
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
pub mod config;
pub mod sensor;

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use buffer::FrameBuffer;
pub use config::{CameraConfig, RenderConfig};
pub use sensor::ImageSensor;

// ---------------------------------------------------------------------------
// ClankersRenderPlugin
// ---------------------------------------------------------------------------

/// Bevy plugin that initialises render resources.
///
/// Registers [`RenderConfig`] and [`FrameBuffer`]. The frame buffer is
/// created from the current render config at startup.
///
/// Runs a startup system that builds the initial [`FrameBuffer`] from the
/// [`RenderConfig`]. If no custom config is inserted before plugin build,
/// the default 512x512 RGB8 config is used.
pub struct ClankersRenderPlugin;

impl Plugin for ClankersRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderConfig>();

        // Build the frame buffer from the render config at startup.
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
        buffer::FrameBuffer,
        config::{CameraConfig, PixelFormat, RenderConfig},
        sensor::ImageSensor,
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
}
