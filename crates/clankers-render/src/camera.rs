//! Offscreen camera sensor spawning.
//!
//! [`SimCamera`] is a marker component that identifies camera entities owned
//! by the sensor system. [`spawn_camera_sensor`] creates a render-target
//! [`Image`], spawns a `Camera3d` entity pointed at that image, and registers
//! a matching [`FrameBuffer`] in [`CameraFrameBuffers`].
//!
//! This module is compiled only when the `gpu` feature is enabled because it
//! depends on `bevy_render`, `bevy_core_pipeline`, and `bevy_pbr`.

#[cfg(feature = "gpu")]
pub use gpu_impl::*;

// ---------------------------------------------------------------------------
// GPU implementation (feature = "gpu")
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod gpu_impl {
    use bevy::camera::RenderTarget;
    use bevy::prelude::*;
    use bevy::render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureUsages,
    };

    use crate::buffer::{CameraFrameBuffers, FrameBuffer};
    use crate::config::{CameraConfig, PixelFormat};

    // -----------------------------------------------------------------------
    // SimCamera
    // -----------------------------------------------------------------------

    /// Marker component identifying a camera entity managed by the sensor system.
    ///
    /// Entities carrying this component are queried by the readback system to
    /// route GPU-captured pixels into the appropriate [`FrameBuffer`].
    #[derive(Component, Debug, Clone, Copy, Default)]
    pub struct SimCamera;

    // -----------------------------------------------------------------------
    // spawn_camera_sensor
    // -----------------------------------------------------------------------

    /// Spawn an offscreen camera sensor and register its frame buffer.
    ///
    /// Creates a GPU [`Image`] asset configured as a render attachment with
    /// `COPY_SRC` usage (required for GPU readback), spawns a `Camera3d`
    /// entity targeting that image, attaches [`SimCamera`] and the provided
    /// [`CameraConfig`], and inserts a matching [`FrameBuffer`] into
    /// [`CameraFrameBuffers`].
    ///
    /// Returns the [`Entity`] of the spawned camera and the [`Handle<Image>`]
    /// of the render target.
    ///
    /// # Panics
    ///
    /// Panics if `config.label` is empty — every sensor camera must have a
    /// unique, non-empty label so the readback system can route pixels
    /// correctly.
    pub fn spawn_camera_sensor(
        commands: &mut Commands,
        images: &mut Assets<Image>,
        camera_frame_buffers: &mut CameraFrameBuffers,
        config: CameraConfig,
        width: u32,
        height: u32,
    ) -> (Entity, Handle<Image>) {
        assert!(
            !config.label.is_empty(),
            "CameraConfig::label must not be empty when spawning a camera sensor"
        );

        // Build the render target image.
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let mut render_target = Image::new_fill(
            size,
            TextureDimension::D2,
            &[0, 0, 0, 255],
            TextureFormat::Rgba8UnormSrgb,
            Default::default(),
        );
        render_target.texture_descriptor.usage =
            TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_SRC;

        let image_handle = images.add(render_target);

        // Register a CPU-side frame buffer keyed by label.
        camera_frame_buffers.insert(
            config.label.clone(),
            FrameBuffer::new(width, height, PixelFormat::Rgba8),
        );

        // Spawn the Camera3d entity.
        let camera_entity = commands
            .spawn((
                Camera3d::default(),
                Camera {
                    target: RenderTarget::Image(image_handle.clone().into()),
                    ..Default::default()
                },
                Transform::default(),
                GlobalTransform::default(),
                SimCamera,
                config,
            ))
            .id();

        (camera_entity, image_handle)
    }
}
