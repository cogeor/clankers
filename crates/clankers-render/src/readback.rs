//! GPU-to-CPU readback plugin for camera sensor frames.
//!
//! [`ImageCopyPlugin`] registers the Bevy [`GpuReadbackPlugin`] and adds a
//! system that:
//!
//! 1. Detects newly spawned [`SimCamera`] entities and attaches a [`Readback`]
//!    component so Bevy begins copying pixels from the GPU every frame.
//! 2. Listens for [`ReadbackComplete`] events on those entities and strips
//!    wgpu row-padding before writing the packed pixels into
//!    [`CameraFrameBuffers`] keyed by the camera's label.
//!
//! This module is compiled only when the `gpu` feature is enabled.

#[cfg(feature = "gpu")]
pub use gpu_impl::*;

// ---------------------------------------------------------------------------
// GPU implementation (feature = "gpu")
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod gpu_impl {
    use bevy::prelude::*;
    use bevy::render::gpu_readback::{GpuReadbackPlugin, Readback, ReadbackComplete};

    use crate::buffer::CameraFrameBuffers;
    use crate::camera::SimCamera;
    use crate::config::CameraConfig;

    // wgpu guarantees rows in a texture-to-buffer copy are padded to 256 bytes.
    // This is a stable constant defined in the WebGPU / wgpu specification.
    const COPY_BYTES_PER_ROW_ALIGNMENT: usize = 256;

    // -----------------------------------------------------------------------
    // ImageCopyPlugin
    // -----------------------------------------------------------------------

    /// Bevy plugin that copies camera render targets from GPU to CPU each frame.
    ///
    /// Add this plugin alongside [`ClankersRenderPlugin`][crate::ClankersRenderPlugin]
    /// when you need live pixel data from [`SimCamera`] entities.
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use clankers_render::prelude::*;
    /// use clankers_render::readback::ImageCopyPlugin;
    ///
    /// App::new()
    ///     .add_plugins((DefaultPlugins, ClankersRenderPlugin, ImageCopyPlugin))
    ///     .run();
    /// ```
    pub struct ImageCopyPlugin;

    impl Plugin for ImageCopyPlugin {
        fn build(&self, app: &mut App) {
            app.add_plugins(GpuReadbackPlugin::default())
                .add_systems(Update, attach_readback_to_new_cameras)
                .add_observer(handle_readback_complete);
        }
    }

    // -----------------------------------------------------------------------
    // Systems
    // -----------------------------------------------------------------------

    /// Attach a [`Readback`] component to each newly spawned [`SimCamera`]
    /// entity so Bevy starts copying the render target image to the CPU.
    fn attach_readback_to_new_cameras(
        mut commands: Commands,
        cameras: Query<(Entity, &Camera), (With<SimCamera>, Without<Readback>)>,
    ) {
        for (entity, camera) in &cameras {
            if let Some(handle) = camera.target.as_image() {
                commands
                    .entity(entity)
                    .insert(Readback::texture(handle.clone()));
            }
        }
    }

    /// Handle a completed GPU readback: strip wgpu row-padding and write
    /// packed pixels into [`CameraFrameBuffers`] for the camera's label.
    fn handle_readback_complete(
        trigger: On<ReadbackComplete>,
        cameras: Query<&CameraConfig, With<SimCamera>>,
        mut camera_frame_buffers: ResMut<CameraFrameBuffers>,
    ) {
        // ReadbackComplete is an EntityEvent: its `entity` field holds the
        // entity the event was triggered on. Since `On<E>` derefs to `E`,
        // we can access `trigger.entity` directly.
        let entity = trigger.entity;
        let Ok(config) = cameras.get(entity) else {
            return;
        };
        let label = &config.label;
        let Some(buf) = camera_frame_buffers.get_mut(label) else {
            return;
        };

        let width = buf.width();
        let height = buf.height();
        let bytes_per_pixel = buf.format().bytes_per_pixel() as usize;
        // ReadbackComplete derefs to Vec<u8> via its #[deref] attribute.
        let raw: &[u8] = &trigger;

        // wgpu aligns each row to 256 bytes (COPY_BYTES_PER_ROW_ALIGNMENT).
        // Strip that padding to produce a tightly-packed frame.
        let aligned_row = align_row_bytes(width as usize * bytes_per_pixel);
        let packed_row = width as usize * bytes_per_pixel;

        if raw.len() < aligned_row * height as usize {
            // Unexpected size; skip silently to avoid a panic.
            return;
        }

        let mut packed = Vec::with_capacity(packed_row * height as usize);
        for row in 0..height as usize {
            let start = row * aligned_row;
            packed.extend_from_slice(&raw[start..start + packed_row]);
        }

        // Guard against format mismatches rather than propagate the panic.
        let expected = (width * height * buf.format().bytes_per_pixel()) as usize;
        if packed.len() == expected {
            buf.write_frame(packed);
        }
    }

    /// Round `bytes` up to the next multiple of `COPY_BYTES_PER_ROW_ALIGNMENT`.
    const fn align_row_bytes(bytes: usize) -> usize {
        let align = COPY_BYTES_PER_ROW_ALIGNMENT;
        (bytes + align - 1) & !(align - 1)
    }
}
