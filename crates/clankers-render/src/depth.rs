//! GPU depth-capture plugin for depth sensor frames.
//!
//! [`ClankersDepthPlugin`] registers GPU readback infrastructure for depth
//! sensing. It inserts a [`DepthFrameBuffer`] resource and provides the
//! [`DepthCamera`] marker component along with the
//! [`spawn_depth_camera_sensor`] helper.
//!
//! The plugin creates an offscreen image with `Depth32Float` format and attaches
//! [`Readback::texture`] to the camera entity so that Bevy's
//! [`GpuReadbackPlugin`] copies the depth attachment bytes to the CPU each
//! frame. The [`handle_depth_readback_complete`] observer interprets those
//! bytes as `f32` depth values (one per pixel) and writes them into the
//! [`DepthFrameBuffer`].
//!
//! Note: connecting the camera's internal depth prepass texture to the
//! registered [`Image`] asset requires a custom render-graph node. This
//! plugin wires up the CPU side completely; the GPU-side render graph
//! integration is expected to be added in a future loop.
//!
//! This module is compiled only when the `gpu` feature is enabled.

#[cfg(feature = "gpu")]
pub use gpu_impl::*;

// ---------------------------------------------------------------------------
// GPU implementation (feature = "gpu")
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod gpu_impl {
    use bevy::core_pipeline::prepass::DepthPrepass;
    use bevy::prelude::*;
    use bevy::render::gpu_readback::{GpuReadbackPlugin, Readback, ReadbackComplete};
    use bevy::render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureUsages,
    };

    use crate::buffer::DepthFrameBuffer;

    // wgpu guarantees rows in a texture-to-buffer copy are padded to 256 bytes.
    const COPY_BYTES_PER_ROW_ALIGNMENT: usize = 256;

    // -----------------------------------------------------------------------
    // DepthCamera
    // -----------------------------------------------------------------------

    /// Marker component identifying a depth-capture camera entity.
    ///
    /// Entities carrying this component have a [`DepthPrepass`] component and
    /// are managed by the depth readback system.
    #[derive(Component, Debug, Clone, Copy, Default)]
    pub struct DepthCamera;

    // -----------------------------------------------------------------------
    // DepthImageHandle
    // -----------------------------------------------------------------------

    /// Component holding the [`Handle<Image>`] for the depth render target.
    ///
    /// Attached to [`DepthCamera`] entities so the readback system can find
    /// the depth image to schedule readback from.
    #[derive(Component, Debug, Clone)]
    pub struct DepthImageHandle(pub Handle<Image>);

    // -----------------------------------------------------------------------
    // ClankersDepthPlugin
    // -----------------------------------------------------------------------

    /// Bevy plugin that captures depth from GPU to CPU each frame.
    ///
    /// Insert a [`DepthFrameBuffer`] resource and add this plugin alongside
    /// [`ClankersRenderPlugin`][crate::ClankersRenderPlugin] to enable depth
    /// sensor readings.
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use clankers_render::prelude::*;
    /// use clankers_render::depth::ClankersDepthPlugin;
    /// use clankers_render::buffer::DepthFrameBuffer;
    ///
    /// App::new()
    ///     .add_plugins((DefaultPlugins, ClankersRenderPlugin, ClankersDepthPlugin))
    ///     .insert_resource(DepthFrameBuffer::new(512, 512))
    ///     .run();
    /// ```
    pub struct ClankersDepthPlugin;

    impl Plugin for ClankersDepthPlugin {
        fn build(&self, app: &mut App) {
            app.add_plugins(GpuReadbackPlugin::default())
                .add_systems(Update, attach_readback_to_depth_cameras)
                .add_observer(handle_depth_readback_complete);
        }
    }

    // -----------------------------------------------------------------------
    // spawn_depth_camera_sensor
    // -----------------------------------------------------------------------

    /// Spawn an offscreen depth camera and prepare a depth image for readback.
    ///
    /// Creates a GPU [`Image`] asset with [`TextureFormat::Depth32Float`]
    /// configured with `COPY_SRC | TEXTURE_BINDING | RENDER_ATTACHMENT`
    /// usages, spawns a `Camera3d` entity with [`DepthPrepass`] and
    /// [`DepthCamera`], attaches [`DepthImageHandle`] so the readback system
    /// can locate the depth image, and inserts a matching
    /// [`DepthFrameBuffer`] resource.
    ///
    /// Returns the [`Entity`] of the spawned camera and the [`Handle<Image>`]
    /// of the depth render target.
    pub fn spawn_depth_camera_sensor(
        commands: &mut Commands,
        images: &mut Assets<Image>,
        width: u32,
        height: u32,
    ) -> (Entity, Handle<Image>) {
        // Build the depth render target image.
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Depth32Float stores one f32 per pixel. COPY_SRC is required for GPU
        // readback; TEXTURE_BINDING allows sampling in shaders (needed by
        // DepthPrepass); RENDER_ATTACHMENT lets the camera write depth here.
        let mut depth_image = Image::new_fill(
            size,
            TextureDimension::D2,
            // 4 zero bytes = 0.0f32 initial depth
            &[0u8, 0u8, 0u8, 0u8],
            TextureFormat::Depth32Float,
            Default::default(),
        );
        depth_image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC
            | TextureUsages::RENDER_ATTACHMENT;

        let depth_handle = images.add(depth_image);

        // Spawn a Camera3d with DepthPrepass so Bevy generates the depth
        // texture during the prepass phase.
        let camera_entity = commands
            .spawn((
                Camera3d::default(),
                Camera::default(),
                Transform::default(),
                GlobalTransform::default(),
                DepthCamera,
                DepthPrepass,
                DepthImageHandle(depth_handle.clone()),
            ))
            .id();

        (camera_entity, depth_handle)
    }

    // -----------------------------------------------------------------------
    // Systems
    // -----------------------------------------------------------------------

    /// Attach a [`Readback`] component to each newly spawned [`DepthCamera`]
    /// entity so Bevy starts copying the depth image to the CPU.
    fn attach_readback_to_depth_cameras(
        mut commands: Commands,
        cameras: Query<(Entity, &DepthImageHandle), (With<DepthCamera>, Without<Readback>)>,
    ) {
        for (entity, depth_handle) in &cameras {
            commands
                .entity(entity)
                .insert(Readback::texture(depth_handle.0.clone()));
        }
    }

    /// Handle a completed GPU depth readback: interpret raw bytes as `f32`
    /// depth values and write them into the [`DepthFrameBuffer`] resource.
    ///
    /// # Row Alignment
    ///
    /// wgpu aligns texture rows to 256 bytes during buffer copies. Each pixel
    /// is 4 bytes (`f32`), so the aligned row stride may be wider than the
    /// actual pixel row. This function strips the padding before writing.
    pub fn handle_depth_readback_complete(
        trigger: On<ReadbackComplete>,
        cameras: Query<&DepthImageHandle, With<DepthCamera>>,
        mut depth_buf: ResMut<DepthFrameBuffer>,
    ) {
        let entity = trigger.entity;

        // Only handle readbacks for DepthCamera entities.
        let Ok(_handle) = cameras.get(entity) else {
            return;
        };

        let width = depth_buf.width();
        let height = depth_buf.height();

        // Raw bytes from the GPU: each pixel is a 4-byte little-endian f32.
        let raw: &[u8] = &trigger;

        let bytes_per_pixel = 4_usize; // f32
        let aligned_row = align_row_bytes(width as usize * bytes_per_pixel);
        let packed_row = width as usize * bytes_per_pixel;

        if raw.len() < aligned_row * height as usize {
            // Unexpected size: skip to avoid panic.
            return;
        }

        // Strip wgpu row padding and collect tightly-packed bytes.
        let mut packed_bytes: Vec<u8> = Vec::with_capacity(packed_row * height as usize);
        for row in 0..height as usize {
            let start = row * aligned_row;
            packed_bytes.extend_from_slice(&raw[start..start + packed_row]);
        }

        // Reinterpret the packed bytes as f32 values.
        let expected_pixels = (width * height) as usize;
        if packed_bytes.len() != expected_pixels * bytes_per_pixel {
            return;
        }

        let depth_values: Vec<f32> = packed_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(bytes)
            })
            .collect();

        if depth_values.len() == expected_pixels {
            depth_buf.write_depth_frame(depth_values);
        }
    }

    /// Round `bytes` up to the next multiple of [`COPY_BYTES_PER_ROW_ALIGNMENT`].
    const fn align_row_bytes(bytes: usize) -> usize {
        let align = COPY_BYTES_PER_ROW_ALIGNMENT;
        (bytes + align - 1) & !(align - 1)
    }
}
