//! Semantic segmentation sensor — palette, frame buffer, and optional GPU plugin.
//!
//! [`SegmentationPalette`] maps class ids to flat RGB colours.
//! [`SegmentationFrameBuffer`] stores the most-recent segmentation render as
//! packed RGB bytes (one `u8` triple per pixel).
//!
//! When the `gpu` feature is enabled, [`ClankersSegmentationPlugin`] wires up
//! an offscreen camera that renders each segmentation class with a uniform flat
//! colour and copies the resulting pixels to the CPU via Bevy's GPU readback.

use std::collections::HashMap;

use bevy::prelude::Resource;

// ---------------------------------------------------------------------------
// SegmentationPalette
// ---------------------------------------------------------------------------

/// Maps segmentation class ids to flat sRGB colours `[r, g, b]` in `[0.0, 1.0]`
/// used during rendering.
///
/// The default palette assigns:
/// - `0` → `[1, 0, 0]` red (ground)
/// - `1` → `[0, 1, 0]` green (wall)
/// - `2` → `[0, 0, 1]` blue (robot)
/// - `3` → `[1, 1, 0]` yellow (obstacle)
/// - `4` → `[0.5, 0.5, 0.5]` gray (table)
///
/// # Example
///
/// ```
/// use clankers_render::segmentation::SegmentationPalette;
///
/// let palette = SegmentationPalette::default();
/// assert!(palette.colors.contains_key(&0));
/// let [r, g, b] = palette.colors[&0];
/// assert!((r - 1.0).abs() < 1e-6);
/// ```
#[derive(Resource, Clone, Debug)]
pub struct SegmentationPalette {
    /// Mapping from class id to sRGB colour `[r, g, b]` in `[0.0, 1.0]`.
    pub colors: HashMap<u32, [f32; 3]>,
}

impl Default for SegmentationPalette {
    fn default() -> Self {
        let mut colors = HashMap::new();
        colors.insert(0, [1.0, 0.0, 0.0]); // ground   = red
        colors.insert(1, [0.0, 1.0, 0.0]); // wall     = green
        colors.insert(2, [0.0, 0.0, 1.0]); // robot    = blue
        colors.insert(3, [1.0, 1.0, 0.0]); // obstacle = yellow
        colors.insert(4, [0.5, 0.5, 0.5]); // table    = gray
        Self { colors }
    }
}

// ---------------------------------------------------------------------------
// SegmentationFrameBuffer
// ---------------------------------------------------------------------------

/// Resource holding the most-recent segmentation render as packed RGB bytes.
///
/// Each pixel is stored as three consecutive `u8` values (R, G, B).
/// The buffer is written by the GPU readback system when the `gpu` feature
/// is active, or can be written directly for testing.
///
/// # Example
///
/// ```
/// use clankers_render::segmentation::SegmentationFrameBuffer;
///
/// let mut buf = SegmentationFrameBuffer::new(4, 2);
/// assert_eq!(buf.width(), 4);
/// assert_eq!(buf.height(), 2);
/// assert_eq!(buf.data().len(), 4 * 2 * 3);
/// ```
#[derive(Resource, Clone, Debug)]
pub struct SegmentationFrameBuffer {
    width: u32,
    height: u32,
    /// Packed RGB bytes: length is always `width * height * 3`.
    data: Vec<u8>,
    frame_counter: u64,
}

impl SegmentationFrameBuffer {
    /// Create a zero-filled segmentation frame buffer.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let byte_count = (width * height * 3) as usize;
        Self {
            width,
            height,
            data: vec![0; byte_count],
            frame_counter: 0,
        }
    }

    /// Width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Raw packed RGB bytes.
    ///
    /// Length is always `width * height * 3`.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Replace the entire frame and increment the frame counter.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal `width * height * 3`.
    pub fn write_frame(&mut self, data: Vec<u8>) {
        let expected = (self.width * self.height * 3) as usize;
        assert_eq!(
            data.len(),
            expected,
            "segmentation frame length {actual} does not match expected {expected}",
            actual = data.len(),
        );
        self.data = data;
        self.frame_counter += 1;
    }

    /// Number of frames written since creation.
    #[must_use]
    pub const fn frame_counter(&self) -> u64 {
        self.frame_counter
    }
}

impl Default for SegmentationFrameBuffer {
    fn default() -> Self {
        Self::new(512, 512)
    }
}

// ---------------------------------------------------------------------------
// GPU plugin (feature = "gpu")
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
pub use gpu_impl::*;

#[cfg(feature = "gpu")]
mod gpu_impl {
    use bevy::camera::RenderTarget;
    use bevy::prelude::*;
    use bevy::render::gpu_readback::{GpuReadbackPlugin, Readback, ReadbackComplete};
    use bevy::render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureUsages,
    };
    use bevy::render::view::RenderLayers;

    use crate::segmentation::{SegmentationFrameBuffer, SegmentationPalette};

    // wgpu guarantees rows in a texture-to-buffer copy are padded to 256 bytes.
    const COPY_BYTES_PER_ROW_ALIGNMENT: usize = 256;

    // -----------------------------------------------------------------------
    // SegmentationCamera
    // -----------------------------------------------------------------------

    /// Marker component identifying the segmentation-capture camera entity.
    ///
    /// Entities carrying this component render on [`RenderLayers::layer(1)`]
    /// and are managed by the segmentation readback system.
    #[derive(Component, Debug, Clone, Copy, Default)]
    pub struct SegmentationCamera;

    // -----------------------------------------------------------------------
    // SegmentationImageHandle
    // -----------------------------------------------------------------------

    /// Component holding the [`Handle<Image>`] for the segmentation render target.
    ///
    /// Attached to [`SegmentationCamera`] entities so the readback system can
    /// locate the RGBA image to schedule readback from.
    #[derive(Component, Debug, Clone)]
    pub struct SegmentationImageHandle(pub Handle<Image>);

    // -----------------------------------------------------------------------
    // SegmentationMaterials
    // -----------------------------------------------------------------------

    /// Resource holding pre-built unlit flat-colour material handles per class.
    ///
    /// Created by [`ClankersSegmentationPlugin`] at startup from the
    /// [`SegmentationPalette`].  The material swap system uses this resource
    /// to replace mesh materials with the correct flat colour for each class.
    #[derive(Resource, Default, Debug)]
    pub struct SegmentationMaterials {
        /// Maps class id → `Handle<StandardMaterial>`.
        pub handles: std::collections::HashMap<u32, Handle<StandardMaterial>>,
    }

    // -----------------------------------------------------------------------
    // ClankersSegmentationPlugin
    // -----------------------------------------------------------------------

    /// Bevy plugin that captures segmentation renders from GPU to CPU each frame.
    ///
    /// Add this plugin alongside [`ClankersRenderPlugin`][crate::ClankersRenderPlugin]
    /// to enable segmentation sensor readings.
    ///
    /// ```no_run
    /// use bevy::prelude::*;
    /// use clankers_render::prelude::*;
    /// use clankers_render::segmentation::{ClankersSegmentationPlugin, SegmentationFrameBuffer};
    ///
    /// App::new()
    ///     .add_plugins((DefaultPlugins, ClankersRenderPlugin, ClankersSegmentationPlugin))
    ///     .insert_resource(SegmentationFrameBuffer::new(512, 512))
    ///     .run();
    /// ```
    pub struct ClankersSegmentationPlugin;

    impl Plugin for ClankersSegmentationPlugin {
        fn build(&self, app: &mut App) {
            app.init_resource::<SegmentationPalette>()
                .init_resource::<SegmentationMaterials>()
                .add_plugins(GpuReadbackPlugin::default())
                .add_systems(Startup, build_segmentation_materials)
                .add_systems(Update, attach_readback_to_segmentation_cameras)
                .add_observer(handle_segmentation_readback_complete);
        }
    }

    // -----------------------------------------------------------------------
    // spawn_segmentation_camera_sensor
    // -----------------------------------------------------------------------

    /// Spawn an offscreen segmentation camera and prepare an RGBA image for readback.
    ///
    /// Creates a GPU [`Image`] asset with [`TextureFormat::Rgba8UnormSrgb`]
    /// configured with `COPY_SRC | TEXTURE_BINDING | RENDER_ATTACHMENT`
    /// usages, spawns a `Camera3d` entity on [`RenderLayers::layer(1)`] with
    /// [`SegmentationCamera`] marker and [`SegmentationImageHandle`], and
    /// inserts a matching [`SegmentationFrameBuffer`] resource.
    ///
    /// Returns the [`Entity`] of the spawned camera and the [`Handle<Image>`]
    /// of the render target.
    pub fn spawn_segmentation_camera_sensor(
        commands: &mut Commands,
        images: &mut Assets<Image>,
        width: u32,
        height: u32,
    ) -> (Entity, Handle<Image>) {
        // Build the RGBA render target image.
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let mut seg_image = Image::new_fill(
            size,
            TextureDimension::D2,
            &[0, 0, 0, 255],
            TextureFormat::Rgba8UnormSrgb,
            Default::default(),
        );
        seg_image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC
            | TextureUsages::RENDER_ATTACHMENT;

        let image_handle = images.add(seg_image);

        // Spawn the Camera3d entity on render layer 1 only.
        let camera_entity = commands
            .spawn((
                Camera3d::default(),
                Camera {
                    target: RenderTarget::Image(image_handle.clone().into()),
                    ..Default::default()
                },
                Transform::default(),
                GlobalTransform::default(),
                SegmentationCamera,
                SegmentationImageHandle(image_handle.clone()),
                RenderLayers::layer(1),
            ))
            .id();

        (camera_entity, image_handle)
    }

    // -----------------------------------------------------------------------
    // Systems
    // -----------------------------------------------------------------------

    /// Build one unlit flat-colour [`StandardMaterial`] per class from the palette.
    fn build_segmentation_materials(
        palette: Res<SegmentationPalette>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut seg_materials: ResMut<SegmentationMaterials>,
    ) {
        for (&class_id, &[r, g, b]) in &palette.colors {
            let mat = StandardMaterial {
                base_color: Color::srgb(r, g, b),
                unlit: true,
                ..Default::default()
            };
            let handle = materials.add(mat);
            seg_materials.handles.insert(class_id, handle);
        }
    }

    /// Attach a [`Readback`] component to each newly spawned [`SegmentationCamera`]
    /// entity so Bevy starts copying the render target image to the CPU.
    fn attach_readback_to_segmentation_cameras(
        mut commands: Commands,
        cameras: Query<
            (Entity, &SegmentationImageHandle),
            (With<SegmentationCamera>, Without<Readback>),
        >,
    ) {
        for (entity, img_handle) in &cameras {
            commands
                .entity(entity)
                .insert(Readback::texture(img_handle.0.clone()));
        }
    }

    /// Handle a completed GPU segmentation readback: interpret raw RGBA bytes,
    /// strip row padding, extract RGB channels, and write into
    /// [`SegmentationFrameBuffer`].
    pub fn handle_segmentation_readback_complete(
        trigger: On<ReadbackComplete>,
        cameras: Query<&SegmentationImageHandle, With<SegmentationCamera>>,
        mut seg_buf: ResMut<SegmentationFrameBuffer>,
    ) {
        let entity = trigger.entity;

        // Only handle readbacks for SegmentationCamera entities.
        let Ok(_handle) = cameras.get(entity) else {
            return;
        };

        let width = seg_buf.width();
        let height = seg_buf.height();

        // Raw RGBA bytes from the GPU; each pixel is 4 bytes.
        let raw: &[u8] = &trigger;

        let bytes_per_pixel = 4_usize; // RGBA
        let aligned_row = align_row_bytes(width as usize * bytes_per_pixel);
        let packed_rgba_row = width as usize * bytes_per_pixel;

        if raw.len() < aligned_row * height as usize {
            // Unexpected size: skip to avoid panic.
            return;
        }

        // Strip wgpu row padding and collect tightly-packed RGBA bytes.
        let mut rgba_bytes: Vec<u8> =
            Vec::with_capacity(packed_rgba_row * height as usize);
        for row in 0..height as usize {
            let start = row * aligned_row;
            rgba_bytes.extend_from_slice(&raw[start..start + packed_rgba_row]);
        }

        // Convert RGBA → RGB.
        let pixel_count = (width * height) as usize;
        if rgba_bytes.len() < pixel_count * 4 {
            return;
        }
        let mut rgb_bytes: Vec<u8> = Vec::with_capacity(pixel_count * 3);
        for chunk in rgba_bytes.chunks_exact(4) {
            rgb_bytes.push(chunk[0]); // R
            rgb_bytes.push(chunk[1]); // G
            rgb_bytes.push(chunk[2]); // B
        }

        if rgb_bytes.len() == pixel_count * 3 {
            seg_buf.write_frame(rgb_bytes);
        }
    }

    /// Round `bytes` up to the next multiple of [`COPY_BYTES_PER_ROW_ALIGNMENT`].
    const fn align_row_bytes(bytes: usize) -> usize {
        let align = COPY_BYTES_PER_ROW_ALIGNMENT;
        (bytes + align - 1) & !(align - 1)
    }

    // -----------------------------------------------------------------------
    // Marker for segmentation-visible entities
    // -----------------------------------------------------------------------

    /// Add this component to entities that should appear in both the main
    /// camera (layer 0) and the segmentation camera (layer 1).
    pub type SegmentationVisible = RenderLayers;

    /// Returns [`RenderLayers`] that includes both the default layer (0) and
    /// the segmentation layer (1).
    #[must_use]
    pub fn both_layers() -> RenderLayers {
        RenderLayers::from_layers(&[0, 1])
    }

    // -----------------------------------------------------------------------
    // GPU tests
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod gpu_tests {
        use super::*;

        #[test]
        fn segmentation_camera_marker_is_default() {
            let _cam = SegmentationCamera;
        }

        #[test]
        fn both_layers_contains_zero_and_one() {
            let layers = both_layers();
            assert!(layers.intersects(&RenderLayers::layer(0)));
            assert!(layers.intersects(&RenderLayers::layer(1)));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmentation_palette_default_has_five_classes() {
        let palette = SegmentationPalette::default();
        assert_eq!(palette.colors.len(), 5);
        assert!(palette.colors.contains_key(&0));
        assert!(palette.colors.contains_key(&1));
        assert!(palette.colors.contains_key(&2));
        assert!(palette.colors.contains_key(&3));
        assert!(palette.colors.contains_key(&4));
    }

    #[test]
    fn segmentation_palette_ground_is_red() {
        let palette = SegmentationPalette::default();
        let [r, g, b] = palette.colors[&0];
        assert!((r - 1.0).abs() < 1e-6, "ground red channel should be 1.0, got {r}");
        assert!(g < 0.01, "ground green channel should be ~0, got {g}");
        assert!(b < 0.01, "ground blue channel should be ~0, got {b}");
    }

    #[test]
    fn segmentation_frame_buffer_new() {
        let buf = SegmentationFrameBuffer::new(4, 2);
        assert_eq!(buf.width(), 4);
        assert_eq!(buf.height(), 2);
        assert_eq!(buf.data().len(), 4 * 2 * 3);
        assert_eq!(buf.frame_counter(), 0);
        assert!(buf.data().iter().all(|&b| b == 0));
    }

    #[test]
    fn segmentation_frame_buffer_write_read_roundtrip() {
        let mut buf = SegmentationFrameBuffer::new(2, 1);
        // 2 pixels × 3 channels = 6 bytes
        let data = vec![255_u8, 0, 0, 0, 255, 0];
        buf.write_frame(data.clone());
        assert_eq!(buf.data(), data.as_slice());
        assert_eq!(buf.frame_counter(), 1);
    }

    #[test]
    fn segmentation_frame_buffer_write_increments_counter() {
        let mut buf = SegmentationFrameBuffer::new(1, 1);
        buf.write_frame(vec![10, 20, 30]);
        buf.write_frame(vec![40, 50, 60]);
        assert_eq!(buf.frame_counter(), 2);
    }

    #[test]
    #[should_panic(expected = "segmentation frame length")]
    fn segmentation_frame_buffer_wrong_size_panics() {
        let mut buf = SegmentationFrameBuffer::new(2, 2);
        // Expected 4*3=12 bytes, passing 5
        buf.write_frame(vec![0; 5]);
    }

    #[test]
    fn segmentation_frame_buffer_default() {
        let buf = SegmentationFrameBuffer::default();
        assert_eq!(buf.width(), 512);
        assert_eq!(buf.height(), 512);
        assert_eq!(buf.data().len(), 512 * 512 * 3);
    }

    #[test]
    fn segmentation_frame_buffer_clone() {
        let mut buf = SegmentationFrameBuffer::new(1, 1);
        buf.write_frame(vec![1, 2, 3]);
        let buf2 = buf.clone();
        assert_eq!(buf2.data(), buf.data());
        assert_eq!(buf2.frame_counter(), buf.frame_counter());
    }
}
