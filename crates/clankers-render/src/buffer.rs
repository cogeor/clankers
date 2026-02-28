//! Frame buffer for storing captured pixel data.
//!
//! [`FrameBuffer`] holds a single frame of pixel data per camera.
//! [`CameraFrameBuffers`] is a Bevy resource that maps camera labels to their
//! respective [`FrameBuffer`]s.
//!
//! In headless mode the buffers exist but are only written to when external
//! code explicitly calls [`FrameBuffer::write_frame`]. When the `gpu` feature
//! is active, the readback system transfers GPU output into the buffers each
//! frame.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::config::{PixelFormat, RenderConfig};

// ---------------------------------------------------------------------------
// FrameBuffer
// ---------------------------------------------------------------------------

/// Resource holding a single captured frame of pixel data.
///
/// # Example
///
/// ```
/// use clankers_render::FrameBuffer;
/// use clankers_render::config::PixelFormat;
///
/// let buf = FrameBuffer::new(4, 2, PixelFormat::Rgb8);
/// assert_eq!(buf.width(), 4);
/// assert_eq!(buf.height(), 2);
/// assert_eq!(buf.data().len(), 4 * 2 * 3);
/// ```
#[derive(Resource, Clone, Debug)]
pub struct FrameBuffer {
    width: u32,
    height: u32,
    format: PixelFormat,
    data: Vec<u8>,
    frame_counter: u64,
}

impl FrameBuffer {
    /// Create a zero-filled frame buffer.
    #[must_use]
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Self {
        let byte_count = (width * height * format.bytes_per_pixel()) as usize;
        Self {
            width,
            height,
            format,
            data: vec![0; byte_count],
            frame_counter: 0,
        }
    }

    /// Create a frame buffer from a [`RenderConfig`].
    #[must_use]
    pub fn from_config(config: &RenderConfig) -> Self {
        Self::new(config.width, config.height, config.format)
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

    /// Pixel format.
    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    /// Raw pixel data as a byte slice.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Mutable access to the raw pixel data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Replace the entire frame data and increment the frame counter.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not match the expected frame size.
    pub fn write_frame(&mut self, data: Vec<u8>) {
        let expected = (self.width * self.height * self.format.bytes_per_pixel()) as usize;
        assert_eq!(
            data.len(),
            expected,
            "frame data length {actual} does not match expected {expected}",
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

    /// Total byte count of the frame.
    #[must_use]
    pub const fn byte_count(&self) -> usize {
        self.data.len()
    }

    /// Access a single pixel by (x, y) coordinates. Returns a slice of
    /// `bytes_per_pixel` bytes.
    ///
    /// # Panics
    ///
    /// Panics if coordinates are out of bounds.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> &[u8] {
        assert!(x < self.width, "x={x} out of bounds (width={})", self.width);
        assert!(
            y < self.height,
            "y={y} out of bounds (height={})",
            self.height
        );
        let bpp = self.format.bytes_per_pixel() as usize;
        let offset = ((y * self.width + x) as usize) * bpp;
        &self.data[offset..offset + bpp]
    }
}

impl Default for FrameBuffer {
    fn default() -> Self {
        let config = RenderConfig::default();
        Self::from_config(&config)
    }
}

// ---------------------------------------------------------------------------
// DepthFrameBuffer
// ---------------------------------------------------------------------------

/// Resource holding a single captured frame of linear depth values.
///
/// Each element in [`data()`][Self::data] is a raw 32-bit floating-point depth
/// sample in the range `[0.0, 1.0]` as produced by the GPU depth attachment.
/// Convert to metres using the linearisation formula in [`DepthSensor`].
///
/// [`DepthSensor`]: crate::sensor::DepthSensor
///
/// # Example
///
/// ```
/// use clankers_render::buffer::DepthFrameBuffer;
///
/// let mut buf = DepthFrameBuffer::new(4, 2);
/// assert_eq!(buf.width(), 4);
/// assert_eq!(buf.height(), 2);
/// assert_eq!(buf.data().len(), 4 * 2);
/// ```
#[derive(Resource, Clone, Debug)]
pub struct DepthFrameBuffer {
    width: u32,
    height: u32,
    data: Vec<f32>,
    frame_counter: u64,
}

impl DepthFrameBuffer {
    /// Create a zero-filled depth frame buffer.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let pixel_count = (width * height) as usize;
        Self {
            width,
            height,
            data: vec![0.0; pixel_count],
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

    /// Raw depth data as a slice of f32 values.
    ///
    /// Length is always `width * height`.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Replace the entire depth frame and increment the frame counter.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal `width * height`.
    pub fn write_depth_frame(&mut self, data: Vec<f32>) {
        let expected = (self.width * self.height) as usize;
        assert_eq!(
            data.len(),
            expected,
            "depth frame length {actual} does not match expected {expected}",
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

impl Default for DepthFrameBuffer {
    fn default() -> Self {
        Self::new(512, 512)
    }
}

// ---------------------------------------------------------------------------
// CameraFrameBuffers
// ---------------------------------------------------------------------------

/// Resource holding one [`FrameBuffer`] per named camera sensor.
///
/// Each camera is identified by the label set in [`CameraConfig::label`]. The
/// readback system (active under the `gpu` feature) writes captured frames into
/// the appropriate entry. Call [`get`][Self::get] to read the latest pixels for
/// a given camera.
///
/// # Example
///
/// ```
/// use clankers_render::buffer::{CameraFrameBuffers, FrameBuffer};
/// use clankers_render::config::PixelFormat;
///
/// let mut bufs = CameraFrameBuffers::default();
/// bufs.insert("front".to_string(), FrameBuffer::new(4, 4, PixelFormat::Rgba8));
/// assert!(bufs.get("front").is_some());
/// ```
///
/// [`CameraConfig::label`]: crate::config::CameraConfig::label
#[derive(Resource, Default, Debug)]
pub struct CameraFrameBuffers(HashMap<String, FrameBuffer>);

impl CameraFrameBuffers {
    /// Return a reference to the [`FrameBuffer`] for the named camera, or
    /// `None` if no buffer has been registered under that label.
    pub fn get(&self, label: &str) -> Option<&FrameBuffer> {
        self.0.get(label)
    }

    /// Return a mutable reference to the [`FrameBuffer`] for the named camera,
    /// or `None` if no buffer has been registered under that label.
    pub fn get_mut(&mut self, label: &str) -> Option<&mut FrameBuffer> {
        self.0.get_mut(label)
    }

    /// Register (or replace) a [`FrameBuffer`] under the given label.
    pub fn insert(&mut self, label: String, buf: FrameBuffer) {
        self.0.insert(label, buf);
    }

    /// Remove the [`FrameBuffer`] registered under the given label.
    pub fn remove(&mut self, label: &str) -> Option<FrameBuffer> {
        self.0.remove(label)
    }

    /// Iterate over all (label, buffer) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &FrameBuffer)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Iterate mutably over all (label, buffer) pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut FrameBuffer)> {
        self.0.iter_mut().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of cameras registered.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if no cameras are registered.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_buffer_new_rgb() {
        let buf = FrameBuffer::new(8, 4, PixelFormat::Rgb8);
        assert_eq!(buf.width(), 8);
        assert_eq!(buf.height(), 4);
        assert_eq!(buf.format(), PixelFormat::Rgb8);
        assert_eq!(buf.byte_count(), 8 * 4 * 3);
        assert_eq!(buf.frame_counter(), 0);
        assert!(buf.data().iter().all(|&b| b == 0));
    }

    #[test]
    fn frame_buffer_new_rgba() {
        let buf = FrameBuffer::new(2, 2, PixelFormat::Rgba8);
        assert_eq!(buf.byte_count(), 2 * 2 * 4);
    }

    #[test]
    fn frame_buffer_from_config() {
        let config = RenderConfig::new(16, 8).with_format(PixelFormat::Rgba8);
        let buf = FrameBuffer::from_config(&config);
        assert_eq!(buf.width(), 16);
        assert_eq!(buf.height(), 8);
        assert_eq!(buf.format(), PixelFormat::Rgba8);
    }

    #[test]
    fn frame_buffer_default() {
        let buf = FrameBuffer::default();
        assert_eq!(buf.width(), 512);
        assert_eq!(buf.height(), 512);
        assert_eq!(buf.format(), PixelFormat::Rgb8);
    }

    #[test]
    fn frame_buffer_write_frame() {
        let mut buf = FrameBuffer::new(2, 1, PixelFormat::Rgb8);
        let data = vec![255, 0, 0, 0, 255, 0];
        buf.write_frame(data.clone());
        assert_eq!(buf.data(), &data);
        assert_eq!(buf.frame_counter(), 1);
    }

    #[test]
    fn frame_buffer_write_increments_counter() {
        let mut buf = FrameBuffer::new(1, 1, PixelFormat::Rgb8);
        buf.write_frame(vec![1, 2, 3]);
        buf.write_frame(vec![4, 5, 6]);
        assert_eq!(buf.frame_counter(), 2);
    }

    #[test]
    #[should_panic(expected = "frame data length")]
    fn frame_buffer_write_wrong_size_panics() {
        let mut buf = FrameBuffer::new(2, 2, PixelFormat::Rgb8);
        buf.write_frame(vec![0; 5]); // Expected 12
    }

    #[test]
    fn frame_buffer_data_mut() {
        let mut buf = FrameBuffer::new(1, 1, PixelFormat::Rgb8);
        buf.data_mut()[0] = 128;
        assert_eq!(buf.data()[0], 128);
    }

    #[test]
    fn frame_buffer_pixel_rgb() {
        let mut buf = FrameBuffer::new(2, 2, PixelFormat::Rgb8);
        // Write a known pattern: row-major, pixel (1,0) is bytes 3..6
        let data = vec![
            10, 20, 30, // (0,0)
            40, 50, 60, // (1,0)
            70, 80, 90, // (0,1)
            100, 110, 120, // (1,1)
        ];
        buf.write_frame(data);
        assert_eq!(buf.pixel(0, 0), &[10, 20, 30]);
        assert_eq!(buf.pixel(1, 0), &[40, 50, 60]);
        assert_eq!(buf.pixel(0, 1), &[70, 80, 90]);
        assert_eq!(buf.pixel(1, 1), &[100, 110, 120]);
    }

    #[test]
    fn frame_buffer_pixel_rgba() {
        let mut buf = FrameBuffer::new(2, 1, PixelFormat::Rgba8);
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        buf.write_frame(data);
        assert_eq!(buf.pixel(0, 0), &[1, 2, 3, 4]);
        assert_eq!(buf.pixel(1, 0), &[5, 6, 7, 8]);
    }

    #[test]
    #[should_panic(expected = "x=2 out of bounds")]
    fn frame_buffer_pixel_x_out_of_bounds() {
        let buf = FrameBuffer::new(2, 2, PixelFormat::Rgb8);
        let _ = buf.pixel(2, 0);
    }

    #[test]
    #[should_panic(expected = "y=2 out of bounds")]
    fn frame_buffer_pixel_y_out_of_bounds() {
        let buf = FrameBuffer::new(2, 2, PixelFormat::Rgb8);
        let _ = buf.pixel(0, 2);
    }

    #[test]
    fn frame_buffer_clone() {
        let buf = FrameBuffer::new(2, 2, PixelFormat::Rgb8);
        let buf2 = buf.clone();
        assert_eq!(buf.width(), buf2.width());
        assert_eq!(buf.height(), buf2.height());
        assert_eq!(buf.data(), buf2.data());
    }

    // -----------------------------------------------------------------------
    // CameraFrameBuffers tests
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // DepthFrameBuffer tests
    // -----------------------------------------------------------------------

    #[test]
    fn depth_frame_buffer_new() {
        let buf = DepthFrameBuffer::new(4, 2);
        assert_eq!(buf.width(), 4);
        assert_eq!(buf.height(), 2);
        assert_eq!(buf.data().len(), 8);
        assert_eq!(buf.frame_counter(), 0);
        assert!(buf.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn depth_frame_buffer_write_read_roundtrip() {
        let mut buf = DepthFrameBuffer::new(2, 1);
        let data = vec![0.25_f32, 0.75_f32];
        buf.write_depth_frame(data.clone());
        assert_eq!(buf.data(), data.as_slice());
        assert_eq!(buf.frame_counter(), 1);
    }

    #[test]
    fn depth_frame_buffer_write_increments_counter() {
        let mut buf = DepthFrameBuffer::new(1, 1);
        buf.write_depth_frame(vec![0.1]);
        buf.write_depth_frame(vec![0.9]);
        assert_eq!(buf.frame_counter(), 2);
    }

    #[test]
    #[should_panic(expected = "depth frame length")]
    fn depth_frame_buffer_write_wrong_size_panics() {
        let mut buf = DepthFrameBuffer::new(2, 2);
        buf.write_depth_frame(vec![0.0; 3]); // Expected 4
    }

    #[test]
    fn depth_frame_buffer_default() {
        let buf = DepthFrameBuffer::default();
        assert_eq!(buf.width(), 512);
        assert_eq!(buf.height(), 512);
        assert_eq!(buf.data().len(), 512 * 512);
    }

    #[test]
    fn depth_frame_buffer_clone() {
        let mut buf = DepthFrameBuffer::new(2, 1);
        buf.write_depth_frame(vec![0.3, 0.7]);
        let buf2 = buf.clone();
        assert_eq!(buf2.data(), buf.data());
        assert_eq!(buf2.frame_counter(), buf.frame_counter());
    }

    // -----------------------------------------------------------------------
    // CameraFrameBuffers tests
    // -----------------------------------------------------------------------

    #[test]
    fn camera_frame_buffers_default_is_empty() {
        let bufs = CameraFrameBuffers::default();
        assert!(bufs.is_empty());
        assert_eq!(bufs.len(), 0);
    }

    #[test]
    fn camera_frame_buffers_insert_and_get() {
        let mut bufs = CameraFrameBuffers::default();
        bufs.insert(
            "front".to_string(),
            FrameBuffer::new(4, 4, PixelFormat::Rgba8),
        );
        assert!(bufs.get("front").is_some());
        assert_eq!(bufs.get("front").unwrap().width(), 4);
        assert!(bufs.get("rear").is_none());
    }

    #[test]
    fn camera_frame_buffers_get_mut() {
        let mut bufs = CameraFrameBuffers::default();
        bufs.insert(
            "cam".to_string(),
            FrameBuffer::new(1, 1, PixelFormat::Rgb8),
        );
        let buf = bufs.get_mut("cam").unwrap();
        buf.write_frame(vec![10, 20, 30]);
        assert_eq!(bufs.get("cam").unwrap().data(), &[10, 20, 30]);
    }

    #[test]
    fn camera_frame_buffers_remove() {
        let mut bufs = CameraFrameBuffers::default();
        bufs.insert(
            "cam".to_string(),
            FrameBuffer::new(1, 1, PixelFormat::Rgb8),
        );
        let removed = bufs.remove("cam");
        assert!(removed.is_some());
        assert!(bufs.is_empty());
    }

    #[test]
    fn camera_frame_buffers_multiple_cameras() {
        let mut bufs = CameraFrameBuffers::default();
        bufs.insert(
            "cam_a".to_string(),
            FrameBuffer::new(2, 2, PixelFormat::Rgb8),
        );
        bufs.insert(
            "cam_b".to_string(),
            FrameBuffer::new(4, 4, PixelFormat::Rgba8),
        );
        assert_eq!(bufs.len(), 2);
        assert_eq!(bufs.get("cam_a").unwrap().format(), PixelFormat::Rgb8);
        assert_eq!(bufs.get("cam_b").unwrap().format(), PixelFormat::Rgba8);
    }

    #[test]
    fn camera_frame_buffers_iter() {
        let mut bufs = CameraFrameBuffers::default();
        bufs.insert(
            "x".to_string(),
            FrameBuffer::new(1, 1, PixelFormat::Rgb8),
        );
        bufs.insert(
            "y".to_string(),
            FrameBuffer::new(2, 2, PixelFormat::Rgba8),
        );
        let mut labels: Vec<&str> = bufs.iter().map(|(label, _)| label).collect();
        labels.sort_unstable();
        assert_eq!(labels, vec!["x", "y"]);
    }
}
