//! Frame buffer for storing captured pixel data.
//!
//! [`FrameBuffer`] is a Bevy resource that holds a single frame of pixel data.
//! In headless mode it stores a blank frame; when rendering is active, the
//! capture system writes pixels into this buffer each frame.

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
        assert_eq!(buf.width(), 256);
        assert_eq!(buf.height(), 256);
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
}
