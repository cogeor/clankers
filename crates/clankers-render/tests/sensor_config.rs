//! Integration tests for render sensor configuration types.
//!
//! Exercises CameraConfig, FrameBuffer, CameraFrameBuffers, ImageSensor,
//! DepthFrameBuffer, and PixelFormat without requiring a GPU.

use clankers_core::traits::{ObservationSensor, Sensor};
use clankers_render::buffer::{CameraFrameBuffers, DepthFrameBuffer, FrameBuffer};
use clankers_render::config::{CameraConfig, PixelFormat};
use clankers_render::sensor::{DepthSensor, ImageSensor, SegmentationSensor};

// ---------------------------------------------------------------------------
// CameraConfig tests
// ---------------------------------------------------------------------------

#[test]
fn camera_config_defaults() {
    let cam = CameraConfig::new();
    assert!(cam.label.is_empty(), "default label should be empty");
    assert!(
        (cam.fov_y - std::f32::consts::FRAC_PI_3).abs() < f32::EPSILON,
        "default fov_y should be PI/3"
    );
    assert!(
        (cam.near - 0.01).abs() < f32::EPSILON,
        "default near should be 0.01"
    );
    assert!(
        (cam.far - 100.0).abs() < f32::EPSILON,
        "default far should be 100.0"
    );
}

#[test]
fn camera_config_with_label() {
    let cam = CameraConfig::new().with_label("front_camera");
    assert_eq!(cam.label, "front_camera");
    // Other fields should remain at defaults
    assert!(
        (cam.fov_y - std::f32::consts::FRAC_PI_3).abs() < f32::EPSILON,
        "fov_y should remain at default after setting label"
    );
}

// ---------------------------------------------------------------------------
// FrameBuffer tests
// ---------------------------------------------------------------------------

#[test]
fn frame_buffer_write_read() {
    let mut buf = FrameBuffer::new(2, 2, PixelFormat::Rgb8);
    assert_eq!(buf.data().len(), 2 * 2 * 3, "initial size should be w*h*3");
    assert_eq!(buf.frame_counter(), 0);

    let pixels = vec![
        255, 0, 0, // pixel (0,0)
        0, 255, 0, // pixel (1,0)
        0, 0, 255, // pixel (0,1)
        128, 128, 128, // pixel (1,1)
    ];
    buf.write_frame(pixels.clone());

    assert_eq!(buf.data(), pixels.as_slice());
    assert_eq!(buf.frame_counter(), 1);
    assert_eq!(buf.pixel(0, 0), &[255, 0, 0]);
    assert_eq!(buf.pixel(1, 1), &[128, 128, 128]);
}

// ---------------------------------------------------------------------------
// CameraFrameBuffers tests
// ---------------------------------------------------------------------------

#[test]
fn camera_frame_buffers_insert_get() {
    let mut bufs = CameraFrameBuffers::default();
    assert!(bufs.is_empty());

    bufs.insert(
        "front".to_string(),
        FrameBuffer::new(8, 8, PixelFormat::Rgba8),
    );
    assert_eq!(bufs.len(), 1);
    assert!(!bufs.is_empty());

    let front = bufs.get("front");
    assert!(front.is_some(), "should find buffer by label");
    let front = front.unwrap();
    assert_eq!(front.width(), 8);
    assert_eq!(front.height(), 8);
    assert_eq!(front.format(), PixelFormat::Rgba8);

    assert!(bufs.get("rear").is_none(), "missing label should return None");
}

// ---------------------------------------------------------------------------
// ImageSensor tests
// ---------------------------------------------------------------------------

#[test]
fn image_sensor_observation_dim() {
    let sensor = ImageSensor::new("cam", 64, 48, 4);
    assert_eq!(
        sensor.observation_dim(),
        64 * 48 * 4,
        "observation_dim should equal width * height * channels"
    );
    assert_eq!(sensor.name(), "cam");
    assert_eq!(sensor.width(), 64);
    assert_eq!(sensor.height(), 48);
    assert_eq!(sensor.channels(), 4);
}

#[test]
fn image_sensor_rgb_dim() {
    let sensor = ImageSensor::new("rgb_cam", 32, 32, 3);
    assert_eq!(sensor.observation_dim(), 32 * 32 * 3);
}

// ---------------------------------------------------------------------------
// DepthFrameBuffer tests
// ---------------------------------------------------------------------------

#[test]
fn depth_frame_buffer_roundtrip() {
    let mut buf = DepthFrameBuffer::new(3, 2);
    assert_eq!(buf.data().len(), 6);
    assert_eq!(buf.frame_counter(), 0);

    let depths = vec![0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    buf.write_depth_frame(depths.clone());
    assert_eq!(buf.data(), depths.as_slice());
    assert_eq!(buf.frame_counter(), 1);
}

// ---------------------------------------------------------------------------
// DepthSensor tests
// ---------------------------------------------------------------------------

#[test]
fn depth_sensor_observation_dim() {
    let sensor = DepthSensor {
        label: "depth_front".to_string(),
        width: 64,
        height: 48,
        near: 0.1,
        far: 100.0,
    };
    assert_eq!(
        sensor.observation_dim(),
        64 * 48,
        "depth sensor observation_dim should be width * height (one f32 per pixel)"
    );
    assert_eq!(sensor.name(), "depth_front");
}

// ---------------------------------------------------------------------------
// SegmentationSensor tests
// ---------------------------------------------------------------------------

#[test]
fn segmentation_sensor_observation_dim() {
    let sensor = SegmentationSensor {
        label: "seg_cam".to_string(),
        width: 32,
        height: 24,
    };
    assert_eq!(
        sensor.observation_dim(),
        32 * 24 * 3,
        "segmentation sensor observation_dim should be width * height * 3 (RGB)"
    );
    assert_eq!(sensor.name(), "seg_cam");
}

// ---------------------------------------------------------------------------
// PixelFormat tests
// ---------------------------------------------------------------------------

#[test]
fn pixel_format_bytes() {
    assert_eq!(PixelFormat::Rgb8.bytes_per_pixel(), 3);
    assert_eq!(PixelFormat::Rgba8.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::DepthF32.bytes_per_pixel(), 4);
}

#[test]
fn pixel_format_channels() {
    assert_eq!(PixelFormat::Rgb8.channels(), 3);
    assert_eq!(PixelFormat::Rgba8.channels(), 4);
    assert_eq!(PixelFormat::DepthF32.channels(), 1);
}

#[test]
fn pixel_format_default_is_rgb8() {
    assert_eq!(PixelFormat::default(), PixelFormat::Rgb8);
}
