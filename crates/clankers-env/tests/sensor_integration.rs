//! Integration tests for the sensor pipeline.
//!
//! Exercises LidarSensor trait implementations, ObservationBuffer roundtrips,
//! and LidarConfig defaults. These tests run headless (no GPU required).

use bevy::prelude::*;
use clankers_core::physics::LidarConfig;
use clankers_core::traits::{ObservationSensor, Sensor};
use clankers_env::buffer::ObservationBuffer;
use clankers_env::sensors::{JointStateSensor, LidarSensor};

// ---------------------------------------------------------------------------
// LidarSensor trait tests
// ---------------------------------------------------------------------------

#[test]
fn lidar_observation_dim() {
    let cfg = LidarConfig {
        num_rays: 16,
        num_channels: 4,
        ..LidarConfig::default()
    };
    let sensor = LidarSensor::new(cfg, Vec3::ZERO, Quat::IDENTITY);
    assert_eq!(
        sensor.observation_dim(),
        16 * 4,
        "observation_dim should equal num_rays * num_channels"
    );
}

#[test]
fn lidar_sensor_name() {
    let sensor = LidarSensor::new(LidarConfig::default(), Vec3::ZERO, Quat::IDENTITY);
    assert_eq!(sensor.name(), "LidarSensor");
}

#[test]
fn lidar_sensor_no_rapier_context_returns_nan() {
    // Without a RapierContext resource in the world, every ray should be NaN.
    let mut world = World::new();
    let cfg = LidarConfig {
        num_rays: 8,
        num_channels: 2,
        ..LidarConfig::default()
    };
    let sensor = LidarSensor::new(cfg, Vec3::ZERO, Quat::IDENTITY);
    let obs = sensor.read(&mut world);
    assert_eq!(obs.len(), 16);
    for v in obs.as_slice() {
        assert!(v.is_nan(), "expected NaN without RapierContext, got {v}");
    }
}

#[test]
fn lidar_single_ray_single_channel_dim() {
    let cfg = LidarConfig {
        num_rays: 1,
        num_channels: 1,
        ..LidarConfig::default()
    };
    let sensor = LidarSensor::new(cfg, Vec3::ZERO, Quat::IDENTITY);
    assert_eq!(sensor.observation_dim(), 1);
}

// ---------------------------------------------------------------------------
// ObservationBuffer roundtrip tests
// ---------------------------------------------------------------------------

#[test]
fn observation_buffer_register_and_read() {
    let mut buf = ObservationBuffer::new();
    let sensor = JointStateSensor::new(3);
    let idx = buf.register(sensor.name(), sensor.observation_dim());

    assert_eq!(buf.dim(), 6, "3 joints * 2 values each = 6");
    assert_eq!(buf.sensor_count(), 1);

    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    buf.write(idx, &values);
    assert_eq!(buf.read(idx), values.as_slice());
}

#[test]
fn observation_buffer_multiple_slots() {
    let mut buf = ObservationBuffer::new();

    // Register a JointStateSensor (3 joints => 6 values)
    let joint_sensor = JointStateSensor::new(3);
    let joint_idx = buf.register(joint_sensor.name(), joint_sensor.observation_dim());

    // Register a LidarSensor (4 rays, 2 channels => 8 values)
    let lidar_cfg = LidarConfig {
        num_rays: 4,
        num_channels: 2,
        ..LidarConfig::default()
    };
    let lidar_sensor = LidarSensor::new(lidar_cfg, Vec3::ZERO, Quat::IDENTITY);
    let lidar_idx = buf.register(lidar_sensor.name(), lidar_sensor.observation_dim());

    assert_eq!(buf.sensor_count(), 2);
    assert_eq!(buf.dim(), 6 + 8, "total dim should be sum of all sensor dims");

    // Write to each slot independently
    let joint_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let lidar_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    buf.write(joint_idx, &joint_values);
    buf.write(lidar_idx, &lidar_values);

    // Read back each slot
    assert_eq!(buf.read(joint_idx), &joint_values);
    assert_eq!(buf.read(lidar_idx), &lidar_values);

    // Verify slot metadata
    let joint_slot = buf.slot(joint_idx).unwrap();
    assert_eq!(joint_slot.name, "JointStateSensor");
    assert_eq!(joint_slot.dim, 6);
    assert_eq!(joint_slot.offset, 0);

    let lidar_slot = buf.slot(lidar_idx).unwrap();
    assert_eq!(lidar_slot.name, "LidarSensor");
    assert_eq!(lidar_slot.dim, 8);
    assert_eq!(lidar_slot.offset, 6);
}

// ---------------------------------------------------------------------------
// LidarConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn lidar_config_default() {
    let cfg = LidarConfig::default();
    assert_eq!(cfg.num_rays, 64, "default num_rays should be 64");
    assert_eq!(cfg.num_channels, 1, "default num_channels should be 1");
    assert!(
        (cfg.max_range - 10.0).abs() < f32::EPSILON,
        "default max_range should be 10.0"
    );
    assert!(
        (cfg.half_fov - std::f32::consts::PI).abs() < f32::EPSILON,
        "default half_fov should be PI"
    );
    assert!(
        (cfg.vertical_half_fov - 0.0).abs() < f32::EPSILON,
        "default vertical_half_fov should be 0.0 (flat 2-D scan)"
    );
    assert_eq!(
        cfg.origin_offset,
        Vec3::ZERO,
        "default origin_offset should be ZERO"
    );
}

// ---------------------------------------------------------------------------
// ObservationBuffer -> Observation conversion
// ---------------------------------------------------------------------------

#[test]
fn observation_buffer_as_observation_roundtrip() {
    let mut buf = ObservationBuffer::new();
    let idx_a = buf.register("sensor_a", 3);
    let idx_b = buf.register("sensor_b", 2);
    buf.write(idx_a, &[10.0, 20.0, 30.0]);
    buf.write(idx_b, &[40.0, 50.0]);

    let obs = buf.as_observation();
    assert_eq!(obs.len(), 5);
    assert!((obs[0] - 10.0).abs() < f32::EPSILON);
    assert!((obs[1] - 20.0).abs() < f32::EPSILON);
    assert!((obs[2] - 30.0).abs() < f32::EPSILON);
    assert!((obs[3] - 40.0).abs() < f32::EPSILON);
    assert!((obs[4] - 50.0).abs() < f32::EPSILON);
}
