//! Frame writer system for exporting PNG files per camera per frame.

use std::path::PathBuf;

use bevy::prelude::*;
use image::{GrayImage, ImageBuffer, Rgb, RgbImage};

use crate::buffer::{CameraFrameBuffers, DepthFrameBuffers};
use crate::segmentation::{SegmentationFrameBuffers, SegmentationPalette};

use super::config::CosmosLogConfig;

/// Segmentation class RGB values that map to "transform" (white) in binary mask.
/// Computed once from config + palette at init.
#[derive(Resource, Debug, Clone)]
pub struct SegTransformColors(pub Vec<[u8; 3]>);

/// Runtime state for the frame writer.
#[derive(Resource, Debug)]
pub struct CosmosWriterState {
    /// Output directory for this run (e.g. `data/20260309-142301-arm-pick/`).
    pub run_dir: PathBuf,
    /// Current frame index.
    pub frame_index: u32,
    /// Per-camera subdirectory (pre-created).
    pub camera_dirs: Vec<(String, PathBuf)>,
}

/// Startup system: create the run directory and camera subdirectories.
#[allow(clippy::needless_pass_by_value)]
pub fn create_run_directory(mut commands: Commands, config: Res<CosmosLogConfig>) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Format as YYYYMMDD-HHMMSS (approximate from unix timestamp).
    let secs_per_day = 86400_u64;
    let days = timestamp / secs_per_day;
    // Simple date calculation (good enough for directory naming).
    let (year, month, day) = days_to_ymd(days);
    let time_of_day = timestamp % secs_per_day;
    let hour = time_of_day / 3600;
    let minute = (time_of_day % 3600) / 60;
    let second = time_of_day % 60;

    let run_name = config.run_name.clone().unwrap_or_else(|| "run".into());
    let dir_name = format!(
        "{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}-{run_name}"
    );
    let run_dir = config.output_root.join(&dir_name);

    let mut camera_dirs = Vec::new();
    for spec in &config.cameras {
        let cam_dir = run_dir.join(&spec.label);
        std::fs::create_dir_all(&cam_dir).unwrap_or_else(|e| {
            eprintln!("CosmosLog: failed to create directory {}: {e}", cam_dir.display());
        });
        camera_dirs.push((spec.label.clone(), cam_dir));
    }

    println!("CosmosLog: output directory → {}", run_dir.display());

    commands.insert_resource(CosmosWriterState {
        run_dir,
        frame_index: 0,
        camera_dirs,
    });
}

/// Compute segmentation transform colors from config + palette at startup.
#[allow(clippy::needless_pass_by_value)]
pub fn init_seg_transform_colors(
    mut commands: Commands,
    config: Res<CosmosLogConfig>,
    palette: Res<SegmentationPalette>,
) {
    // Map class names to class IDs and then to palette RGB u8 values.
    let class_name_to_id: std::collections::HashMap<&str, u32> = [
        ("ground", 0),
        ("wall", 1),
        ("robot", 2),
        ("obstacle", 3),
        ("table", 4),
    ]
    .into_iter()
    .collect();

    let mut transform_colors = Vec::new();
    for name in &config.seg_transform_classes {
        if let Some(&id) = class_name_to_id.get(name.as_str()) {
            if let Some(&[r, g, b]) = palette.colors.get(&id) {
                transform_colors.push([
                    (r * 255.0) as u8,
                    (g * 255.0) as u8,
                    (b * 255.0) as u8,
                ]);
            }
        }
    }
    commands.insert_resource(SegTransformColors(transform_colors));
}

/// PostUpdate system: write RGB, depth, and segmentation PNGs for each camera.
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
pub fn write_cosmos_frames(
    cam_bufs: Res<CameraFrameBuffers>,
    depth_bufs: Res<DepthFrameBuffers>,
    seg_bufs: Res<SegmentationFrameBuffers>,
    config: Res<CosmosLogConfig>,
    seg_colors: Res<SegTransformColors>,
    mut state: ResMut<CosmosWriterState>,
) {
    let frame_idx = state.frame_index;

    for (label, dir) in &state.camera_dirs {
        let rgb_key = format!("cosmos_{label}");

        // --- RGB ---
        if let Some(buf) = cam_bufs.get(&rgb_key) {
            let w = buf.width();
            let h = buf.height();
            let data = buf.data();
            // CameraFrameBuffers stores RGBA8; convert to RGB.
            let rgb: Vec<u8> = data
                .chunks_exact(4)
                .flat_map(|px| [px[0], px[1], px[2]])
                .collect();
            if let Some(img) = RgbImage::from_raw(w, h, rgb) {
                let path = dir.join(format!("rgb_{frame_idx:05}.png"));
                if let Err(e) = img.save(&path) {
                    eprintln!("CosmosLog: failed to save RGB frame: {e}");
                }
            }
        }

        // --- Depth ---
        if let Some(buf) = depth_bufs.get(label) {
            let w = buf.width();
            let h = buf.height();
            let max_depth = config.depth_max_m;
            // Normalize f32 depth to u8 grayscale.
            let gray: Vec<u8> = buf
                .data()
                .iter()
                .map(|&d| {
                    let normalized = (d / max_depth).clamp(0.0, 1.0);
                    (normalized * 255.0) as u8
                })
                .collect();
            if let Some(img) = GrayImage::from_raw(w, h, gray) {
                let path = dir.join(format!("depth_{frame_idx:05}.png"));
                if let Err(e) = img.save(&path) {
                    eprintln!("CosmosLog: failed to save depth frame: {e}");
                }
            }
        }

        // --- Segmentation → binary mask ---
        if let Some(buf) = seg_bufs.get(label) {
            let w = buf.width();
            let h = buf.height();
            let data = buf.data();
            // Remap: if pixel matches a transform class → white, else black.
            let mask: Vec<u8> = data
                .chunks_exact(3)
                .flat_map(|px| {
                    let is_transform = seg_colors.0.iter().any(|c| c == px);
                    let val = if is_transform { 255_u8 } else { 0_u8 };
                    [val, val, val]
                })
                .collect();
            if let Some(img) = ImageBuffer::<Rgb<u8>, _>::from_raw(w, h, mask) {
                let path = dir.join(format!("seg_{frame_idx:05}.png"));
                if let Err(e) = img.save(&path) {
                    eprintln!("CosmosLog: failed to save seg frame: {e}");
                }
            }
        }
    }

    state.frame_index += 1;
}

// ---------------------------------------------------------------------------
// Date helpers (avoid chrono dependency)
// ---------------------------------------------------------------------------

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
