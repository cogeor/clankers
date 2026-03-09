//! Metadata JSON generation for Cosmos log runs.

use bevy::prelude::*;
use serde_json::{Map, Value, json};

use super::config::{CameraPlacement, CosmosLogConfig};
use super::writer::CosmosWriterState;

/// System that runs in [`Last`] and writes `metadata.json` once when
/// [`AppExit`] has been sent (i.e., the app is shutting down).
#[allow(clippy::needless_pass_by_value)]
pub fn write_cosmos_metadata_on_exit(
    exit_events: MessageReader<AppExit>,
    config: Res<CosmosLogConfig>,
    state: Res<CosmosWriterState>,
    mut written: Local<bool>,
) {
    if *written || exit_events.is_empty() {
        return;
    }
    *written = true;
    write_cosmos_metadata(&config, &state);
}

/// Write `metadata.json` to the run directory.
///
/// Can also be called directly from example code after `app.run()` returns,
/// but the preferred path is via the `Last` schedule system above.
pub fn write_cosmos_metadata(config: &CosmosLogConfig, state: &CosmosWriterState) {
    let frame_count = state.frame_index;
    let fps = config.fps;
    let duration_s = if fps > 0 {
        frame_count as f64 / fps as f64
    } else {
        0.0
    };

    let mut cameras_map = Map::new();
    for spec in &config.cameras {
        let (w, h) = spec.resolution;
        let mut cam_obj = Map::new();
        cam_obj.insert("resolution".into(), json!([w, h]));
        cam_obj.insert("fov_deg".into(), json!(spec.fov_deg));

        match &spec.placement {
            CameraPlacement::Fixed { position, target } => {
                cam_obj.insert("kind".into(), json!("fixed"));
                cam_obj.insert(
                    "position".into(),
                    json!([position.x, position.y, position.z]),
                );
                cam_obj.insert("target".into(), json!([target.x, target.y, target.z]));
            }
            CameraPlacement::FollowLink { link_name, .. } => {
                cam_obj.insert("kind".into(), json!("follow_link"));
                cam_obj.insert("follow_link".into(), json!(link_name));
            }
        }

        cameras_map.insert(spec.label.clone(), Value::Object(cam_obj));
    }

    let metadata = json!({
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration_s,
        "depth_max_m": config.depth_max_m,
        "seg_transform": config.seg_transform_classes,
        "cameras": cameras_map,
    });

    let path = state.run_dir.join("metadata.json");
    match serde_json::to_string_pretty(&metadata) {
        Ok(json_str) => {
            if let Err(e) = std::fs::write(&path, json_str) {
                eprintln!("CosmosLog: failed to write metadata: {e}");
            } else {
                println!(
                    "CosmosLog: metadata → {} ({} frames, {:.1}s)",
                    path.display(),
                    frame_count,
                    duration_s
                );
            }
        }
        Err(e) => eprintln!("CosmosLog: failed to serialize metadata: {e}"),
    }
}
