//! Camera spawning and synchronization for Cosmos logging.

use bevy::prelude::*;

use bevy::camera::visibility::RenderLayers;

use crate::buffer::CameraFrameBuffers;
use crate::camera::spawn_camera_sensor;
use crate::config::CameraConfig;
use crate::depth_material::DEPTH_RENDER_LAYER;
use crate::segmentation::{
    SegmentationCameraLabel, SegmentationFrameBuffer, SegmentationFrameBuffers,
    spawn_segmentation_camera_sensor,
};

use super::config::{CameraPlacement, CosmosLogConfig};

/// Marker component for all cosmos log camera entities (RGB, depth, or seg).
#[derive(Component, Debug, Clone)]
pub struct CosmosLogCamera {
    /// The camera spec label this entity belongs to.
    pub label: String,
    /// Which modality this camera captures.
    pub modality: CosmosModality,
}

/// Which rendering modality a cosmos log camera captures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CosmosModality {
    /// Standard RGB color.
    Rgb,
    /// Depth (f32 → grayscale u8).
    Depth,
    /// Segmentation (palette → binary mask).
    Seg,
}

/// Startup system: spawn one camera triplet (RGB + depth + seg) per [`CameraSpec`].
#[allow(clippy::needless_pass_by_value)]
pub fn spawn_cosmos_cameras(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut cam_bufs: ResMut<CameraFrameBuffers>,
    mut seg_bufs: ResMut<SegmentationFrameBuffers>,
    config: Res<CosmosLogConfig>,
) {
    for spec in &config.cameras {
        let (w, h) = spec.resolution;
        let transform = placement_to_transform(&spec.placement);
        let fov_rad = spec.fov_deg.to_radians();

        // --- RGB camera ---
        let rgb_label = format!("cosmos_{}", spec.label);
        let cam_config = CameraConfig {
            label: rgb_label.clone(),
            ..Default::default()
        };
        let (rgb_entity, _) =
            spawn_camera_sensor(&mut commands, &mut images, &mut cam_bufs, cam_config, w, h);
        commands.entity(rgb_entity).insert((
            transform,
            Projection::Perspective(PerspectiveProjection {
                fov: fov_rad,
                ..Default::default()
            }),
            CosmosLogCamera {
                label: spec.label.clone(),
                modality: CosmosModality::Rgb,
            },
        ));

        // --- Depth camera (RGB camera on depth layer, depth material renders grayscale) ---
        let depth_label = format!("cosmos_{}_depth", spec.label);
        let depth_cam_config = CameraConfig {
            label: depth_label.clone(),
            ..Default::default()
        };
        let (depth_entity, _) = spawn_camera_sensor(
            &mut commands,
            &mut images,
            &mut cam_bufs,
            depth_cam_config,
            w,
            h,
        );
        commands.entity(depth_entity).insert((
            transform,
            Projection::Perspective(PerspectiveProjection {
                fov: fov_rad,
                ..Default::default()
            }),
            bevy::core_pipeline::tonemapping::Tonemapping::None,
            RenderLayers::layer(DEPTH_RENDER_LAYER),
            CosmosLogCamera {
                label: spec.label.clone(),
                modality: CosmosModality::Depth,
            },
        ));

        // --- Segmentation camera ---
        seg_bufs.insert(spec.label.clone(), SegmentationFrameBuffer::new(w, h));
        let (seg_entity, _) = spawn_segmentation_camera_sensor(&mut commands, &mut images, w, h);
        commands.entity(seg_entity).insert((
            transform,
            Projection::Perspective(PerspectiveProjection {
                fov: fov_rad,
                ..Default::default()
            }),
            SegmentationCameraLabel(spec.label.clone()),
            CosmosLogCamera {
                label: spec.label.clone(),
                modality: CosmosModality::Seg,
            },
        ));

        println!(
            "CosmosLog: spawned camera triplet '{}' ({}×{}, {:.0}° FOV)",
            spec.label, w, h, spec.fov_deg
        );
    }
}

/// Convert a [`CameraPlacement`] to an initial [`Transform`].
fn placement_to_transform(placement: &CameraPlacement) -> Transform {
    match placement {
        CameraPlacement::Fixed { position, target } => {
            Transform::from_translation(*position).looking_at(*target, Vec3::Y)
        }
        CameraPlacement::FollowLink {
            offset,
            orientation,
            ..
        } => Transform {
            translation: *offset,
            rotation: *orientation,
            ..Default::default()
        },
    }
}

/// Update system: sync [`FollowLink`][CameraPlacement::FollowLink] cameras
/// to their tracked link entity.
///
/// Queries entities with a `LinkVisual` name matching the spec's `link_name`
/// and updates all cosmos log cameras that belong to that spec.
///
/// `LinkVisual` is expected to be a newtype `(pub String)` component provided
/// by the example's visual setup. Since `clankers-render` doesn't depend on
/// the examples crate, this system takes a generic approach: it looks for
/// entities with a `Name` component matching the link name.
#[allow(clippy::needless_pass_by_value)]
pub fn sync_cosmos_follow_cameras(
    config: Res<CosmosLogConfig>,
    named: Query<(&Name, &GlobalTransform)>,
    mut cameras: Query<(&CosmosLogCamera, &mut Transform)>,
) {
    for spec in &config.cameras {
        let CameraPlacement::FollowLink {
            link_name,
            offset,
            orientation,
        } = &spec.placement
        else {
            continue;
        };

        // Find the link entity by Name.
        let Some((_, link_gt)) = named.iter().find(|(n, _)| n.as_str() == link_name) else {
            continue;
        };

        let link_tf = link_gt.compute_transform();
        let cam_pos = link_tf.translation + link_tf.rotation * *offset;
        let cam_rot = link_tf.rotation * *orientation;

        for (cosmos_cam, mut tf) in &mut cameras {
            if cosmos_cam.label == spec.label {
                tf.translation = cam_pos;
                tf.rotation = cam_rot;
            }
        }
    }
}
