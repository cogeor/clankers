//! Custom depth material for rendering linear depth as grayscale.
//!
//! [`DepthMaterial`] renders each fragment's linear view-space depth
//! normalised by `max_depth` into an RGB grayscale colour. This avoids
//! the need for a custom render-graph node to extract the depth prepass
//! texture: instead, a second set of shadow entities on a dedicated
//! render layer carries this material, and a regular RGB camera captures
//! the result via the standard readback pipeline.

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

use clankers_core::types::SegmentationClass;

/// Render layer used by depth shadow entities and depth cameras.
pub const DEPTH_RENDER_LAYER: usize = 2;

/// Custom material that outputs linear depth as grayscale.
///
/// `max_depth` defines the clamp distance: fragments at this distance
/// or further output white (1.0); fragments at the camera output black
/// (0.0).
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct DepthMaterial {
    #[uniform(0)]
    pub uniforms: DepthMaterialUniform,
}

/// GPU-side uniform for the depth material.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct DepthMaterialUniform {
    pub max_depth: f32,
}

impl DepthMaterial {
    /// Create a depth material with the given maximum depth in metres.
    #[must_use]
    pub fn new(max_depth: f32) -> Self {
        Self {
            uniforms: DepthMaterialUniform { max_depth },
        }
    }
}

impl Material for DepthMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path("embedded://clankers_render/depth_material.wgsl".into())
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

/// Bevy plugin that registers [`DepthMaterial`] and its shader.
pub struct DepthMaterialPlugin;

impl Plugin for DepthMaterialPlugin {
    fn build(&self, app: &mut App) {
        // Register the WGSL shader as an embedded asset so the asset server
        // can find it via `embedded://clankers_render/depth_material.wgsl`.
        bevy::asset::embedded_asset!(app, "depth_material.wgsl");

        app.add_plugins(MaterialPlugin::<DepthMaterial>::default());

        // Shadow entity systems for depth rendering.
        app.add_systems(
            Update,
            (
                spawn_depth_shadows,
                sync_depth_shadows.after(spawn_depth_shadows),
            ),
        );
    }
}

// ---------------------------------------------------------------------------
// Depth shadow entities
// ---------------------------------------------------------------------------

/// Resource holding the shared [`DepthMaterial`] handle used by all shadows.
#[derive(Resource, Debug, Clone)]
pub struct DepthMaterialHandle(pub Handle<DepthMaterial>);

/// Marker: this entity is a depth shadow clone of another entity.
#[derive(Component, Debug)]
pub struct DepthShadow(pub Entity);

/// Marker: a depth shadow has been spawned for this entity.
#[derive(Component, Debug)]
struct DepthShadowSpawned;

/// Spawn depth shadow entities on [`DEPTH_RENDER_LAYER`] for each entity
/// with [`SegmentationClass`] + [`Mesh3d`].
#[allow(clippy::type_complexity)]
fn spawn_depth_shadows(
    mut commands: Commands,
    query: Query<
        (Entity, &Mesh3d, &Transform, Option<&ChildOf>),
        (
            With<SegmentationClass>,
            Without<DepthShadowSpawned>,
            Without<DepthShadow>,
        ),
    >,
    depth_mat: Option<Res<DepthMaterialHandle>>,
) {
    let Some(depth_mat) = depth_mat else {
        return;
    };

    for (entity, mesh, tf, maybe_parent) in &query {
        let shadow_bundle = (
            DepthShadow(entity),
            mesh.clone(),
            MeshMaterial3d(depth_mat.0.clone()),
            *tf,
            RenderLayers::layer(DEPTH_RENDER_LAYER),
            Visibility::default(),
        );

        if let Some(child_of) = maybe_parent {
            commands.entity(child_of.parent()).with_children(|p| {
                p.spawn(shadow_bundle);
            });
        } else {
            commands.spawn(shadow_bundle);
        }

        commands.entity(entity).insert(DepthShadowSpawned);
    }
}

/// Copy transforms from source entities to their parentless depth shadows.
fn sync_depth_shadows(
    sources: Query<&GlobalTransform, Without<DepthShadow>>,
    mut shadows: Query<(&DepthShadow, &mut Transform), Without<ChildOf>>,
) {
    for (shadow, mut tf) in &mut shadows {
        if let Ok(source_gt) = sources.get(shadow.0) {
            let source_tf = source_gt.compute_transform();
            tf.translation = source_tf.translation;
            tf.rotation = source_tf.rotation;
            tf.scale = source_tf.scale;
        }
    }
}
