//! Camera setup for the visualization scene.
//!
//! Uses `bevy_panorbit_camera` for orbit camera controls.
//! The camera is automatically disabled when egui wants pointer input
//! (e.g. dragging sliders) to prevent conflicts.
//!
//! ## Observation camera
//!
//! [`ObservationCamera`] provides a second viewport camera that renders in a
//! corner of the screen, independent of the main orbit camera. Use
//! [`ObsCameraConfig`] resource to configure its position, and
//! [`sync_obs_camera_viewport`] to keep the viewport pinned to a screen corner.

use bevy::camera::{ClearColorConfig, Viewport};
use bevy::prelude::*;
use bevy_egui::EguiContexts;
use bevy_panorbit_camera::PanOrbitCamera;

/// Spawn the default orbit camera looking at the origin.
pub fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        // Camera looking at origin from a reasonable starting position.
        Transform::from_xyz(3.0, 2.5, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera {
            focus: Vec3::ZERO,
            radius: Some(5.0),
            ..default()
        },
        Camera3d::default(),
    ));
}

/// Disable orbit camera when egui wants pointer input (slider drags, clicks).
///
/// Checks both `is_pointer_over_area()` (pointer hovering over egui) and
/// `wants_pointer_input()` (egui actively consuming pointer events, e.g.
/// during a slider drag even if the pointer drifts outside the panel).
pub fn egui_camera_gate(mut contexts: EguiContexts, mut cameras: Query<&mut PanOrbitCamera>) {
    let egui_wants_pointer = contexts
        .ctx_mut()
        .is_ok_and(|ctx| ctx.is_pointer_over_area() || ctx.wants_pointer_input());
    for mut cam in &mut cameras {
        cam.enabled = !egui_wants_pointer;
    }
}

/// Spawn a ground plane and basic lighting.
pub fn spawn_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground plane (lowered slightly to avoid z-fighting with table/base).
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(10.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.38, 0.35),
            ..default()
        })),
        Transform::from_xyz(0.0, -0.01, 0.0),
    ));

    // Directional light (sun).
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.4, 0.0)),
    ));

    // Ambient light.
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
        ..default()
    });
}

// ---------------------------------------------------------------------------
// Observation camera: independent viewport overlay
// ---------------------------------------------------------------------------

/// Marker for an observation camera that renders in a viewport overlay.
///
/// This camera is independent of the main orbit camera and can be positioned
/// freely (e.g. end-effector camera, side-view observation camera).
#[derive(Component)]
pub struct ObservationCamera;

/// Which corner of the window to pin the observation viewport to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewportCorner {
    #[default]
    TopRight,
    TopLeft,
    BottomRight,
    BottomLeft,
}

/// Configuration for the observation camera viewport.
#[derive(Resource, Clone)]
pub struct ObsCameraConfig {
    /// Fraction of window width for the viewport (0.25 = 1/4 width).
    pub width_fraction: f32,
    /// Aspect ratio (width/height). Default 4:3.
    pub aspect: f32,
    /// Corner to pin the viewport to.
    pub corner: ViewportCorner,
    /// Field of view in radians.
    pub fov: f32,
}

impl Default for ObsCameraConfig {
    fn default() -> Self {
        Self {
            width_fraction: 0.25,
            aspect: 4.0 / 3.0,
            corner: ViewportCorner::TopRight,
            fov: 70_f32.to_radians(),
        }
    }
}

/// Spawn an observation camera with the specified transform.
///
/// Call this as a startup system or use it directly in your binary's setup.
/// The camera renders on top of the main camera (order = 1) with its own
/// clear color, and its viewport is kept pinned by [`sync_obs_camera_viewport`].
#[allow(clippy::needless_pass_by_value)]
pub fn spawn_obs_camera(mut commands: Commands, config: Res<ObsCameraConfig>) {
    commands.spawn((
        ObservationCamera,
        Camera3d::default(),
        Camera {
            order: 1,
            clear_color: ClearColorConfig::Custom(Color::srgb(0.05, 0.05, 0.1)),
            ..default()
        },
        Projection::Perspective(PerspectiveProjection {
            fov: config.fov,
            ..default()
        }),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));
}

/// Keep the observation camera viewport pinned to the configured corner.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value
)]
pub fn sync_obs_camera_viewport(
    windows: Query<&Window>,
    config: Res<ObsCameraConfig>,
    mut obs_cam: Query<&mut Camera, With<ObservationCamera>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let w = window.physical_width();
    let h = window.physical_height();
    let vp_w = (w as f32 * config.width_fraction) as u32;
    let vp_h = (vp_w as f32 / config.aspect) as u32;

    let pos = match config.corner {
        ViewportCorner::TopRight => UVec2::new(w.saturating_sub(vp_w), 0),
        ViewportCorner::TopLeft => UVec2::ZERO,
        ViewportCorner::BottomRight => {
            UVec2::new(w.saturating_sub(vp_w), h.saturating_sub(vp_h))
        }
        ViewportCorner::BottomLeft => UVec2::new(0, h.saturating_sub(vp_h)),
    };

    for mut cam in &mut obs_cam {
        cam.viewport = Some(Viewport {
            physical_position: pos,
            physical_size: UVec2::new(vp_w, vp_h),
            ..default()
        });
    }
}
