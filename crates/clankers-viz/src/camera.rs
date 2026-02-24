//! Camera setup for the visualization scene.
//!
//! Uses `bevy_panorbit_camera` for orbit camera controls.
//! The camera is automatically disabled when egui wants pointer input
//! (e.g. dragging sliders) to prevent conflicts.

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
        .map_or(false, |ctx| {
            ctx.is_pointer_over_area() || ctx.wants_pointer_input()
        });
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
    // Ground plane.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(10.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.38, 0.35),
            ..default()
        })),
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
