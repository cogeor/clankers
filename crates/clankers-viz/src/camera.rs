//! Camera setup for the visualization scene.
//!
//! Uses `bevy_panorbit_camera` for orbit camera controls.

use bevy::prelude::*;
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
