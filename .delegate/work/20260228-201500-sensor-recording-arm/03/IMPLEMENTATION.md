# Implementation Log

## Task 1: QueryPipeline access in RapierContext

Completed: 2026-02-28T21:24:12Z

### Changes

- `crates/clankers-physics/src/rapier/context.rs`: Added `cast_ray_bevy()` helper method that builds a temporary `QueryPipeline` view from the existing `broad_phase` (which is a `DefaultBroadPhase = BroadPhaseBvh` in rapier3d 0.32). Added `QueryFilter` and `Ray` to the rapier3d imports.

### Verification

- [x] Compiles: `cargo build -p clankers-physics` succeeds
- [x] Method signature: `pub fn cast_ray_bevy(&self, origin: Vec3, direction: Vec3, max_range: f32, solid: bool) -> Option<f32>`

### Notes

**API deviation from plan:** The plan said to add `pub query_pipeline: rapier3d::prelude::QueryPipeline` as a stored field and call `query_pipeline.update(...)` after each substep. However, in **rapier3d 0.32**, `QueryPipeline` is **not** a standalone stored struct — it is a temporary borrow view created from `BroadPhaseBvh::as_query_pipeline(dispatcher, bodies, colliders, filter)`. It has lifetime parameters and cannot be stored in a struct.

The BVH (broad phase) is already updated automatically by `physics_pipeline.step()`, so no separate update call is needed after substeps. The correct pattern for 0.32 is to create the `QueryPipeline` on-the-fly when needed via the existing `broad_phase` field. This is equivalent in functionality and is consistent with the rapier3d 0.32 documentation examples.

The helper method `cast_ray_bevy` accepts Bevy `Vec3` types because parry3d (rapier's math layer) uses `glam::Vec3` as its `Vector` type when compiling for f32/3D, so no conversion is needed — they are the same type.

---

## Task 2: Update QueryPipeline in rapier_step_system

Completed: 2026-02-28T21:24:12Z

### Changes

- `crates/clankers-physics/src/rapier/systems.rs`: No changes made.

### Verification

- [x] File unchanged: `systems.rs` does not require modification

### Notes

**No changes required.** The plan specified calling `context.query_pipeline.update(...)` after substeps. Since the plan's `query_pipeline` field concept does not apply to rapier3d 0.32 (see Task 1 notes), and since `physics_pipeline.step()` already keeps the `BroadPhaseBvh` up-to-date, no explicit pipeline update is needed in `rapier_step_system`.

---

## Task 3: LidarConfig component

Completed: 2026-02-28T21:24:12Z

### Changes

- `crates/clankers-core/src/physics.rs`: Added `LidarConfig` component with fields `num_rays`, `num_channels`, `max_range`, `half_fov`, `vertical_half_fov`, `origin_offset`. Added `Default` impl (64 rays, 1 channel, 10.0m range, PI half-fov, 0.0 vertical, zero offset). Added two unit tests (`lidar_config_default_values`, `lidar_config_custom`). Added `LidarConfig` to `Send + Sync` test.
- `crates/clankers-core/src/lib.rs`: Added `LidarConfig` to the prelude physics exports.

### Verification

- [x] `cargo test -p clankers-core` passes: 206 tests ok
- [x] `physics::tests::lidar_config_default_values` passes
- [x] `physics::tests::lidar_config_custom` passes
- [x] `physics::tests::physics_types_are_send_sync` passes (includes LidarConfig)

---

## Task 4: LidarSensor

Completed: 2026-02-28T21:24:12Z

### Changes

- `crates/clankers-env/Cargo.toml`: Added `clankers-physics.workspace = true` dependency so `LidarSensor` can access `RapierContext`.
- `crates/clankers-env/src/sensors.rs`:
  - Added `LidarConfig` to the `clankers_core::physics` import
  - Added `use clankers_physics::rapier::RapierContext;` import
  - Implemented `LidarSensor` struct with `config: LidarConfig`, `sensor_origin: Vec3`, `sensor_rotation: Quat`
  - Implemented `Sensor` trait: `read()` fetches `RapierContext` from world, computes azimuth/elevation angles per ray/channel, calls `ctx.cast_ray_bevy()`, returns flat f32 `Observation` (NaN for no-hit or when `RapierContext` is absent)
  - Implemented `ObservationSensor` trait: `observation_dim()` returns `num_rays * num_channels`
  - Added unit tests: `lidar_config_default`, `lidar_observation_dim`, `lidar_sensor_name`, `lidar_sensor_no_rapier_context_returns_nan`, `lidar_sensor_single_ray_dim`
  - Added `LidarSensor` to `sensor_types_are_send_sync` test
- `crates/clankers-env/src/lib.rs`: Added `LidarSensor` to the prelude sensors exports.

### Verification

- [x] `cargo test -p clankers-env` passes: 116 tests ok
- [x] `sensors::tests::lidar_config_default` passes
- [x] `sensors::tests::lidar_observation_dim` passes
- [x] `sensors::tests::lidar_sensor_name` passes
- [x] `sensors::tests::lidar_sensor_no_rapier_context_returns_nan` passes
- [x] `sensors::tests::lidar_sensor_single_ray_dim` passes
- [x] `sensors::tests::sensor_types_are_send_sync` passes (includes LidarSensor)
- [x] Full workspace build: `cargo build -j 24` succeeds (all crates compile)

### Notes

Ray direction formula: local direction `(sin(az)*cos(el), sin(el), cos(el)*cos(az))` is rotated into world space by `sensor_rotation`. This gives +Z forward with +Y up convention where azimuth sweeps the XZ plane and elevation tilts around the Y axis.

When `RapierContext` is not registered as a Bevy resource (e.g., in unit tests without rapier), the sensor returns an all-NaN observation rather than panicking.

---
