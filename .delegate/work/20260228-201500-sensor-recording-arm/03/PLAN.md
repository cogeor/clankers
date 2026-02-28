# Loop 03: LidarSensor — QueryPipeline, LidarConfig, LidarSensor

## Goal
Implement CPU raycasting via Rapier QueryPipeline for lidar simulation.

## Changes

### 1. `crates/clankers-physics/src/rapier/context.rs`
- Add `pub query_pipeline: rapier3d::prelude::QueryPipeline` to RapierContext
- Initialize with `QueryPipeline::new()` in RapierContext constructor

### 2. `crates/clankers-physics/src/rapier/systems.rs`
- After each substep in rapier_step_system, call query_pipeline.update(&rigid_body_set, &collider_set)

### 3. `crates/clankers-core/src/physics.rs`
- Add LidarConfig component: num_rays, num_channels, max_range, half_fov, vertical_half_fov, origin_offset
- Export from prelude

### 4. `crates/clankers-env/src/sensors.rs`
- Add LidarSensor struct implementing ObservationSensor + Sensor
- read() queries RapierContext, fires cast_ray for each (channel, azimuth) pair
- Returns flat Observation of distances (NaN for no hit)
- Unit tests: zero rays, single ray hit/miss, max-range clamp, multi-channel
