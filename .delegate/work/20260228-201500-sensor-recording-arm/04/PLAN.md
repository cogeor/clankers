# Loop 04: DepthSensor — DepthFrameBuffer, ClankersDepthPlugin

## Goal
Add depth sensor capability via GPU depth prepass readback.

## Changes

### 1. `crates/clankers-render/src/config.rs`
- Add `PixelFormat::DepthF32` variant (4 bytes per pixel)

### 2. `crates/clankers-render/src/buffer.rs`
- Add `DepthFrameBuffer` resource: width, height, Vec<f32> depth values
- write_depth_frame(Vec<f32>), read methods
- Unit tests for write/read roundtrip

### 3. `crates/clankers-render/src/depth.rs` (NEW)
- `ClankersDepthPlugin` (behind `gpu` feature):
  - Creates offscreen Image with Depth32Float format
  - Spawns DepthCamera with DepthPrepass component
  - Readback system copies depth texture to DepthFrameBuffer

### 4. `crates/clankers-render/src/sensor.rs`
- Add `DepthSensor` implementing the Sensor trait pattern
- read() reads DepthFrameBuffer, applies linearisation
- observation_dim() returns width * height

### 5. `crates/clankers-render/src/lib.rs`
- Export DepthSensor, DepthFrameBuffer, ClankersDepthPlugin
