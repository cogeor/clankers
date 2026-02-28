# Loop 02: Camera Sensor GPU Capture Bridge

## Goal
Implement offscreen camera sensors that render to texture and copy pixels to CPU.

## Changes

### 1. `crates/clankers-render/Cargo.toml`
- Add bevy features: `bevy_render`, `bevy_core_pipeline`, `bevy_pbr` (gated behind `gpu` feature)

### 2. `crates/clankers-render/src/config.rs`
- Add `label: String` field to `CameraConfig`
- Add `with_label()` builder method

### 3. `crates/clankers-render/src/camera.rs` (NEW)
- `SimCamera` marker component
- `spawn_camera_sensor()` helper: creates Image with RENDER_ATTACHMENT|COPY_SRC,
  spawns Camera3d with RenderTarget::Image, attaches SimCamera + CameraConfig

### 4. `crates/clankers-render/src/buffer.rs`
- Add `CameraFrameBuffers(HashMap<String, FrameBuffer>)` resource
- Keep FrameBuffer as inner per-camera type
- get/get_mut/insert methods keyed by label

### 5. `crates/clankers-render/src/readback.rs` (NEW)
- `ImageCopyPlugin`: Bevy plugin
- System that queries SimCamera entities with their render target Image handle
- Uses Readback component or manual readback to copy pixels from GPU to CPU
- Strips wgpu row-padding
- Writes into CameraFrameBuffers by label

### 6. `crates/clankers-render/src/sensor.rs`
- Update ImageSensor to store label and resolution
- read() looks up CameraFrameBuffers by label
- observation_dim() returns width*height*channels (non-zero)

### 7. `crates/clankers-render/src/lib.rs`
- Register CameraFrameBuffers resource
- Add ImageCopyPlugin (behind gpu feature)
- Export new types
