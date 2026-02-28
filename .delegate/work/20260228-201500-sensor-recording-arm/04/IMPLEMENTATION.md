# Implementation Log

## Task 1: PixelFormat::DepthF32

Completed: 2026-02-28T20:15:00Z

### Changes

- `crates/clankers-render/src/config.rs`: Added `DepthF32` variant to `PixelFormat`. `bytes_per_pixel()` returns 4, `channels()` returns 1. Added test assertions for both methods.

### Verification

- [x] `PixelFormat::DepthF32.bytes_per_pixel() == 4`: passes
- [x] `PixelFormat::DepthF32.channels() == 1`: passes

---

## Task 2: DepthFrameBuffer

Completed: 2026-02-28T20:15:00Z

### Changes

- `crates/clankers-render/src/buffer.rs`: Added `DepthFrameBuffer` resource struct with fields `width`, `height`, `data: Vec<f32>`, `frame_counter: u64`. Implemented `new(width, height)`, `write_depth_frame(Vec<f32>)`, `data()`, `width()`, `height()`, `frame_counter()`, `Default`. Added 6 unit tests covering new, write/read roundtrip, counter increment, wrong-size panic, default, and clone.

### Verification

- [x] `depth_frame_buffer_new`: passes
- [x] `depth_frame_buffer_write_read_roundtrip`: passes
- [x] `depth_frame_buffer_write_increments_counter`: passes
- [x] `depth_frame_buffer_write_wrong_size_panics`: passes
- [x] `depth_frame_buffer_default`: passes
- [x] `depth_frame_buffer_clone`: passes

---

## Task 3: ClankersDepthPlugin (depth.rs)

Completed: 2026-02-28T20:15:00Z

### Changes

- `crates/clankers-render/src/depth.rs` (NEW): Created `ClankersDepthPlugin` behind `#[cfg(feature = "gpu")]`. Pattern follows `readback.rs` exactly:
  - `DepthCamera` marker component
  - `DepthImageHandle(Handle<Image>)` component carrying the depth image ref
  - `spawn_depth_camera_sensor()` helper that creates a `Depth32Float` image with `TEXTURE_BINDING | COPY_SRC | RENDER_ATTACHMENT` usages, spawns `Camera3d + DepthCamera + DepthPrepass + DepthImageHandle`
  - `attach_readback_to_depth_cameras` system: attaches `Readback::texture(handle)` to newly spawned depth cameras (mirrors `attach_readback_to_new_cameras` in readback.rs)
  - `handle_depth_readback_complete` observer: strips wgpu row-padding (256-byte alignment), reinterprets raw bytes as `f32` via `f32::from_le_bytes`, writes to `DepthFrameBuffer`
  - `GpuReadbackPlugin` registered in `ClankersDepthPlugin::build`

### Notes

`DepthPrepass` is confirmed available as `bevy::core_pipeline::prepass::DepthPrepass` in Bevy 0.17.3. Bevy's internal depth prepass writes its texture through its own render graph nodes; connecting the prepass depth texture to the registered `Depth32Float` Image asset requires a custom render-graph node. That GPU-side wiring is deferred to a future loop. The CPU side (DepthFrameBuffer, DepthSensor, plugin structure, readback observer) is fully implemented and compile-verified.

### Verification

- [x] Compiles under `cfg(feature = "gpu")`: confirmed via `cargo build`
- [x] `DepthPrepass` import resolves from `bevy::core_pipeline::prepass`: confirmed
- [x] `Readback::texture(handle)` accepts `Handle<Image>`: confirmed

---

## Task 4: DepthSensor

Completed: 2026-02-28T20:15:00Z

### Changes

- `crates/clankers-render/src/sensor.rs`: Added `DepthSensor` struct with public fields `label`, `width`, `height`, `near`, `far`. Implemented `Sensor` and `ObservationSensor` traits. `read()` reads `DepthFrameBuffer`, applies linearisation `(2*near*far) / (far + near - raw*(far-near))`. `observation_dim()` returns `width * height`. Added `linearise(raw: f32) -> f32` helper method. Added 6 unit tests.

### Verification

- [x] `depth_sensor_observation_dim`: passes
- [x] `depth_sensor_name`: passes
- [x] `depth_sensor_rate_is_none`: passes
- [x] `depth_sensor_linearise_known_values` (near=1, far=10, raw=0.0 and raw=1.0): passes
- [x] `depth_sensor_reads_from_depth_frame_buffer`: passes
- [x] `depth_sensor_fallback_when_no_buffer`: passes

---

## Task 5: Exports in lib.rs

Completed: 2026-02-28T20:15:00Z

### Changes

- `crates/clankers-render/src/lib.rs`: Added `pub mod depth`. Re-exported `DepthFrameBuffer` from `buffer`, `DepthSensor` from `sensor`. Prelude includes `DepthFrameBuffer`, `DepthSensor`, and (under `gpu`) `ClankersDepthPlugin`, `DepthCamera`, `spawn_depth_camera_sensor`. Updated module-level doc comment.

### Verification

- [x] `cargo build -j 24 -p clankers-render`: Finished successfully (1.79s)
- [x] `cargo test -j 24 -p clankers-render`: 60 passed, 0 failed; 7 doc-tests passed

---
