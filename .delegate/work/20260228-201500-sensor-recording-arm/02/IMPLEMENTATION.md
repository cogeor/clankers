# Implementation Log

## Task 1: Add `gpu` feature to Cargo.toml

Completed: 2026-02-28

### Changes

- `crates/clankers-render/Cargo.toml`: Added a `gpu` feature that enables
  `bevy/bevy_render`, `bevy/bevy_core_pipeline`, and `bevy/bevy_pbr`. The
  workspace-level bevy dependency already has `default-features = false`, so
  the gpu feature explicitly pulls in only the rendering sub-crates needed.

### Verification

- [x] `cargo build -j 24 -p clankers-render`: ok (headless, no gpu feature)
- [x] `cargo build -j 24 -p clankers-render --features gpu`: ok (full render stack)

---

## Task 2: Add `label` field to `CameraConfig`

Completed: 2026-02-28

### Changes

- `crates/clankers-render/src/config.rs`:
  - Added `pub label: String` field to `CameraConfig`.
  - Changed `new()` from `const fn` to `fn` (String allocation requires heap).
  - Changed builder methods (`with_fov_y`, `with_near`, `with_far`) from
    `const fn` to `fn` to match consistency.
  - Added `with_label(label: impl Into<String>) -> Self` builder.
  - Added two new tests: `camera_config_default_label_is_empty` and
    `camera_config_with_label`.

### Verification

- [x] All pre-existing `CameraConfig` tests still pass.
- [x] New label tests pass.

---

## Task 3: Add `CameraFrameBuffers` to `buffer.rs`

Completed: 2026-02-28

### Changes

- `crates/clankers-render/src/buffer.rs`:
  - Added `use std::collections::HashMap`.
  - Added `CameraFrameBuffers(HashMap<String, FrameBuffer>)` as a public
    `Resource` struct.
  - Added methods: `get`, `get_mut`, `insert`, `remove`, `iter`, `iter_mut`,
    `len`, `is_empty`.
  - Added 6 unit tests covering empty default, insert/get, get_mut, remove,
    multiple cameras, and iteration.

### Verification

- [x] 7 new `camera_frame_buffers_*` tests pass.
- [x] All pre-existing `frame_buffer_*` tests still pass.

---

## Task 4: Create `camera.rs` — `SimCamera` + `spawn_camera_sensor()`

Completed: 2026-02-28

### Changes

- `crates/clankers-render/src/camera.rs` (NEW):
  - `SimCamera` marker component (gated behind `#[cfg(feature = "gpu")]`).
  - `spawn_camera_sensor(commands, images, camera_frame_buffers, config, width, height)`:
    - Creates an `Image` asset with `TEXTURE_BINDING | COPY_DST | RENDER_ATTACHMENT | COPY_SRC` usage.
    - Format: `TextureFormat::Rgba8UnormSrgb`.
    - Registers a `FrameBuffer::new(width, height, PixelFormat::Rgba8)` in
      `CameraFrameBuffers` keyed by `config.label`.
    - Spawns a `Camera3d` entity with `Camera { target: RenderTarget::Image(...) }`,
      `SimCamera`, and the `CameraConfig`.
    - Returns `(Entity, Handle<Image>)`.
    - Panics if `config.label.is_empty()`.
  - Import notes: `RenderTarget` accessed via `bevy::camera::RenderTarget`
    (re-exported from `bevy_camera`); `Camera3d` and `Camera` from
    `bevy::prelude::*`.

### Verification

- [x] `cargo build -j 24 -p clankers-render --features gpu`: ok
- [x] Module compiles clean with no warnings.

---

## Task 5: Create `readback.rs` — `ImageCopyPlugin`

Completed: 2026-02-28

### Changes

- `crates/clankers-render/src/readback.rs` (NEW):
  - `ImageCopyPlugin` (gated behind `#[cfg(feature = "gpu")]`):
    - Adds `GpuReadbackPlugin::default()`.
    - Adds `attach_readback_to_new_cameras` system (Update): queries
      `SimCamera` entities without `Readback`, attaches `Readback::texture(handle)`.
    - Adds `handle_readback_complete` observer for `ReadbackComplete` events:
      - Gets entity from `trigger.entity` (field on `ReadbackComplete`).
      - Looks up `CameraConfig` for the entity to find its label.
      - Looks up `CameraFrameBuffers` by label.
      - Strips wgpu row-padding (256-byte alignment) from the raw GPU bytes.
      - Calls `buf.write_frame(packed)`.
  - Used `On<ReadbackComplete>` (Bevy 0.17 renamed `Trigger` to `On`).
  - Hardcoded `COPY_BYTES_PER_ROW_ALIGNMENT = 256` (stable wgpu constant,
    not re-exported by bevy's render_resource module).

### Verification

- [x] `cargo build -j 24 -p clankers-render --features gpu`: ok
- [x] Doctest for `ImageCopyPlugin` passes with `--features gpu`.

---

## Task 6: Update `sensor.rs` — `ImageSensor` with width/height/channels

Completed: 2026-02-28

### Changes

- `crates/clankers-render/src/sensor.rs`:
  - Added `width: u32`, `height: u32`, `channels: u32` fields to `ImageSensor`.
  - Changed constructor to `new(label, width, height, channels)`.
  - Added `width()`, `height()`, `channels()` accessors.
  - `observation_dim()` now returns `(width * height * channels) as usize`
    (non-zero for any real camera).
  - `read()` now:
    1. Tries `CameraFrameBuffers.get(label)` first.
    2. Falls back to a zero-filled `Observation` of `observation_dim()` length
       if no buffer found (or `CameraFrameBuffers` resource absent).
  - Updated all tests to use new 4-arg constructor.
  - Added new tests: `image_sensor_observation_dim_nonzero`,
    `image_sensor_observation_dim_rgba`, `image_sensor_fallback_when_no_buffer`,
    `image_sensor_fallback_when_no_resource`, `image_sensor_width_height_channels`.
  - Fixed doctest: added `use clankers_core::traits::ObservationSensor`.

### Verification

- [x] `observation_dim()` returns `64 * 48 * 3 = 9216` (non-zero).
- [x] `read()` finds data in `CameraFrameBuffers` by label.
- [x] Fallback returns zero vec of correct size.
- [x] All 9 sensor tests pass.

---

## Task 7: Update `lib.rs` — register new resources and plugins

Completed: 2026-02-28

### Changes

- `crates/clankers-render/src/lib.rs`:
  - Added `pub mod camera` and `pub mod readback`.
  - Added `init_resource::<CameraFrameBuffers>()` in `ClankersRenderPlugin::build`.
  - Added `pub use buffer::{CameraFrameBuffers, FrameBuffer}` re-export.
  - Prelude now exports `CameraFrameBuffers`, and (under `#[cfg(feature = "gpu")]`)
    `SimCamera` and `ImageCopyPlugin`.
  - Updated module-level doc to describe the `gpu` feature.
  - Added `camera_frame_buffers_initialised_empty` test.
  - Updated `plugin_builds_without_panic` to also assert `CameraFrameBuffers`
    is present.

### Verification

- [x] `plugin_builds_without_panic`: asserts `CameraFrameBuffers` is some.
- [x] `camera_frame_buffers_initialised_empty`: new test passes.
- [x] All 48 unit tests pass without `gpu` feature.
- [x] All 48 unit tests pass with `--features gpu`.
- [x] 5 doctests pass without `gpu` feature; 6 doctests pass with `--features gpu`.

---

## Final Build & Test Summary

```
cargo build -j 24 -p clankers-render
  -> Finished `dev` profile

cargo test -j 24 -p clankers-render
  -> test result: ok. 48 passed; 0 failed (unit tests)
  -> test result: ok. 5 passed; 0 failed (doc tests)

cargo build -j 24 -p clankers-render --features gpu
  -> Finished `dev` profile

cargo test -j 24 -p clankers-render --features gpu
  -> test result: ok. 48 passed; 0 failed (unit tests)
  -> test result: ok. 6 passed; 0 failed (doc tests)
```

### Files Created/Modified

| File | Action |
|------|--------|
| `crates/clankers-render/Cargo.toml` | Modified — added `gpu` feature |
| `crates/clankers-render/src/config.rs` | Modified — added `label` to `CameraConfig`, `with_label()` |
| `crates/clankers-render/src/buffer.rs` | Modified — added `CameraFrameBuffers` resource |
| `crates/clankers-render/src/camera.rs` | Created — `SimCamera`, `spawn_camera_sensor()` |
| `crates/clankers-render/src/readback.rs` | Created — `ImageCopyPlugin`, readback observer |
| `crates/clankers-render/src/sensor.rs` | Modified — `ImageSensor` with label + dimensions |
| `crates/clankers-render/src/lib.rs` | Modified — registered resources, new modules, exports |
