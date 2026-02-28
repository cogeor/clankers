# Implementation Log

## Task 1: SegmentationClass component in clankers-core

Completed: 2026-02-28

### Changes

- `crates/clankers-core/src/types.rs`: Added `SegmentationClass(pub u32)` Bevy component with constants `GROUND=0`, `WALL=1`, `ROBOT=2`, `OBSTACLE=3`, `TABLE=4`. Added five unit tests covering constants, copy/eq/hash, debug format, and inequality.
- `crates/clankers-core/src/lib.rs`: Exported `SegmentationClass` from the prelude `types::{ ... }` list.

### Verification

- [x] `SegmentationClass::GROUND.0 == 0`: confirmed by test
- [x] `SegmentationClass::ROBOT.0 == 2`: confirmed by test
- [x] Copy + Eq + Hash: confirmed by test
- [x] Debug format contains type name: confirmed by test
- [x] Exported from prelude: confirmed by doc-test compilation

---

## Task 2: SegmentationPalette + SegmentationFrameBuffer (new file)

Completed: 2026-02-28

### Changes

- `crates/clankers-render/src/segmentation.rs` (NEW): Created with:
  - `SegmentationPalette`: Resource with `colors: HashMap<u32, [f32; 3]>` (sRGB triples, no `bevy_color` dependency needed at non-gpu level). Default assigns red/green/blue/yellow/gray to classes 0-4.
  - `SegmentationFrameBuffer`: Resource with packed RGB `Vec<u8>`, `write_frame()`, `frame_counter()`, `width()`, `height()`, `data()`. Same pattern as `DepthFrameBuffer`.
  - `ClankersSegmentationPlugin` (behind `#[cfg(feature = "gpu")]`): registers palette, materials resource, `GpuReadbackPlugin`, `build_segmentation_materials` startup system, `attach_readback_to_segmentation_cameras` update system, `handle_segmentation_readback_complete` observer.
  - `SegmentationCamera` marker component.
  - `SegmentationImageHandle` component.
  - `SegmentationMaterials` resource.
  - `spawn_segmentation_camera_sensor()` helper.
  - `both_layers()` helper returning `RenderLayers::from_layers(&[0, 1])`.
  - Seven non-gpu tests: palette has 5 classes, ground color is red, buffer new/write/counter/wrong-size-panic/default/clone.

### Notes

- `Color` from `bevy::color` requires the `bevy_color` feature which is not in the base Bevy deps for this crate. Used `[f32; 3]` RGB tuples for the palette instead. Inside `gpu_impl` (which has `bevy_pbr` → `bevy_color`), converts via `Color::srgb(r, g, b)` when building `StandardMaterial`.

### Verification

- [x] Palette default has 5 classes: `segmentation_palette_default_has_five_classes` passes
- [x] Ground color is red `[1.0, 0.0, 0.0]`: `segmentation_palette_ground_is_red` passes
- [x] FrameBuffer write/read roundtrip: `segmentation_frame_buffer_write_read_roundtrip` passes
- [x] Frame counter increments: `segmentation_frame_buffer_write_increments_counter` passes
- [x] Wrong-size write panics: `segmentation_frame_buffer_wrong_size_panics` passes
- [x] Default is 512×512: `segmentation_frame_buffer_default` passes

---

## Task 3: SegmentationSensor in sensor.rs

Completed: 2026-02-28

### Changes

- `crates/clankers-render/src/sensor.rs`: Added import `use crate::segmentation::SegmentationFrameBuffer;` at top and import inside test module. Added `SegmentationSensor` struct with `label`, `width`, `height` fields. Implements `Sensor<Output = Observation>` (reads `SegmentationFrameBuffer`, normalises `u8 / 255.0`, falls back to zeros) and `ObservationSensor` (`observation_dim() = width * height * 3`). Added six tests.

### Verification

- [x] `observation_dim() == 64 * 48 * 3`: `segmentation_sensor_observation_dim` passes
- [x] Name returns label: `segmentation_sensor_name` passes
- [x] Rate is None: `segmentation_sensor_rate_is_none` passes
- [x] Reads and normalises u8 bytes: `segmentation_sensor_reads_from_frame_buffer_and_normalises` passes
- [x] All values in [0,1]: `segmentation_sensor_normalises_to_unit_range` passes
- [x] Fallback zeros when no resource: `segmentation_sensor_fallback_when_no_buffer` passes
- [x] u8 → f32 linearisation spot checks: `segmentation_sensor_u8_to_f32_linearisation` passes

---

## Task 4: Exports in clankers-render/src/lib.rs

Completed: 2026-02-28

### Changes

- `crates/clankers-render/src/lib.rs`: Added `pub mod segmentation;`, top-level re-exports `SegmentationFrameBuffer`, `SegmentationPalette`, `SegmentationSensor`. In the `prelude` module: added non-gpu exports `SegmentationFrameBuffer`, `SegmentationPalette`, `SegmentationSensor`; added gpu-gated exports `ClankersSegmentationPlugin`, `SegmentationCamera`, `SegmentationImageHandle`, `SegmentationMaterials`, `both_layers`, `spawn_segmentation_camera_sensor`.

### Verification

- [x] Module declared and visible: build succeeds
- [x] All new types accessible from `clankers_render::prelude`: confirmed by successful compilation

---

## Build and Test Summary

```
cargo build -j 24 -p clankers-core -p clankers-render  →  Finished (no errors)
cargo test  -j 24 -p clankers-render -p clankers-core  →  286 passed, 0 failed
```

- clankers-core: 211 tests pass (includes 5 new SegmentationClass tests)
- clankers-render: 75 tests pass (includes 7 new segmentation buffer/palette tests + 7 new SegmentationSensor tests)
- All doc-tests pass (21 total including new SegmentationPalette, SegmentationFrameBuffer, SegmentationSensor doc-tests)
