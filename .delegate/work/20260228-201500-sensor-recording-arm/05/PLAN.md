# Loop 05: SegmentationSensor — SegmentationClass, Palette, ClankersSegmentationPlugin

## Goal
Add semantic segmentation sensor with flat-color material override rendering.

## Changes

### 1. `crates/clankers-core/src/types.rs`
- Add SegmentationClass(pub u32) Bevy component
- Constants: CLASS_GROUND=0, CLASS_WALL=1, CLASS_ROBOT=2, CLASS_OBSTACLE=3, CLASS_TABLE=4
- Export from prelude

### 2. `crates/clankers-render/src/segmentation.rs` (NEW)
- SegmentationPalette resource: HashMap<u32, Color> with defaults
- SegmentationFrameBuffer resource
- ClankersSegmentationPlugin (behind `gpu` feature):
  - Creates per-class unlit StandardMaterial handles
  - Spawns SegmentationCamera on RenderLayers::layer(1)
  - Material swap in PreUpdate, restore in PostUpdate
  - Readback to SegmentationFrameBuffer

### 3. `crates/clankers-render/src/sensor.rs`
- Add SegmentationSensor implementing ObservationSensor
- read() reads SegmentationFrameBuffer, converts to [0,1] float RGB
- observation_dim() returns width * height * 3

### 4. `crates/clankers-render/src/lib.rs`
- Export SegmentationSensor, SegmentationFrameBuffer, etc.
