# TASK: Sensor, Recording, and Arm Manipulation Pipeline

## Overview

Implement a complete sensor → recording → Python training pipeline with an arm manipulation
demo. All features must be modular, testable, and designed so that different sensor types,
robots, and learning approaches can be integrated with minimal code changes.

## Design Principles

1. **Trait-based sensors**: All sensors implement the existing `Sensor`/`ObservationSensor` traits.
   Adding a new sensor type means implementing one trait, not modifying framework code.
2. **Plugin architecture**: Each capability (depth, segmentation, recording, lidar) is a separate
   Bevy plugin that can be added independently.
3. **Registration pattern**: Sensors register through `SensorRegistry`. Recording channels
   register through a `RecorderConfig`. No hard-wired dependencies between features.
4. **Integration tests**: Focus on end-to-end tests that spawn a minimal Bevy app, register
   sensors, step physics, and verify data flows correctly through the pipeline.

## Sub-tasks (dependency order)

### 1. Image defaults 512x512
Source: 20260228-200700-image-defaults/TASK.md
- Change RenderConfig::default() from 256x256 to 512x512
- Update 3 test assertions and 3 doc-comments

### 2. Camera sensor GPU capture bridge
Source: 20260228-195200-camera-sensor/TASK.md
- SimCamera component + CameraSensorBundle
- CameraFrameBuffers (HashMap<String, FrameBuffer>)
- GPU readback (Readback or manual copy_texture_to_buffer)
- ImageSensor reads from named CameraFrameBuffers
- observation_dim() returns correct non-zero value

### 3. LidarSensor (CPU raycasting)
Source: 20260228-195500-lidar-depth-segmentation/TASK.md (sub-task 1)
- QueryPipeline in RapierContext, updated after each substep
- LidarConfig component
- LidarSensor implementing ObservationSensor

### 4. DepthSensor (GPU depth readback)
Source: 20260228-195500-lidar-depth-segmentation/TASK.md (sub-task 2)
- PixelFormat::DepthF32
- DepthFrameBuffer
- ClankersDepthPlugin (offscreen depth camera + Readback)
- DepthSensor implementing ObservationSensor

### 5. SegmentationSensor (flat-color material override)
Source: 20260228-195500-lidar-depth-segmentation/TASK.md (sub-task 3)
- SegmentationClass component
- SegmentationPalette resource
- ClankersSegmentationPlugin (RenderLayers, material swap, offscreen camera)
- SegmentationSensor implementing ObservationSensor

### 6. Recording module (MCAP) + Python bridge
Source: 20260228-200500-python-recording-bridge/TASK.md
- clankers-record crate with RecorderPlugin, MCAP writer
- Channels: /joint_states, /actions, /reward, /camera/image
- Binary obs protocol extension for TCP gym server
- Python McapEpisodeLoader + EpisodeDataset
- Viz replay mode (VizMode::Replay, PlaybackState, timeline scrubber)

### 7. Arm manipulation scene
Source: 20260228-200600-arm-scene-objects/TASK.md
- URDF collision geometry + 2-finger gripper
- Table + 3 interactable objects (cubes, cylinder)
- Proximity-triggered grasp (FixedJoint)
- SegmentationClass on all entities
- EndEffectorState population
- Full sensor integration demo

## Acceptance Criteria

- All sensors implement ObservationSensor trait consistently
- Depth, segmentation, lidar, camera all work as independent Bevy plugins
- Recording captures any registered sensor data to MCAP
- Python can load MCAP episodes into numpy arrays
- Online training gets binary image data via TCP
- Arm manipulation demo exercises all sensors
- cargo build -j 24 passes
- Integration tests verify sensor → buffer → readout pipeline
- Adding a new sensor type requires implementing one trait + registering
