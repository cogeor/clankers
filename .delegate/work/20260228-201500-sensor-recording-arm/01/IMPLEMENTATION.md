# Loop 01 Implementation

## config.rs
- Changed `Self::new(256, 256)` → `Self::new(512, 512)` in `Default for RenderConfig`
- Updated doc-example from 640x480 to 512x512
- Updated `render_config_default` test: 256→512

## buffer.rs
- Updated `frame_buffer_default` test: 256→512

## lib.rs
- Updated doc-example from 640x480 to 512x512
- Updated comment: "default 256x256" → "default 512x512"
- Updated `plugin_default_config` test: 256→512
