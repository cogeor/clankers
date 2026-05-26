//! Regenerate the committed `python/tests/fixtures/two_camera.mcap`
//! fixture used by the W6 multi-camera Python loader tests.
//!
//! # Regenerate
//!
//! ```text
//! cargo run -j 8 -p clankers-record --example fixture_gen --features camera
//! ```
//!
//! The output lands at `python/tests/fixtures/two_camera.mcap`. The
//! binary is tracked in git via `git add -f` because `*.mcap` is
//! `.gitignore`d (see root `.gitignore` line 51).
//!
//! # Fixture content
//!
//! Two camera labels `front` and `wrist`, three frames per camera at
//! 16x16 RGB8 (3 bytes per pixel = 768 bytes per frame). Deterministic
//! per-pixel patterns:
//!
//! - `front`: vertical gradient — row `y` is filled with byte `(y * 16)`
//!   across all channels.
//! - `wrist`: horizontal gradient — column `x` is filled with byte
//!   `(x * 16)` across all channels.
//!
//! Three frames per camera mean six raw-image MCAP records. Timestamps
//! advance by 10 ms per `app.update()` so each frame has a distinct
//! `log_time`.
//!
//! # Stability
//!
//! Treat the fixture as **regeneration-only — never hand-edit**. The
//! committed `.mcap` is the blessed output of one regeneration; the
//! recorder's byte layout depends on `mcap::Writer` framing, the
//! recorder's sequence counter, and `CameraFrameBuffers`' internal
//! `HashMap` iteration order. The roundtrip test
//! (`tests/multi_camera_roundtrip.rs`) asserts the channel **set**
//! rather than byte order, so functional equivalence survives
//! regeneration even when byte-for-byte equality does not.
//!
//! Regenerate from a clean checkout when the recorder byte format or
//! channel schema changes (W7 PR4 async writer is the expected next
//! consumer). Verify the resulting file stays under 20 KB (hard
//! ceiling) and ideally under 15 KB (target).

use std::path::PathBuf;

use bevy::MinimalPlugins;
use bevy::prelude::*;

use clankers_core::time::SimTime;
use clankers_record::prelude::*;
use clankers_render::buffer::{CameraFrameBuffers, FrameBuffer};
use clankers_render::config::PixelFormat;

const FIXTURE_PATH: &str = "python/tests/fixtures/two_camera.mcap";
const FRAME_WIDTH: u32 = 16;
const FRAME_HEIGHT: u32 = 16;
const FRAMES_PER_CAMERA: u32 = 3;
const FRAME_DT_NS: u64 = 10_000_000; // 10 ms

/// Build the deterministic pixel pattern for the `front` camera at
/// frame index `frame_idx`. Vertical gradient: row `y` filled with
/// `(y * 16) + frame_idx` (mod 256) across all three channels.
fn front_pattern(frame_idx: u8) -> Vec<u8> {
    let bpp = PixelFormat::Rgb8.bytes_per_pixel() as usize;
    let mut data = vec![0u8; (FRAME_WIDTH * FRAME_HEIGHT) as usize * bpp];
    for y in 0..FRAME_HEIGHT {
        let row_byte = (u8::try_from(y).unwrap_or(0).wrapping_mul(16)).wrapping_add(frame_idx);
        for x in 0..FRAME_WIDTH {
            let offset = ((y * FRAME_WIDTH + x) as usize) * bpp;
            data[offset] = row_byte;
            data[offset + 1] = row_byte;
            data[offset + 2] = row_byte;
        }
    }
    data
}

/// Build the deterministic pixel pattern for the `wrist` camera at
/// frame index `frame_idx`. Horizontal gradient: column `x` filled
/// with `(x * 16) + frame_idx` (mod 256) across all three channels.
fn wrist_pattern(frame_idx: u8) -> Vec<u8> {
    let bpp = PixelFormat::Rgb8.bytes_per_pixel() as usize;
    let mut data = vec![0u8; (FRAME_WIDTH * FRAME_HEIGHT) as usize * bpp];
    for y in 0..FRAME_HEIGHT {
        for x in 0..FRAME_WIDTH {
            let col_byte = (u8::try_from(x).unwrap_or(0).wrapping_mul(16)).wrapping_add(frame_idx);
            let offset = ((y * FRAME_WIDTH + x) as usize) * bpp;
            data[offset] = col_byte;
            data[offset + 1] = col_byte;
            data[offset + 2] = col_byte;
        }
    }
    data
}

/// System that updates both camera frame buffers with a fresh
/// deterministic pattern at each `app.update()`. Frame index is read
/// from the `front` buffer's frame counter — both buffers advance in
/// lockstep.
fn write_camera_patterns(mut frame_buffers: ResMut<CameraFrameBuffers>) {
    // Snapshot the current frame index from `front` BEFORE mutating —
    // `write_frame` increments the counter. `frame_counter` is `u64`
    // but the fixture writes < 256 frames so `u8` truncation is safe.
    let frame_idx_u64 = frame_buffers
        .get("front")
        .map_or(0u64, FrameBuffer::frame_counter);
    if frame_idx_u64 >= u64::from(FRAMES_PER_CAMERA) {
        return;
    }
    let frame_idx = u8::try_from(frame_idx_u64).unwrap_or(0);
    if let Some(buf) = frame_buffers.get_mut("front") {
        buf.write_frame(front_pattern(frame_idx));
    }
    if let Some(buf) = frame_buffers.get_mut("wrist") {
        buf.write_frame(wrist_pattern(frame_idx));
    }
}

/// System that advances `SimTime` by [`FRAME_DT_NS`] per update. Runs
/// after the recorder so successive frames carry distinct timestamps.
fn advance_sim_time(mut sim_time: ResMut<SimTime>) {
    sim_time.advance(FRAME_DT_NS);
}

fn main() {
    let output = PathBuf::from(FIXTURE_PATH);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).expect("create fixtures dir");
    }

    // Build the two-camera frame buffer resource up-front so the
    // recorder's first PostUpdate observes both labels.
    let mut frame_buffers = CameraFrameBuffers::default();
    frame_buffers.insert(
        "front".to_string(),
        FrameBuffer::new(FRAME_WIDTH, FRAME_HEIGHT, PixelFormat::Rgb8),
    );
    frame_buffers.insert(
        "wrist".to_string(),
        FrameBuffer::new(FRAME_WIDTH, FRAME_HEIGHT, PixelFormat::Rgb8),
    );

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(SimTime::new());
    app.insert_resource(frame_buffers);
    app.insert_resource(RecordingConfig {
        output_path: output.clone(),
        record_joints: false,
        record_actions: false,
        record_rewards: false,
        record_body_poses: false,
        ..RecordingConfig::default()
    });
    app.add_plugins(RecorderPlugin);

    // `write_camera_patterns` runs in `PreUpdate` so the recorder
    // (which runs in `PostUpdate`) observes the freshly-written frame.
    // `advance_sim_time` runs in `Last` so successive `app.update()`
    // calls record distinct timestamps without skewing the first
    // frame.
    app.add_systems(PreUpdate, write_camera_patterns);
    app.add_systems(Last, advance_sim_time);

    app.finish();
    app.cleanup();

    for _ in 0..FRAMES_PER_CAMERA {
        app.update();
    }

    // Drop the app to run `Recorder::Drop` which finalises the MCAP
    // footer.
    drop(app);

    let size = std::fs::metadata(&output)
        .expect("read fixture metadata")
        .len();
    let display = output.display();
    println!("fixture_gen: wrote {display} ({size} bytes)");
    if size >= 20480 {
        eprintln!("fixture_gen: WARNING — fixture is {size} bytes, hard ceiling is 20480");
    }
}
