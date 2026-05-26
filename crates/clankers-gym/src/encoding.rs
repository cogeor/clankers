//! Typed observation encoding for the gym wire protocol.
//!
//! Promoted in W4 PR1 from [`crate::protocol::ObsEncoding`] (now deprecated).
//! [`EncodedObservation`] is the typed wrapper used uniformly on
//! [`Response::Reset`](crate::protocol::Response::Reset) and
//! [`Response::Step`](crate::protocol::Response::Step) so vision envs
//! satisfy the Gymnasium space contract on the first reset.
//!
//! See `docs/plans/WS4-plan.md` § 4 for the rationale.
//!
//! # Wire format
//!
//! The JSON header for a [`EncodedObservation::RawU8Image`] carries
//! `width`, `height`, `channels`, and `layout`. The raw pixel bytes
//! ship as a second length-prefixed binary frame immediately after the
//! JSON response. The `payload: Vec<u8>` field on the enum is
//! `#[serde(skip)]` so it never serialises; on the read side it is
//! always reconstructed from the binary frame.
//!
//! # Layout
//!
//! [`ImageLayout::Hwc`] is the default — matches the in-tree image
//! observation materialisation:
//! [`ImageSensor::read`](../../clankers_render/sensor/struct.ImageSensor.html#method.read)
//! flattens its `FrameBuffer` of `width * height * channels` bytes
//! row-major HWC (verified by inspection of `clankers-render/src/sensor.rs`
//! and `clankers-render/src/buffer.rs`).
//! [`ImageLayout::Chw`] is reserved for future channel-first sensors.

use std::collections::BTreeMap;

use clankers_core::types::{BatchResetResult, BatchStepResult, ObservationSpace};
use clankers_core::view::ObservationView;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ImageLayout
// ---------------------------------------------------------------------------

/// Memory layout of a raw image observation.
///
/// Used as a tag on [`EncodedObservation::RawU8Image`] so clients can
/// decode the binary payload into a tensor of the correct shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageLayout {
    /// Height × Width × Channels (row-major). Matches Bevy/wgpu's
    /// standard RGB/RGBA texture readback.
    Hwc,
    /// Channels × Height × Width (channel-first). Reserved for future
    /// channel-first sensors.
    Chw,
}

// ---------------------------------------------------------------------------
// EncodedObservation
// ---------------------------------------------------------------------------

/// Typed wrapper around an observation sent on the wire.
///
/// Promoted from `protocol::ObsEncoding` in W4 PR1. Used uniformly on
/// [`Response::Reset`](crate::protocol::Response::Reset) and
/// [`Response::Step`](crate::protocol::Response::Step). When the
/// variant is [`RawU8Image`](Self::RawU8Image) the raw pixel bytes
/// follow as a separate length-prefixed binary frame; the `payload`
/// field is `#[serde(skip)]` so it never appears in the JSON.
///
/// # Example
///
/// ```
/// use clankers_gym::encoding::{EncodedObservation, ImageLayout};
///
/// let enc = EncodedObservation::RawU8Image {
///     width: 64,
///     height: 64,
///     channels: 3,
///     layout: ImageLayout::Hwc,
///     payload: vec![],
/// };
/// let json = serde_json::to_string(&enc).unwrap();
/// assert!(json.contains("RawU8Image"));
/// assert!(!json.contains("payload"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EncodedObservation {
    /// Flat `Vec<f32>` carried inline in the JSON. Default for `Box` and
    /// other non-image observation spaces.
    ///
    /// Implementation note: serde's internally-tagged representation
    /// cannot pair a `#[serde(tag = "type")]` discriminator with a
    /// newtype variant wrapping a sequence, so `FlatF32` is a struct
    /// variant carrying a single `data: Vec<f32>` field.
    FlatF32 {
        /// The flat f32 observation values.
        data: Vec<f32>,
    },
    /// Raw `u8` pixel image. The header (`width`, `height`, `channels`,
    /// `layout`) is JSON; the pixel bytes ship as a separate
    /// length-prefixed binary frame immediately after the JSON response.
    RawU8Image {
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
        /// Number of channels (e.g. 3 for RGB, 4 for RGBA, 1 for grayscale).
        channels: u8,
        /// Memory layout of the binary payload.
        layout: ImageLayout,
        /// Pixel bytes. Never serialised; populated by the read side
        /// from the follow-up binary frame.
        #[serde(skip)]
        payload: Vec<u8>,
    },
    /// Dictionary of sub-observations keyed by name.
    Dict {
        /// Per-key sub-observation encodings.
        spaces: BTreeMap<String, Self>,
    },
    /// Batched `f32` observations carried as a single binary frame.
    ///
    /// Added in protocol `1.2.0` (W7 PR2). The JSON header carries
    /// `num_envs` and `obs_dim`; the raw `f32` bytes (length
    /// `num_envs * obs_dim * 4`) ship as a separate length-prefixed
    /// binary frame containing a [`crate::binary_frame::BinaryFrameHeader`]
    /// + payload.
    ///
    /// The `payload` field is `#[serde(skip)]` so it never appears in
    /// the JSON; on the read side it is reconstructed by calling
    /// [`crate::binary_frame::decode_batch_f32`] on the follow-up
    /// binary frame.
    BatchF32 {
        /// Number of environments in the batch.
        num_envs: u32,
        /// Per-env observation dimensionality.
        obs_dim: u32,
        /// In-process payload bytes. Never serialised; populated by the
        /// read side from the follow-up binary frame.
        #[serde(skip)]
        payload: Vec<u8>,
    },
    /// Batched raw `u8` image stack carried as a single binary frame.
    ///
    /// Added in protocol `1.2.0` (W7 PR2). The header (`num_envs`,
    /// `width`, `height`, `channels`, `layout`) is JSON; the pixel
    /// bytes ship as a separate length-prefixed binary frame containing
    /// a [`crate::binary_frame::BinaryFrameHeader`] + payload.
    BatchRawU8Image {
        /// Number of environments in the batch.
        num_envs: u32,
        /// Image width in pixels (per env).
        width: u32,
        /// Image height in pixels (per env).
        height: u32,
        /// Number of channels (e.g. 3 for RGB).
        channels: u8,
        /// Memory layout of each per-env image tile.
        layout: ImageLayout,
        /// In-process payload bytes. Never serialised; populated by the
        /// read side from the follow-up binary frame.
        #[serde(skip)]
        payload: Vec<u8>,
    },
}

// ---------------------------------------------------------------------------
// encode_observation helper
// ---------------------------------------------------------------------------

/// Encode a borrowed observation view into an
/// [`EncodedObservation`] plus an optional binary pixel payload.
///
/// - For [`ObservationSpace::Image`] when `binary` is `true`, returns
///   `(EncodedObservation::RawU8Image { payload: vec![] }, Some(pixel_bytes))`
///   where each `f32` in the view is clamped to `[0.0, 1.0]` and scaled
///   to a `u8` byte. The caller writes the bytes as a length-prefixed
///   binary frame immediately after the JSON response.
/// - Otherwise returns `(EncodedObservation::FlatF32(_), None)`.
///
/// # Naming-deviation note
///
/// The WS4-plan signature uses an `ObservationSchema` argument; the
/// in-tree gym server holds the looser [`ObservationSpace`] descriptor,
/// which already carries `width`/`height`/`channels` directly. We use
/// [`ObservationSpace`] for the dispatch — the schema's per-slot shape
/// is the same information indirectly.
#[must_use]
pub fn encode_observation(
    view: &ObservationView<'_>,
    space: &ObservationSpace,
    binary: bool,
) -> (EncodedObservation, Option<Vec<u8>>) {
    match space {
        ObservationSpace::Image {
            width,
            height,
            channels,
        } if binary => {
            // Convert f32 observation data to u8 pixels (multiply by 255).
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let pixel_bytes: Vec<u8> = view
                .as_f32()
                .iter()
                .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                .collect();
            let enc = EncodedObservation::RawU8Image {
                width: *width,
                height: *height,
                #[allow(clippy::cast_possible_truncation)]
                channels: u8::try_from(*channels).unwrap_or(3),
                layout: ImageLayout::Hwc,
                payload: Vec::new(),
            };
            (enc, Some(pixel_bytes))
        }
        _ => (
            EncodedObservation::FlatF32 {
                data: view.as_f32().to_vec(),
            },
            None,
        ),
    }
}

// ---------------------------------------------------------------------------
// Batch binary encoders (W7 PR2)
// ---------------------------------------------------------------------------

/// Flatten a batched `f32` step result into an
/// [`EncodedObservation::BatchF32`] JSON header + a single binary
/// frame (header + payload) for transmission.
///
/// The returned `Vec<u8>` is the bytes to be written as a separate
/// length-prefixed binary frame immediately after the JSON envelope.
///
/// # Panics
///
/// Panics if `result.observations` is empty (the dispatch site must
/// only call this when `num_envs > 0`) or if observations have
/// inconsistent `obs_dim`.
#[must_use]
pub fn encode_batch_step_binary(result: &BatchStepResult) -> (EncodedObservation, Vec<u8>) {
    encode_batch_flat_f32(&result.observations)
}

/// Same as [`encode_batch_step_binary`] but for a [`BatchResetResult`].
#[must_use]
pub fn encode_batch_reset_binary(result: &BatchResetResult) -> (EncodedObservation, Vec<u8>) {
    encode_batch_flat_f32(&result.observations)
}

fn encode_batch_flat_f32(
    observations: &[clankers_core::types::Observation],
) -> (EncodedObservation, Vec<u8>) {
    assert!(
        !observations.is_empty(),
        "encode_batch_flat_f32 requires at least one observation",
    );
    let num_envs = u32::try_from(observations.len()).expect("num_envs overflows u32");
    let obs_dim_usize = observations[0].len();
    let obs_dim = u32::try_from(obs_dim_usize).expect("obs_dim overflows u32");

    let total = num_envs as usize * obs_dim_usize;
    let mut flat = Vec::with_capacity(total);
    for obs in observations {
        debug_assert_eq!(
            obs.len(),
            obs_dim_usize,
            "batch observations must share obs_dim",
        );
        flat.extend_from_slice(obs.as_slice());
    }

    let payload = crate::binary_frame::encode_batch_f32(num_envs, obs_dim, &flat);
    let enc = EncodedObservation::BatchF32 {
        num_envs,
        obs_dim,
        payload: Vec::new(),
    };
    (enc, payload)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clankers_core::schema::SchemaDtype;

    #[test]
    fn image_layout_serde_roundtrip() {
        for layout in [ImageLayout::Hwc, ImageLayout::Chw] {
            let json = serde_json::to_string(&layout).unwrap();
            let back: ImageLayout = serde_json::from_str(&json).unwrap();
            assert_eq!(layout, back);
        }
    }

    #[test]
    fn encoded_observation_flat_f32_roundtrip() {
        let enc = EncodedObservation::FlatF32 {
            data: vec![1.0, 2.0, 3.0],
        };
        let json = serde_json::to_string(&enc).unwrap();
        let back: EncodedObservation = serde_json::from_str(&json).unwrap();
        assert!(json.contains("FlatF32"));
        match back {
            EncodedObservation::FlatF32 { data } => assert_eq!(data, vec![1.0, 2.0, 3.0]),
            _ => panic!("expected FlatF32"),
        }
    }

    #[test]
    fn encoded_observation_raw_u8_image_header_only() {
        // payload is #[serde(skip)] — JSON must never carry the bytes.
        let enc = EncodedObservation::RawU8Image {
            width: 320,
            height: 240,
            channels: 3,
            layout: ImageLayout::Hwc,
            payload: vec![1, 2, 3, 4],
        };
        let json = serde_json::to_string(&enc).unwrap();
        assert!(json.contains("RawU8Image"));
        assert!(json.contains("width"));
        assert!(json.contains("height"));
        assert!(json.contains("channels"));
        assert!(json.contains("layout"));
        assert!(json.contains("Hwc"));
        assert!(!json.contains("payload"));

        // Round-trip: the deserialised payload is empty (default).
        let back: EncodedObservation = serde_json::from_str(&json).unwrap();
        match back {
            EncodedObservation::RawU8Image {
                width,
                height,
                channels,
                layout,
                payload,
            } => {
                assert_eq!(width, 320);
                assert_eq!(height, 240);
                assert_eq!(channels, 3);
                assert_eq!(layout, ImageLayout::Hwc);
                assert!(payload.is_empty());
            }
            _ => panic!("expected RawU8Image"),
        }
    }

    #[test]
    fn encoded_observation_dict_roundtrip() {
        let mut inner = BTreeMap::new();
        inner.insert(
            "vec".to_string(),
            EncodedObservation::FlatF32 { data: vec![1.0] },
        );
        let enc = EncodedObservation::Dict { spaces: inner };
        let json = serde_json::to_string(&enc).unwrap();
        let back: EncodedObservation = serde_json::from_str(&json).unwrap();
        match back {
            EncodedObservation::Dict { spaces } => {
                assert_eq!(spaces.len(), 1);
                assert!(spaces.contains_key("vec"));
            }
            _ => panic!("expected Dict"),
        }
    }

    #[test]
    fn encode_observation_flat_for_continuous_space() {
        let data = [0.5_f32, -0.25, 1.0];
        let shape = [3_usize];
        let view = ObservationView::new(&data, SchemaDtype::F32, &shape);
        let space = ObservationSpace::Box {
            low: vec![-1.0; 3],
            high: vec![1.0; 3],
        };
        let (enc, payload) = encode_observation(&view, &space, true);
        assert!(payload.is_none());
        match enc {
            EncodedObservation::FlatF32 { data } => assert_eq!(data, vec![0.5, -0.25, 1.0]),
            _ => panic!("expected FlatF32"),
        }
    }

    #[test]
    fn encode_observation_raw_u8_for_image_space_with_binary() {
        // 2x2 RGB image, all white -> 12 bytes of 0xFF.
        let data = vec![1.0_f32; 2 * 2 * 3];
        let shape = [2_usize, 2, 3];
        let view = ObservationView::new(&data, SchemaDtype::F32, &shape);
        let space = ObservationSpace::Image {
            width: 2,
            height: 2,
            channels: 3,
        };
        let (enc, payload) = encode_observation(&view, &space, true);
        let bytes = payload.expect("binary payload expected");
        assert_eq!(bytes.len(), 2 * 2 * 3);
        assert!(bytes.iter().all(|&b| b == 255));
        match enc {
            EncodedObservation::RawU8Image {
                width,
                height,
                channels,
                layout,
                payload,
            } => {
                assert_eq!(width, 2);
                assert_eq!(height, 2);
                assert_eq!(channels, 3);
                assert_eq!(layout, ImageLayout::Hwc);
                assert!(payload.is_empty());
            }
            _ => panic!("expected RawU8Image"),
        }
    }

    #[test]
    fn encode_observation_image_space_without_binary_falls_back_to_flat() {
        // When binary=false, image envs serialise as FlatF32 (back-compat).
        let data = vec![0.5_f32; 2 * 2 * 3];
        let shape = [2_usize, 2, 3];
        let view = ObservationView::new(&data, SchemaDtype::F32, &shape);
        let space = ObservationSpace::Image {
            width: 2,
            height: 2,
            channels: 3,
        };
        let (enc, payload) = encode_observation(&view, &space, false);
        assert!(payload.is_none());
        assert!(matches!(enc, EncodedObservation::FlatF32 { .. }));
    }

    // ---- W7 PR2: BatchF32 / BatchRawU8Image variants ----

    #[test]
    fn encoded_observation_batch_f32_roundtrip() {
        let enc = EncodedObservation::BatchF32 {
            num_envs: 4,
            obs_dim: 17,
            payload: vec![0xAA, 0xBB], // must NOT appear in JSON
        };
        let json = serde_json::to_string(&enc).unwrap();
        assert!(json.contains("BatchF32"));
        assert!(json.contains("num_envs"));
        assert!(json.contains("obs_dim"));
        assert!(!json.contains("payload"));

        let back: EncodedObservation = serde_json::from_str(&json).unwrap();
        match back {
            EncodedObservation::BatchF32 {
                num_envs,
                obs_dim,
                payload,
            } => {
                assert_eq!(num_envs, 4);
                assert_eq!(obs_dim, 17);
                assert!(payload.is_empty());
            }
            _ => panic!("expected BatchF32"),
        }
    }

    #[test]
    fn encoded_observation_batch_raw_u8_roundtrip() {
        let enc = EncodedObservation::BatchRawU8Image {
            num_envs: 8,
            width: 64,
            height: 64,
            channels: 3,
            layout: ImageLayout::Hwc,
            payload: vec![0xCC],
        };
        let json = serde_json::to_string(&enc).unwrap();
        assert!(json.contains("BatchRawU8Image"));
        assert!(json.contains("num_envs"));
        assert!(json.contains("width"));
        assert!(json.contains("height"));
        assert!(json.contains("channels"));
        assert!(json.contains("layout"));
        assert!(json.contains("Hwc"));
        assert!(!json.contains("payload"));

        let back: EncodedObservation = serde_json::from_str(&json).unwrap();
        match back {
            EncodedObservation::BatchRawU8Image {
                num_envs,
                width,
                height,
                channels,
                layout,
                payload,
            } => {
                assert_eq!(num_envs, 8);
                assert_eq!(width, 64);
                assert_eq!(height, 64);
                assert_eq!(channels, 3);
                assert_eq!(layout, ImageLayout::Hwc);
                assert!(payload.is_empty());
            }
            _ => panic!("expected BatchRawU8Image"),
        }
    }

    #[test]
    fn encode_batch_step_binary_produces_flat_payload() {
        use clankers_core::types::{Observation, StepInfo};
        let result = BatchStepResult {
            observations: vec![
                Observation::new(vec![1.0, 2.0, 3.0]),
                Observation::new(vec![4.0, 5.0, 6.0]),
            ],
            rewards: vec![0.0, 0.0],
            terminated: vec![false, false],
            truncated: vec![false, false],
            infos: vec![StepInfo::default(), StepInfo::default()],
        };
        let (enc, payload) = encode_batch_step_binary(&result);
        match enc {
            EncodedObservation::BatchF32 {
                num_envs,
                obs_dim,
                payload: jpay,
            } => {
                assert_eq!(num_envs, 2);
                assert_eq!(obs_dim, 3);
                assert!(jpay.is_empty());
            }
            _ => panic!("expected BatchF32"),
        }
        // Header is 24 B + 6 f32 = 24 + 24 = 48 B.
        assert_eq!(payload.len(), 24 + 6 * 4);

        let (header, flat) = crate::binary_frame::decode_batch_f32(&payload).unwrap();
        assert_eq!(header.num_envs, 2);
        assert_eq!(header.dim, 3);
        assert_eq!(flat, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn encode_batch_reset_binary_produces_flat_payload() {
        use clankers_core::types::{Observation, ResetInfo};
        let result = BatchResetResult {
            observations: vec![
                Observation::new(vec![0.1, 0.2]),
                Observation::new(vec![0.3, 0.4]),
            ],
            infos: vec![ResetInfo::default(), ResetInfo::default()],
        };
        let (enc, payload) = encode_batch_reset_binary(&result);
        assert!(matches!(
            enc,
            EncodedObservation::BatchF32 {
                num_envs: 2,
                obs_dim: 2,
                ..
            },
        ));
        let (header, flat) = crate::binary_frame::decode_batch_f32(&payload).unwrap();
        assert_eq!(header.num_envs, 2);
        assert_eq!(header.dim, 2);
        assert_eq!(flat, &[0.1, 0.2, 0.3, 0.4]);
    }
}
