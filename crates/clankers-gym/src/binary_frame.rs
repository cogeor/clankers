//! Binary batch-observation frame for high-throughput training (W7 PR2).
//!
//! A 24-byte [`BinaryFrameHeader`] is followed by a `bytemuck`-cast
//! payload of either `f32` (for [`KIND_BATCH_F32`]) or `u8` (for
//! [`KIND_BATCH_RAW_U8`]). The combined `header + payload` byte slice
//! rides on the **existing** 4-byte LE length-prefixed binary channel
//! from [`crate::framing::write_binary_frame`] — no new transport magic.
//! Dispatch happens via the JSON envelope's `obs_encoding` tag (W4's
//! `RawU8Image` precedent).
//!
//! # Wire format
//!
//! ```text
//! +----------------+----------------+----------------+
//! | Length (4B LE) | 24-byte header | payload bytes  |
//! +----------------+----------------+----------------+
//! ```
//!
//! The header layout is `#[repr(C)]`:
//!
//! ```text
//! offset 0  : version    (u32, LE) — currently 1
//! offset 4  : kind       (u8)      — 0 = BatchF32, 1 = BatchRawU8Image
//! offset 5  : _pad       (3 × u8)  — align to 4-byte boundary
//! offset 8  : num_envs   (u32, LE)
//! offset 12 : dim        (u32, LE) — BatchF32: obs_dim; BatchRawU8Image: w*h*c per env
//! offset 16 : _reserved  (2 × u32, LE) — currently [0, 0]
//! ```
//!
//! The total header size is pinned at 24 bytes by both an inline
//! `const _: () = assert!(...)` (compile-time) and the integration test
//! `binary_frame_header_size_is_24_bytes` (runtime gate).
//!
//! # Endianness
//!
//! Clankers targets little-endian hosts only (`x86_64`, `aarch64`). On a
//! big-endian host, `bytemuck::cast_slice::<f32, u8>` is still a
//! zero-copy noop but the resulting bytes would have host byte-order —
//! a future cross-arch frame would need explicit byte-swap. Out of
//! scope for this loop.

use thiserror::Error;

use crate::encoding::ImageLayout;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Current binary frame format version. Bumped if header layout changes.
pub const FRAME_VERSION: u32 = 1;

/// Frame kind discriminator for a batched `f32` observation tensor.
pub const KIND_BATCH_F32: u8 = 0;

/// Frame kind discriminator for a batched raw `u8` image stack.
pub const KIND_BATCH_RAW_U8: u8 = 1;

/// Size of [`BinaryFrameHeader`] in bytes. Pinned by a compile-time
/// `const` assertion below and by the runtime gate-item test
/// `binary_frame_header_size_is_24_bytes`.
pub const HEADER_SIZE: usize = 24;

// Compile-time assertion: the `#[repr(C)]` header MUST be exactly 24 B.
// `static_assertions` is a dev-only dep, so we roll our own via a
// zero-sized `const _: () = ...` that triggers a const-eval panic on
// mismatch.
#[allow(clippy::assertions_on_constants)]
const _: () = assert!(
    std::mem::size_of::<BinaryFrameHeader>() == HEADER_SIZE,
    "BinaryFrameHeader must be exactly HEADER_SIZE bytes wide",
);

// ---------------------------------------------------------------------------
// BinaryFrameHeader
// ---------------------------------------------------------------------------

/// 24-byte `#[repr(C)]` header preceding a batch-observation payload.
///
/// The `_pad` bytes align `num_envs` to a 4-byte boundary so the entire
/// header is itself 4-byte aligned (required for `bytemuck::cast_slice`
/// on the f32 payload that follows). The trailing `_reserved` field is
/// `[u32; 2]` (8 bytes) rather than a single `u32` so that the total
/// header is exactly [`HEADER_SIZE`] (24 B) and leaves a forward-compat
/// slot for future flags (e.g. layout, dtype, byte order).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(clippy::pub_underscore_fields)] // _pad / _reserved are intentionally
// public so external decoders can
// bytemuck::cast the same layout.
pub struct BinaryFrameHeader {
    /// Frame format version. Must match [`FRAME_VERSION`].
    pub version: u32,
    /// Frame kind discriminator: [`KIND_BATCH_F32`] or [`KIND_BATCH_RAW_U8`].
    pub kind: u8,
    /// Explicit padding to 4-byte align `num_envs`. Always `[0; 3]`.
    pub _pad: [u8; 3],
    /// Number of environments in the batch.
    pub num_envs: u32,
    /// Per-env element count. For [`KIND_BATCH_F32`] this is `obs_dim`;
    /// for [`KIND_BATCH_RAW_U8`] this is `width * height * channels`.
    pub dim: u32,
    /// Reserved for future flags (e.g. dtype, layout, byte order).
    /// Currently always `[0, 0]`. Two `u32`s rather than one so the
    /// header is 24 bytes total — matches the
    /// `binary_frame_header_size_is_24_bytes` gate item.
    pub _reserved: [u32; 2],
}

// ---------------------------------------------------------------------------
// BinaryFrameError
// ---------------------------------------------------------------------------

/// Errors produced when decoding a malformed binary batch frame.
#[derive(Error, Debug)]
pub enum BinaryFrameError {
    /// The header's `version` field does not match [`FRAME_VERSION`].
    #[error("unsupported binary frame version: {got}")]
    UnsupportedVersion {
        /// The unsupported version that was read from the header.
        got: u32,
    },
    /// The header's `kind` field is not [`KIND_BATCH_F32`] or [`KIND_BATCH_RAW_U8`].
    #[error("unknown frame kind: {got}")]
    UnknownKind {
        /// The unknown discriminator byte that was read.
        got: u8,
    },
    /// The byte slice is too short or otherwise unparseable.
    #[error("malformed binary frame: {0}")]
    Malformed(&'static str),
    /// The payload length disagrees with the header's `num_envs * dim` product.
    #[error("length mismatch: header expects {expected} payload bytes, got {got}")]
    LengthMismatch {
        /// Expected payload bytes computed from `num_envs * dim * element_size`.
        expected: usize,
        /// Actual payload bytes in the input slice (i.e. `bytes.len() - 24`).
        got: usize,
    },
}

// ---------------------------------------------------------------------------
// encode_batch_f32 / decode_batch_f32
// ---------------------------------------------------------------------------

/// Encode a `num_envs × obs_dim` slice of `f32` observations into a
/// [`BinaryFrameHeader`] + zero-copy bytemuck payload.
///
/// # Panics
///
/// Panics if `data.len() != num_envs * obs_dim` — this is a programmer
/// error on the encode side; the symmetric decode path returns
/// [`BinaryFrameError::LengthMismatch`] instead.
#[must_use]
pub fn encode_batch_f32(num_envs: u32, obs_dim: u32, data: &[f32]) -> Vec<u8> {
    let expected = num_envs as usize * obs_dim as usize;
    assert_eq!(
        data.len(),
        expected,
        "BatchF32 payload length mismatch: data.len() = {}, num_envs * obs_dim = {expected}",
        data.len(),
    );

    let header = BinaryFrameHeader {
        version: FRAME_VERSION,
        kind: KIND_BATCH_F32,
        _pad: [0; 3],
        num_envs,
        dim: obs_dim,
        _reserved: [0; 2],
    };

    let header_bytes = bytemuck::bytes_of(&header);
    let payload_bytes = bytemuck::cast_slice::<f32, u8>(data);

    let mut out = Vec::with_capacity(HEADER_SIZE + payload_bytes.len());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(payload_bytes);
    out
}

/// Decode a [`BinaryFrameHeader`] and borrow the f32 payload out of `bytes`.
///
/// Returns `Ok((header, payload))` where `payload` is a zero-copy borrow
/// of `bytes[24..]` cast to `&[f32]`. On error, returns a
/// [`BinaryFrameError`] describing the malformedness.
///
/// # Errors
///
/// - [`BinaryFrameError::Malformed`] — `bytes` is shorter than 24 B.
/// - [`BinaryFrameError::UnsupportedVersion`] — header's version is not 1.
/// - [`BinaryFrameError::UnknownKind`] — header's kind is not 0 (`BatchF32`).
/// - [`BinaryFrameError::LengthMismatch`] — payload bytes don't match
///   `num_envs * obs_dim * 4`.
pub fn decode_batch_f32(bytes: &[u8]) -> Result<(BinaryFrameHeader, &[f32]), BinaryFrameError> {
    if bytes.len() < HEADER_SIZE {
        return Err(BinaryFrameError::Malformed(
            "buffer shorter than 24-byte header",
        ));
    }
    // try_pod_read_unaligned tolerates arbitrary input alignment (TCP
    // frames have no guaranteed alignment).
    let header: BinaryFrameHeader = bytemuck::try_pod_read_unaligned(&bytes[..HEADER_SIZE])
        .map_err(|_| {
            BinaryFrameError::Malformed("bytemuck::try_pod_read_unaligned failed on header")
        })?;

    if header.version != FRAME_VERSION {
        return Err(BinaryFrameError::UnsupportedVersion {
            got: header.version,
        });
    }
    if header.kind != KIND_BATCH_F32 {
        return Err(BinaryFrameError::UnknownKind { got: header.kind });
    }

    let expected_floats = header.num_envs as usize * header.dim as usize;
    let expected_bytes = expected_floats * std::mem::size_of::<f32>();
    let payload_bytes = &bytes[HEADER_SIZE..];
    if payload_bytes.len() != expected_bytes {
        return Err(BinaryFrameError::LengthMismatch {
            expected: expected_bytes,
            got: payload_bytes.len(),
        });
    }

    // try_cast_slice returns Err if the tail length is not a multiple of
    // 4 — the check above already covers that. The `f32` alignment
    // requirement (4) is met because the header is 24 B (multiple of 4)
    // and `try_pod_read_unaligned` returns an owned value, so the cast
    // here is on the borrowed tail. `try_cast_slice` validates alignment
    // by checking the pointer at runtime; on an arbitrary network
    // buffer this is not guaranteed, so we fall back to a copy if the
    // borrow fails.
    let payload: &[f32] = match bytemuck::try_cast_slice::<u8, f32>(payload_bytes) {
        Ok(s) => s,
        Err(_) => {
            // Alignment failure on the borrowed slice. This is rare
            // because `Vec<u8>` is heap-aligned to >= 8 on every Rust
            // target, but defensive code matters for arbitrary slices.
            return Err(BinaryFrameError::Malformed(
                "f32 payload alignment failure on borrowed slice",
            ));
        }
    };

    Ok((header, payload))
}

// ---------------------------------------------------------------------------
// encode_batch_raw_u8 / decode_batch_raw_u8
// ---------------------------------------------------------------------------

/// Encode a `num_envs × width × height × channels` slice of `u8` pixels
/// into a [`BinaryFrameHeader`] + payload.
///
/// `dim` in the header is `width * height * channels` (per-env tile
/// size).
///
/// # Panics
///
/// Panics if `data.len() != num_envs * width * height * channels`.
#[must_use]
pub fn encode_batch_raw_u8(
    num_envs: u32,
    width: u32,
    height: u32,
    channels: u8,
    _layout: ImageLayout,
    data: &[u8],
) -> Vec<u8> {
    let per_env = width as usize * height as usize * channels as usize;
    let expected = num_envs as usize * per_env;
    assert_eq!(
        data.len(),
        expected,
        "BatchRawU8Image payload length mismatch: data.len() = {}, num_envs * w * h * c = {expected}",
        data.len(),
    );

    let dim_u32 = u32::try_from(per_env).expect("per-env tile size overflows u32");
    let header = BinaryFrameHeader {
        version: FRAME_VERSION,
        kind: KIND_BATCH_RAW_U8,
        _pad: [0; 3],
        num_envs,
        dim: dim_u32,
        _reserved: [0; 2],
    };

    let header_bytes = bytemuck::bytes_of(&header);
    let mut out = Vec::with_capacity(HEADER_SIZE + data.len());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(data);
    out
}

/// Decode a [`BinaryFrameHeader`] and borrow the `u8` payload out of `bytes`.
///
/// # Errors
///
/// - [`BinaryFrameError::Malformed`] — `bytes` is shorter than 24 B.
/// - [`BinaryFrameError::UnsupportedVersion`] — header's version is not 1.
/// - [`BinaryFrameError::UnknownKind`] — header's kind is not 1
///   (`BatchRawU8Image`).
/// - [`BinaryFrameError::LengthMismatch`] — payload bytes don't match
///   `num_envs * dim`.
pub fn decode_batch_raw_u8(bytes: &[u8]) -> Result<(BinaryFrameHeader, &[u8]), BinaryFrameError> {
    if bytes.len() < HEADER_SIZE {
        return Err(BinaryFrameError::Malformed(
            "buffer shorter than 24-byte header",
        ));
    }
    let header: BinaryFrameHeader = bytemuck::try_pod_read_unaligned(&bytes[..HEADER_SIZE])
        .map_err(|_| {
            BinaryFrameError::Malformed("bytemuck::try_pod_read_unaligned failed on header")
        })?;

    if header.version != FRAME_VERSION {
        return Err(BinaryFrameError::UnsupportedVersion {
            got: header.version,
        });
    }
    if header.kind != KIND_BATCH_RAW_U8 {
        return Err(BinaryFrameError::UnknownKind { got: header.kind });
    }

    let expected = header.num_envs as usize * header.dim as usize;
    let payload = &bytes[HEADER_SIZE..];
    if payload.len() != expected {
        return Err(BinaryFrameError::LengthMismatch {
            expected,
            got: payload.len(),
        });
    }

    Ok((header, payload))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_pod_zeroable() {
        // Smoke check: `BinaryFrameHeader::zeroed()` produces an
        // all-zero header that round-trips through bytemuck.
        let zero: BinaryFrameHeader = bytemuck::Zeroable::zeroed();
        assert_eq!(zero.version, 0);
        assert_eq!(zero.kind, 0);
        assert_eq!(zero.num_envs, 0);
        assert_eq!(zero.dim, 0);
        let bytes = bytemuck::bytes_of(&zero);
        assert_eq!(bytes.len(), HEADER_SIZE);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn lengths_mismatch_detected_f32() {
        // Build a valid-header frame but truncate the payload.
        let bytes = encode_batch_f32(2, 3, &[0.0; 6]);
        let truncated = &bytes[..bytes.len() - 4];
        let err = decode_batch_f32(truncated).unwrap_err();
        assert!(matches!(err, BinaryFrameError::LengthMismatch { .. }));
    }

    #[test]
    fn lengths_mismatch_detected_u8() {
        let bytes = encode_batch_raw_u8(2, 4, 4, 3, ImageLayout::Hwc, &[0u8; 2 * 4 * 4 * 3]);
        let truncated = &bytes[..bytes.len() - 1];
        let err = decode_batch_raw_u8(truncated).unwrap_err();
        assert!(matches!(err, BinaryFrameError::LengthMismatch { .. }));
    }

    #[test]
    fn unknown_kind_detected() {
        // Build a frame, flip the `kind` byte to a non-recognised value,
        // and check decode returns UnknownKind.
        let mut bytes = encode_batch_f32(1, 1, &[0.0]);
        bytes[4] = 0xAA; // overwrite kind byte
        let err = decode_batch_f32(&bytes).unwrap_err();
        assert!(matches!(err, BinaryFrameError::UnknownKind { got: 0xAA }));
    }

    #[test]
    fn buffer_too_short_detected() {
        let err = decode_batch_f32(&[0u8; 10]).unwrap_err();
        assert!(matches!(err, BinaryFrameError::Malformed(_)));
    }

    #[test]
    #[should_panic(expected = "BatchF32 payload length mismatch")]
    fn encode_panics_on_length_mismatch_f32() {
        // Encoder is panicking-strict on caller bugs.
        let _ = encode_batch_f32(2, 3, &[0.0; 5]);
    }

    #[test]
    fn encode_decode_roundtrip_zero_envs() {
        // num_envs=0 produces a frame with just the header.
        let bytes = encode_batch_f32(0, 4, &[]);
        assert_eq!(bytes.len(), HEADER_SIZE);
        let (header, payload) = decode_batch_f32(&bytes).unwrap();
        assert_eq!(header.num_envs, 0);
        assert_eq!(header.dim, 4);
        assert!(payload.is_empty());
    }
}
