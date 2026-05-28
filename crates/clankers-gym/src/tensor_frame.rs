//! Generic binary tensor frame (P4.1).
//!
//! `CODE_QUALITY_REVIEW` § "Protocol Binary Tensor Path" / P4.1. The
//! existing [`crate::binary_frame`] surface specialises on two
//! batch-observation shapes (`BatchF32`, `BatchRawU8Image`). Each new
//! numeric payload (action batches, body-pose readback, contact
//! buffers) currently requires a new `kind` discriminant + a new
//! decoder branch.
//!
//! This module defines the generic shape Phase 4 will converge on:
//! one `TensorFrameHeader` carrying `(dtype, ndim, shape, layout)`
//! that subsumes every batch / image / arbitrary-tensor payload.
//!
//! # What's here
//!
//! - [`TensorDtype`] / [`TensorLayout`] discriminants.
//! - [`TensorFrameHeader`] — fixed 48-byte `#[repr(C)]` header with
//!   a `shape: [u32; 8]` array (so the header size is constant
//!   regardless of `ndim`; future rank-9+ tensors aren't planned).
//! - [`MAX_NDIM`] / [`TENSOR_HEADER_SIZE`] constants.
//!
//! # What is NOT here (yet)
//!
//! Encode / decode helpers (`encode_tensor`, `decode_tensor`) and
//! Python decoder are P4.2 / P4.3. The migration of `batch_f32` /
//! `batch_raw_u8` / image obs onto this header is P4.4.
//!
//! Wire format that we WILL converge on:
//!
//! ```text
//! +----------------+--------------------------+--------------+
//! | Length (4B LE) | 48-byte TensorFrameHeader| payload bytes|
//! +----------------+--------------------------+--------------+
//! ```
//!
//! Header layout (`#[repr(C)]`):
//!
//! ```text
//! offset 0  : version       (u32, LE) — currently 1
//! offset 4  : dtype         (u8)      — TensorDtype discriminant
//! offset 5  : ndim          (u8)      — 0..=MAX_NDIM
//! offset 6  : layout        (u8)      — TensorLayout discriminant
//! offset 7  : _pad          (1 × u8)  — align to 4-byte boundary
//! offset 8  : shape         (8 × u32, LE) — only first ndim entries
//!                                          are meaningful
//! offset 40 : payload_len   (u32, LE) — total payload byte count
//! offset 44 : _reserved     (1 × u32, LE) — currently 0
//! ```
//!
//! Total: 48 bytes. Pinned by both `const _: () = assert!(...)` and
//! the unit test `tensor_frame_header_size_is_48_bytes`.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Current tensor-frame format version. Bumped if header layout changes.
pub const TENSOR_FRAME_VERSION: u32 = 1;

/// Maximum tensor rank the header can carry.
///
/// Eight dimensions is more than enough for any robotics tensor we
/// ship; `NumPy` / `PyTorch` default-max is 64 but that would balloon
/// header size for no real-world benefit.
pub const MAX_NDIM: usize = 8;

/// Size of [`TensorFrameHeader`] in bytes. Pinned at the const-eval
/// barrier below and in the unit test
/// `tensor_frame_header_size_is_48_bytes`.
pub const TENSOR_HEADER_SIZE: usize = 48;

// ---------------------------------------------------------------------------
// TensorDtype
// ---------------------------------------------------------------------------

/// Element type of a tensor payload.
///
/// Encoded as a single `u8` in [`TensorFrameHeader::dtype`]. Discriminants
/// are explicit so the wire value is stable across Rust toolchain
/// versions.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDtype {
    /// 32-bit IEEE-754 float. Most common observation / action dtype.
    F32 = 0,
    /// 8-bit unsigned integer. Image observations.
    U8 = 1,
    /// 32-bit signed integer. Discrete observation indices.
    I32 = 2,
    /// 64-bit signed integer.
    I64 = 3,
    /// 64-bit IEEE-754 float. Higher-precision physics readback.
    F64 = 4,
}

impl TensorDtype {
    /// Byte size of one element.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F32 | Self::I32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    /// Decode a `u8` discriminant. Returns `None` for unknown values
    /// so callers can surface a typed error rather than silently
    /// promoting forward-compatibility unknowns.
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::U8),
            2 => Some(Self::I32),
            3 => Some(Self::I64),
            4 => Some(Self::F64),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// TensorLayout
// ---------------------------------------------------------------------------

/// Memory layout of a tensor payload.
///
/// Two values cover every shape we ship today; future layouts (NHWC
/// image batches with explicit alignment, blocked tile layouts) can
/// extend the discriminant without changing the header size.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorLayout {
    /// C-order / row-major. `NumPy` default.
    RowMajor = 0,
    /// Fortran-order / column-major.
    ColMajor = 1,
}

impl TensorLayout {
    /// Decode a `u8` discriminant. See [`TensorDtype::from_u8`].
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::RowMajor),
            1 => Some(Self::ColMajor),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// TensorFrameHeader
// ---------------------------------------------------------------------------

/// 48-byte `#[repr(C)]` header preceding an arbitrary tensor payload.
///
/// See the module docs for the byte layout. `shape[0..ndim]` carries
/// the tensor dimensions in layout order; remaining slots are zeroed.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(clippy::pub_underscore_fields)]
pub struct TensorFrameHeader {
    /// Frame format version. Must match [`TENSOR_FRAME_VERSION`].
    pub version: u32,
    /// Element dtype. Decoded via [`TensorDtype::from_u8`].
    pub dtype: u8,
    /// Number of dimensions. `0 <= ndim <= MAX_NDIM`.
    pub ndim: u8,
    /// Memory layout. Decoded via [`TensorLayout::from_u8`].
    pub layout: u8,
    /// Explicit padding to 4-byte align `shape`. Always `0`.
    pub _pad: u8,
    /// Tensor shape; only the first [`Self::ndim`] entries are
    /// meaningful. Remaining slots MUST be zero so the header
    /// hashes / equals reproducibly.
    pub shape: [u32; MAX_NDIM],
    /// Total payload byte count following the header.
    pub payload_len: u32,
    /// Forward-compat slot. Always zero today.
    pub _reserved: u32,
}

// Compile-time assertion: header must be exactly TENSOR_HEADER_SIZE.
#[allow(clippy::assertions_on_constants)]
const _: () = assert!(
    std::mem::size_of::<TensorFrameHeader>() == TENSOR_HEADER_SIZE,
    "TensorFrameHeader must be exactly TENSOR_HEADER_SIZE bytes wide",
);

impl TensorFrameHeader {
    /// Build a header from typed parameters. The shape array is
    /// truncated / zero-padded to [`MAX_NDIM`] slots.
    ///
    /// # Errors
    ///
    /// Returns [`TensorFrameError::TooManyDims`] when `shape.len() > MAX_NDIM`,
    /// or [`TensorFrameError::PayloadLenMismatch`] when `payload_len`
    /// disagrees with `shape.iter().product::<usize>() * dtype.size_bytes()`.
    pub fn new(
        dtype: TensorDtype,
        shape: &[u32],
        layout: TensorLayout,
        payload_len: u32,
    ) -> Result<Self, TensorFrameError> {
        if shape.len() > MAX_NDIM {
            return Err(TensorFrameError::TooManyDims {
                got: shape.len(),
                max: MAX_NDIM,
            });
        }
        let n_elements: u64 = shape.iter().map(|d| u64::from(*d)).product();
        let expected = n_elements * dtype.size_bytes() as u64;
        if u64::from(payload_len) != expected {
            return Err(TensorFrameError::PayloadLenMismatch {
                declared: payload_len,
                expected_for_shape: expected,
            });
        }
        let mut shape_arr = [0u32; MAX_NDIM];
        shape_arr[..shape.len()].copy_from_slice(shape);
        Ok(Self {
            version: TENSOR_FRAME_VERSION,
            dtype: dtype as u8,
            ndim: u8::try_from(shape.len()).expect("MAX_NDIM <= 255"),
            layout: layout as u8,
            _pad: 0,
            shape: shape_arr,
            payload_len,
            _reserved: 0,
        })
    }

    /// Decoded dtype, or `None` for an unknown discriminant (future
    /// peers should ignore unknown payloads rather than panic).
    #[must_use]
    pub const fn typed_dtype(&self) -> Option<TensorDtype> {
        TensorDtype::from_u8(self.dtype)
    }

    /// Decoded layout, or `None` for an unknown discriminant.
    #[must_use]
    pub const fn typed_layout(&self) -> Option<TensorLayout> {
        TensorLayout::from_u8(self.layout)
    }

    /// Borrow the active prefix of [`Self::shape`] (first `ndim`
    /// entries). Trailing zero slots are not included.
    #[must_use]
    pub fn active_shape(&self) -> &[u32] {
        &self.shape[..usize::from(self.ndim)]
    }
}

// ---------------------------------------------------------------------------
// TensorFrameError
// ---------------------------------------------------------------------------

/// Failure modes for [`TensorFrameHeader::new`] and future
/// encode / decode helpers (P4.2). Distinct error type so callers can
/// match on the specific divergence.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TensorFrameError {
    /// `shape.len() > MAX_NDIM`.
    #[error("tensor frame: shape has {got} dims, max is {max}")]
    TooManyDims {
        /// Caller-supplied ndim.
        got: usize,
        /// Header capacity ([`MAX_NDIM`]).
        max: usize,
    },
    /// `payload_len` disagrees with `shape × dtype.size_bytes()`.
    #[error(
        "tensor frame: declared payload_len {declared} != expected \
         {expected_for_shape} from shape × dtype size"
    )]
    PayloadLenMismatch {
        /// Caller-supplied byte count.
        declared: u32,
        /// Byte count expected from `shape × dtype.size_bytes()`.
        expected_for_shape: u64,
    },
    /// Decoder saw a frame whose version it doesn't recognise.
    #[error("tensor frame: unsupported version {got} (expected {TENSOR_FRAME_VERSION})")]
    UnsupportedVersion {
        /// Wire value.
        got: u32,
    },
    /// Decoder saw an unknown dtype discriminant.
    #[error("tensor frame: unknown dtype discriminant {got}")]
    UnknownDtype {
        /// Wire value.
        got: u8,
    },
    /// Decoder saw an unknown layout discriminant.
    #[error("tensor frame: unknown layout discriminant {got}")]
    UnknownLayout {
        /// Wire value.
        got: u8,
    },
}

// ---------------------------------------------------------------------------
// Encode / decode (P4.2)
// ---------------------------------------------------------------------------

/// Serialise `(header, payload)` into a single contiguous `Vec<u8>`.
///
/// Wire layout: 48-byte `TensorFrameHeader` (native LE on the wire — both
/// peers are little-endian) followed by `payload`. The header's
/// `payload_len` field must agree with `payload.len()` (otherwise this is
/// a `PayloadLenMismatch` — encoders should always construct the header
/// via [`TensorFrameHeader::new`] which validates the invariant).
///
/// # Errors
///
/// [`TensorFrameError::PayloadLenMismatch`] when the header's declared
/// `payload_len` does not match `payload.len()`.
pub fn encode_tensor(
    header: &TensorFrameHeader,
    payload: &[u8],
) -> Result<Vec<u8>, TensorFrameError> {
    if header.payload_len as usize != payload.len() {
        return Err(TensorFrameError::PayloadLenMismatch {
            declared: header.payload_len,
            expected_for_shape: payload.len() as u64,
        });
    }
    let mut out = Vec::with_capacity(TENSOR_HEADER_SIZE + payload.len());
    out.extend_from_slice(bytemuck::bytes_of(header));
    out.extend_from_slice(payload);
    Ok(out)
}

/// Decode a wire-format tensor frame into a `(header, payload)` pair.
///
/// Validates the version byte, the dtype/layout discriminants, the
/// payload length declared in the header against `buf.len() -
/// TENSOR_HEADER_SIZE`, and the `shape × dtype.size_bytes()`
/// consistency. The returned `payload` slice borrows from `buf`.
///
/// # Errors
///
/// - [`TensorFrameError::PayloadLenMismatch`] when `buf` is shorter than
///   the header or the header's declared payload size does not match the
///   trailing byte count.
/// - [`TensorFrameError::UnsupportedVersion`] when `header.version`
///   isn't [`TENSOR_FRAME_VERSION`].
/// - [`TensorFrameError::UnknownDtype`] / [`TensorFrameError::UnknownLayout`]
///   for unrecognised discriminants.
/// - [`TensorFrameError::TooManyDims`] when `header.ndim > MAX_NDIM`.
pub fn decode_tensor(buf: &[u8]) -> Result<(TensorFrameHeader, &[u8]), TensorFrameError> {
    if buf.len() < TENSOR_HEADER_SIZE {
        return Err(TensorFrameError::PayloadLenMismatch {
            declared: 0,
            expected_for_shape: buf.len() as u64,
        });
    }
    let (head_bytes, payload) = buf.split_at(TENSOR_HEADER_SIZE);
    let header: TensorFrameHeader = *bytemuck::from_bytes(head_bytes);

    if header.version != TENSOR_FRAME_VERSION {
        return Err(TensorFrameError::UnsupportedVersion {
            got: header.version,
        });
    }
    let dtype = TensorDtype::from_u8(header.dtype)
        .ok_or(TensorFrameError::UnknownDtype { got: header.dtype })?;
    let _layout = TensorLayout::from_u8(header.layout)
        .ok_or(TensorFrameError::UnknownLayout { got: header.layout })?;
    if usize::from(header.ndim) > MAX_NDIM {
        return Err(TensorFrameError::TooManyDims {
            got: usize::from(header.ndim),
            max: MAX_NDIM,
        });
    }
    if header.payload_len as usize != payload.len() {
        return Err(TensorFrameError::PayloadLenMismatch {
            declared: header.payload_len,
            expected_for_shape: payload.len() as u64,
        });
    }
    // Cross-check the payload size against shape × dtype.
    let n_elements: u64 = header
        .active_shape()
        .iter()
        .map(|d| u64::from(*d))
        .product();
    let expected = n_elements * dtype.size_bytes() as u64;
    if u64::from(header.payload_len) != expected {
        return Err(TensorFrameError::PayloadLenMismatch {
            declared: header.payload_len,
            expected_for_shape: expected,
        });
    }
    Ok((header, payload))
}

// ---------------------------------------------------------------------------
// Batch / image encoders on top of the shared header (P4.4)
// ---------------------------------------------------------------------------

/// Encode a flat `[num_envs * obs_dim]` `f32` batch as a tensor frame
/// of shape `(num_envs, obs_dim)`.
///
/// `CODE_QUALITY_REVIEW` § Phase 4.4 — replacement encoder for the legacy
/// `binary_frame::encode_batch_f32`. Produces a single contiguous
/// buffer with the 48-byte [`TensorFrameHeader`] in front; the existing
/// `write_binary_frame` transport delivers it unchanged.
///
/// The legacy path stays in `binary_frame.rs` for backward
/// compatibility with peers that don't speak the tensor format; the
/// `Capabilities::TENSOR_BATCH` flag (added in a follow-up commit)
/// will let producers / consumers negotiate.
///
/// # Errors
///
/// Propagates [`TensorFrameError::PayloadLenMismatch`] when the input
/// `data` length is not `num_envs * obs_dim`.
pub fn encode_batch_f32_as_tensor(
    num_envs: u32,
    obs_dim: u32,
    data: &[f32],
) -> Result<Vec<u8>, TensorFrameError> {
    let expected = u64::from(num_envs) * u64::from(obs_dim);
    if data.len() as u64 != expected {
        return Err(TensorFrameError::PayloadLenMismatch {
            declared: 0,
            expected_for_shape: expected * 4,
        });
    }
    let payload_bytes: &[u8] = bytemuck::cast_slice(data);
    let payload_len =
        u32::try_from(payload_bytes.len()).map_err(|_| TensorFrameError::PayloadLenMismatch {
            declared: 0,
            expected_for_shape: payload_bytes.len() as u64,
        })?;
    let header = TensorFrameHeader::new(
        TensorDtype::F32,
        &[num_envs, obs_dim],
        TensorLayout::RowMajor,
        payload_len,
    )?;
    encode_tensor(&header, payload_bytes)
}

/// Encode a flat `[num_envs * width * height * channels]` `u8` batch
/// as a tensor frame of shape `(num_envs, height, width, channels)`.
///
/// HWC layout — the Bevy / cosmos pipeline already produces tiles in
/// height-major order. The shape order matches `NumPy` / TensorFlow
/// `NHWC`; `PyTorch` consumers reorder with `transpose(0, 3, 1, 2)`.
///
/// # Errors
///
/// Propagates [`TensorFrameError::PayloadLenMismatch`] when the input
/// `data` length is not `num_envs * width * height * channels`.
pub fn encode_batch_raw_u8_as_tensor(
    num_envs: u32,
    width: u32,
    height: u32,
    channels: u32,
    data: &[u8],
) -> Result<Vec<u8>, TensorFrameError> {
    let expected = u64::from(num_envs) * u64::from(width) * u64::from(height) * u64::from(channels);
    if data.len() as u64 != expected {
        return Err(TensorFrameError::PayloadLenMismatch {
            declared: 0,
            expected_for_shape: expected,
        });
    }
    let payload_len =
        u32::try_from(data.len()).map_err(|_| TensorFrameError::PayloadLenMismatch {
            declared: 0,
            expected_for_shape: data.len() as u64,
        })?;
    let header = TensorFrameHeader::new(
        TensorDtype::U8,
        &[num_envs, height, width, channels],
        TensorLayout::RowMajor,
        payload_len,
    )?;
    encode_tensor(&header, data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    #[test]
    fn tensor_frame_header_size_is_48_bytes() {
        // Gate item: header must be byte-pinned at 48 B so Python
        // and Rust agree on offsets without sharing a code path.
        assert_eq!(std::mem::size_of::<TensorFrameHeader>(), TENSOR_HEADER_SIZE);
    }

    #[test]
    fn dtype_size_bytes_matches_ieee754() {
        assert_eq!(TensorDtype::F32.size_bytes(), 4);
        assert_eq!(TensorDtype::U8.size_bytes(), 1);
        assert_eq!(TensorDtype::I32.size_bytes(), 4);
        assert_eq!(TensorDtype::I64.size_bytes(), 8);
        assert_eq!(TensorDtype::F64.size_bytes(), 8);
    }

    #[test]
    fn from_u8_round_trips_known_discriminants() {
        for d in [
            TensorDtype::F32,
            TensorDtype::U8,
            TensorDtype::I32,
            TensorDtype::I64,
            TensorDtype::F64,
        ] {
            let back = TensorDtype::from_u8(d as u8).unwrap();
            assert_eq!(d, back);
        }
        assert!(TensorDtype::from_u8(0xFF).is_none());
        assert!(TensorLayout::from_u8(0xFF).is_none());
    }

    #[test]
    fn new_rejects_too_many_dims() {
        let nine = vec![1u32; MAX_NDIM + 1];
        let err =
            TensorFrameHeader::new(TensorDtype::F32, &nine, TensorLayout::RowMajor, 0).unwrap_err();
        assert_eq!(
            err,
            TensorFrameError::TooManyDims {
                got: MAX_NDIM + 1,
                max: MAX_NDIM,
            }
        );
    }

    #[test]
    fn new_rejects_payload_len_mismatch() {
        // shape (2, 3), dtype f32 -> expected 24 bytes; declared 8 -> mismatch.
        let err = TensorFrameHeader::new(TensorDtype::F32, &[2, 3], TensorLayout::RowMajor, 8)
            .unwrap_err();
        assert_eq!(
            err,
            TensorFrameError::PayloadLenMismatch {
                declared: 8,
                expected_for_shape: 24,
            }
        );
    }

    #[test]
    fn new_accepts_consistent_shape_and_payload() {
        // shape (4, 8) f32 -> 4 * 8 * 4 = 128 bytes.
        let header =
            TensorFrameHeader::new(TensorDtype::F32, &[4, 8], TensorLayout::RowMajor, 128).unwrap();
        assert_eq!(header.version, TENSOR_FRAME_VERSION);
        assert_eq!(header.ndim, 2);
        assert_eq!(header.active_shape(), &[4, 8]);
        assert_eq!(header.payload_len, 128);
        assert_eq!(header.typed_dtype(), Some(TensorDtype::F32));
        assert_eq!(header.typed_layout(), Some(TensorLayout::RowMajor));
    }

    // -------- encode / decode (P4.2) --------

    #[test]
    fn encode_then_decode_roundtrips_f32_tensor() {
        // (2, 3) f32 → 24 bytes.
        let elements: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let payload = bytemuck::cast_slice::<f32, u8>(&elements);
        let header =
            TensorFrameHeader::new(TensorDtype::F32, &[2, 3], TensorLayout::RowMajor, 24).unwrap();
        let buf = encode_tensor(&header, payload).unwrap();
        assert_eq!(buf.len(), TENSOR_HEADER_SIZE + 24);

        let (back_header, back_payload) = decode_tensor(&buf).unwrap();
        assert_eq!(back_header, header);
        let back: &[f32] = bytemuck::cast_slice(back_payload);
        assert_eq!(back, &elements);
    }

    #[test]
    fn encode_rejects_payload_size_mismatch() {
        // Header claims 24 bytes; payload is 8.
        let header =
            TensorFrameHeader::new(TensorDtype::F32, &[2, 3], TensorLayout::RowMajor, 24).unwrap();
        let payload = [0u8; 8];
        let err = encode_tensor(&header, &payload).unwrap_err();
        assert_eq!(
            err,
            TensorFrameError::PayloadLenMismatch {
                declared: 24,
                expected_for_shape: 8,
            }
        );
    }

    #[test]
    fn decode_rejects_truncated_buffer() {
        let buf = vec![0u8; TENSOR_HEADER_SIZE - 1];
        let err = decode_tensor(&buf).unwrap_err();
        assert!(matches!(err, TensorFrameError::PayloadLenMismatch { .. }));
    }

    #[test]
    fn decode_rejects_unknown_version() {
        let mut header =
            TensorFrameHeader::new(TensorDtype::F32, &[2, 3], TensorLayout::RowMajor, 24).unwrap();
        header.version = 99;
        let payload = vec![0u8; 24];
        let mut buf = Vec::new();
        buf.extend_from_slice(bytemuck::bytes_of(&header));
        buf.extend_from_slice(&payload);
        let err = decode_tensor(&buf).unwrap_err();
        assert_eq!(err, TensorFrameError::UnsupportedVersion { got: 99 });
    }

    #[test]
    fn decode_rejects_unknown_dtype() {
        let mut header =
            TensorFrameHeader::new(TensorDtype::F32, &[2, 3], TensorLayout::RowMajor, 24).unwrap();
        header.dtype = 250;
        let payload = vec![0u8; 24];
        let mut buf = Vec::new();
        buf.extend_from_slice(bytemuck::bytes_of(&header));
        buf.extend_from_slice(&payload);
        let err = decode_tensor(&buf).unwrap_err();
        assert_eq!(err, TensorFrameError::UnknownDtype { got: 250 });
    }

    #[test]
    fn decode_rejects_payload_length_disagreement() {
        let header =
            TensorFrameHeader::new(TensorDtype::F32, &[2, 3], TensorLayout::RowMajor, 24).unwrap();
        // Buffer carries only 16 trailing bytes, header says 24.
        let mut buf = Vec::new();
        buf.extend_from_slice(bytemuck::bytes_of(&header));
        buf.extend_from_slice(&[0u8; 16]);
        let err = decode_tensor(&buf).unwrap_err();
        assert_eq!(
            err,
            TensorFrameError::PayloadLenMismatch {
                declared: 24,
                expected_for_shape: 16,
            }
        );
    }

    #[test]
    fn encode_decode_u8_image_roundtrips() {
        // 32×32×3 uint8 image → 3072 bytes.
        let n = 32 * 32 * 3;
        let payload: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let header = TensorFrameHeader::new(
            TensorDtype::U8,
            &[32, 32, 3],
            TensorLayout::RowMajor,
            n as u32,
        )
        .unwrap();
        let buf = encode_tensor(&header, &payload).unwrap();
        let (back_h, back_p) = decode_tensor(&buf).unwrap();
        assert_eq!(back_h.active_shape(), &[32, 32, 3]);
        assert_eq!(back_p, payload.as_slice());
    }

    // -------- P4.4: batch / image migration --------

    #[test]
    fn encode_batch_f32_as_tensor_roundtrips_via_decoder() {
        let num_envs = 4u32;
        let obs_dim = 3u32;
        let data: Vec<f32> = (0..(num_envs * obs_dim)).map(|i| i as f32 * 0.5).collect();
        let buf = encode_batch_f32_as_tensor(num_envs, obs_dim, &data).unwrap();
        let (header, payload) = decode_tensor(&buf).unwrap();
        assert_eq!(header.active_shape(), &[num_envs, obs_dim]);
        assert_eq!(header.typed_dtype(), Some(TensorDtype::F32));
        let back: &[f32] = bytemuck::cast_slice(payload);
        assert_eq!(back, data.as_slice());
    }

    #[test]
    fn encode_batch_f32_rejects_length_mismatch() {
        let err = encode_batch_f32_as_tensor(2, 3, &[1.0; 5]).unwrap_err();
        assert!(matches!(err, TensorFrameError::PayloadLenMismatch { .. }));
    }

    #[test]
    fn encode_batch_raw_u8_as_tensor_roundtrips_via_decoder() {
        let num_envs = 2u32;
        let width = 4u32;
        let height = 3u32;
        let channels = 3u32;
        let n = (num_envs * width * height * channels) as usize;
        let data: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let buf = encode_batch_raw_u8_as_tensor(num_envs, width, height, channels, &data).unwrap();
        let (header, payload) = decode_tensor(&buf).unwrap();
        // NHWC ordering: shape is (num_envs, height, width, channels).
        assert_eq!(header.active_shape(), &[num_envs, height, width, channels]);
        assert_eq!(header.typed_dtype(), Some(TensorDtype::U8));
        assert_eq!(payload, data.as_slice());
    }

    #[test]
    fn encode_batch_raw_u8_rejects_length_mismatch() {
        let err = encode_batch_raw_u8_as_tensor(2, 4, 3, 3, &[0u8; 10]).unwrap_err();
        assert!(matches!(err, TensorFrameError::PayloadLenMismatch { .. }));
    }

    #[test]
    fn new_zero_pads_shape_tail() {
        // Only the first ndim slots may be non-zero so the header
        // hashes / equals reproducibly across producers that may
        // leave junk in the tail.
        let header =
            TensorFrameHeader::new(TensorDtype::U8, &[64, 64, 3], TensorLayout::RowMajor, 12288)
                .unwrap();
        assert_eq!(header.active_shape(), &[64, 64, 3]);
        for &v in &header.shape[3..] {
            assert_eq!(v, 0, "shape tail must be zero-padded");
        }
    }
}
