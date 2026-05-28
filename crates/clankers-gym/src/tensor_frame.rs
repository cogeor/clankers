//! Generic binary tensor frame (P4.1).
//!
//! CODE_QUALITY_REVIEW § "Protocol Binary Tensor Path" / P4.1. The
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

/// Maximum tensor rank the header can carry. Eight dimensions is more
/// than enough for any robotics tensor we ship; NumPy / PyTorch
/// default-max is 64 but that would balloon header size for no
/// real-world benefit.
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
    /// C-order / row-major. NumPy default.
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
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
