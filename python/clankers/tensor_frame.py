"""Python decoder for the Rust ``TensorFrameHeader`` wire format (P4.3).

CODE_QUALITY_REVIEW § Phase 4.3 — pair Python decoder for the Rust
``crates/clankers-gym/src/tensor_frame.rs`` 48-byte header. Each new
numeric payload (action batches, body-pose readback, contact buffers,
batched images) ships through this header instead of growing a new
``binary_frame`` discriminant on either side of the wire.

Wire layout (must stay byte-for-byte identical to the Rust struct)::

    offset 0  : version       (u32, LE)
    offset 4  : dtype         (u8)
    offset 5  : ndim          (u8)
    offset 6  : layout        (u8)
    offset 7  : _pad          (u8)
    offset 8  : shape         (8 x u32, LE)
    offset 40 : payload_len   (u32, LE)
    offset 44 : _reserved     (u32, LE)

Total: 48 bytes. Pinned by ``TENSOR_HEADER_SIZE`` and the round-trip
unit tests in ``python/tests/test_tensor_frame.py``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

TENSOR_FRAME_VERSION: Final[int] = 1
TENSOR_HEADER_SIZE: Final[int] = 48
MAX_NDIM: Final[int] = 8


class TensorDtype(IntEnum):
    """Element type. Discriminant matches the Rust ``TensorDtype`` enum."""

    F32 = 0
    U8 = 1
    I32 = 2
    I64 = 3
    F64 = 4

    @property
    def size_bytes(self) -> int:
        return _DTYPE_SIZE_BYTES[self]

    @property
    def numpy_dtype(self) -> np.dtype:
        return _DTYPE_NUMPY[self]


_DTYPE_SIZE_BYTES: Final[dict[TensorDtype, int]] = {
    TensorDtype.F32: 4,
    TensorDtype.U8: 1,
    TensorDtype.I32: 4,
    TensorDtype.I64: 8,
    TensorDtype.F64: 8,
}

_DTYPE_NUMPY: Final[dict[TensorDtype, np.dtype]] = {
    TensorDtype.F32: np.dtype("<f4"),
    TensorDtype.U8: np.dtype("u1"),
    TensorDtype.I32: np.dtype("<i4"),
    TensorDtype.I64: np.dtype("<i8"),
    TensorDtype.F64: np.dtype("<f8"),
}


class TensorLayout(IntEnum):
    """Memory layout. Matches the Rust ``TensorLayout`` enum."""

    ROW_MAJOR = 0
    COL_MAJOR = 1


class TensorFrameError(ValueError):
    """Base class for tensor-frame decoding failures."""


class UnsupportedVersion(TensorFrameError):
    """Header advertises a version we don't decode."""


class UnknownDtype(TensorFrameError):
    """Header ``dtype`` byte is not a known ``TensorDtype`` discriminant."""


class UnknownLayout(TensorFrameError):
    """Header ``layout`` byte is not a known ``TensorLayout`` discriminant."""


class PayloadLenMismatch(TensorFrameError):
    """Declared payload length disagrees with the buffer or shape * dtype."""


class TooManyDims(TensorFrameError):
    """Header ``ndim`` exceeds ``MAX_NDIM``."""


@dataclass(frozen=True, slots=True)
class TensorFrameHeader:
    """Decoded header fields. Mirrors the Rust struct."""

    version: int
    dtype: TensorDtype
    ndim: int
    layout: TensorLayout
    shape: tuple[int, ...]
    payload_len: int

    @property
    def active_shape(self) -> tuple[int, ...]:
        return self.shape[: self.ndim]


# struct format string for the 48-byte header:
#   <I  version (4B)
#   B   dtype (1B)
#   B   ndim (1B)
#   B   layout (1B)
#   B   _pad (1B)
#   8I  shape (32B)
#   I   payload_len (4B)
#   I   _reserved (4B)
_HEADER_STRUCT: Final[struct.Struct] = struct.Struct("<I4B8II I")
assert _HEADER_STRUCT.size == TENSOR_HEADER_SIZE, (
    f"header struct format must be {TENSOR_HEADER_SIZE} bytes, got {_HEADER_STRUCT.size}"
)


def decode_header(buf: bytes | memoryview) -> TensorFrameHeader:
    """Decode the leading 48 bytes of ``buf`` into a ``TensorFrameHeader``.

    Validates every discriminant and shape * dtype size invariant. Use
    :func:`decode_tensor` when you also want to materialise the payload
    into a ``numpy.ndarray``.

    Raises
    ------
    UnsupportedVersion
        Header version is not ``TENSOR_FRAME_VERSION``.
    UnknownDtype
        Dtype byte is not a known discriminant.
    UnknownLayout
        Layout byte is not a known discriminant.
    TooManyDims
        ``ndim > MAX_NDIM``.
    PayloadLenMismatch
        Declared ``payload_len`` does not match ``shape * dtype.size_bytes``.
    """
    if len(buf) < TENSOR_HEADER_SIZE:
        raise PayloadLenMismatch(
            f"buffer of {len(buf)} bytes is shorter than header ({TENSOR_HEADER_SIZE})"
        )
    fields = _HEADER_STRUCT.unpack_from(buf, 0)
    (
        version,
        dtype_byte,
        ndim,
        layout_byte,
        _pad,
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        payload_len,
        _reserved,
    ) = fields

    if version != TENSOR_FRAME_VERSION:
        raise UnsupportedVersion(
            f"unsupported tensor frame version {version} (expected {TENSOR_FRAME_VERSION})"
        )
    if ndim > MAX_NDIM:
        raise TooManyDims(f"ndim {ndim} exceeds MAX_NDIM {MAX_NDIM}")
    try:
        dtype = TensorDtype(dtype_byte)
    except ValueError as e:
        raise UnknownDtype(f"unknown dtype discriminant {dtype_byte}") from e
    try:
        layout = TensorLayout(layout_byte)
    except ValueError as e:
        raise UnknownLayout(f"unknown layout discriminant {layout_byte}") from e

    shape_full: tuple[int, ...] = (s0, s1, s2, s3, s4, s5, s6, s7)
    active_shape = shape_full[:ndim]
    n_elements = 1
    for d in active_shape:
        n_elements *= d
    expected = n_elements * dtype.size_bytes
    if payload_len != expected:
        raise PayloadLenMismatch(
            f"header payload_len {payload_len} != shape product {expected} "
            f"(shape={active_shape}, dtype={dtype.name})"
        )
    return TensorFrameHeader(
        version=version,
        dtype=dtype,
        ndim=ndim,
        layout=layout,
        shape=shape_full,
        payload_len=payload_len,
    )


def decode_tensor(buf: bytes | memoryview) -> tuple[TensorFrameHeader, NDArray]:
    """Decode a full ``(header, ndarray)`` pair from ``buf``.

    The returned ndarray is reshaped to ``header.active_shape`` and
    carries the dtype encoded in the header. ``layout=ROW_MAJOR`` is
    materialised as C-order; ``COL_MAJOR`` as Fortran-order.

    Raises
    ------
    PayloadLenMismatch
        ``buf`` does not contain enough trailing bytes after the header,
        or the declared ``payload_len`` disagrees with the buffer tail.
    """
    header = decode_header(buf)
    payload_start = TENSOR_HEADER_SIZE
    payload_end = payload_start + header.payload_len
    if payload_end > len(buf):
        raise PayloadLenMismatch(f"buffer truncated: need {payload_end} bytes, have {len(buf)}")
    payload = bytes(buf[payload_start:payload_end])
    flat = np.frombuffer(payload, dtype=header.dtype.numpy_dtype)
    order: Literal["C", "F"] = "C" if header.layout == TensorLayout.ROW_MAJOR else "F"
    array = np.reshape(flat, header.active_shape, order=order)
    return header, array


def encode_tensor(
    array: NDArray,
    *,
    dtype: TensorDtype | None = None,
    layout: TensorLayout = TensorLayout.ROW_MAJOR,
) -> bytes:
    """Encode a NumPy array into the wire format.

    Primarily for round-trip tests. Production wire traffic is produced
    by the Rust side (`encode_tensor` in ``tensor_frame.rs``); this
    helper exists so Python tests can exercise the format without
    spinning up the Rust process.
    """
    if dtype is None:
        dtype = _numpy_to_tensor_dtype(array.dtype)
    expected_np = dtype.numpy_dtype
    if array.dtype != expected_np:
        array = array.astype(expected_np)
    if layout == TensorLayout.ROW_MAJOR:
        contig = np.ascontiguousarray(array)
    else:
        contig = np.asfortranarray(array)
    payload = contig.tobytes(order="C" if layout == TensorLayout.ROW_MAJOR else "F")
    ndim = contig.ndim
    if ndim > MAX_NDIM:
        raise TooManyDims(f"ndim {ndim} exceeds MAX_NDIM {MAX_NDIM}")
    shape_arr = [0] * MAX_NDIM
    for i, d in enumerate(contig.shape):
        shape_arr[i] = int(d)
    payload_len = len(payload)
    header_bytes = _HEADER_STRUCT.pack(
        TENSOR_FRAME_VERSION,
        int(dtype),
        ndim,
        int(layout),
        0,  # _pad
        *shape_arr,
        payload_len,
        0,  # _reserved
    )
    return header_bytes + payload


def _numpy_to_tensor_dtype(np_dtype: np.dtype) -> TensorDtype:
    for td, exp in _DTYPE_NUMPY.items():
        if np_dtype == exp:
            return td
    # Accept the system-native f4/f8 etc. dtypes when they happen to be
    # the same byte size; on little-endian platforms these compare equal
    # to the explicit ``<f4`` etc. above, so this fallback only fires on
    # exotic platforms we don't currently target.
    if np_dtype.kind == "f" and np_dtype.itemsize == 4:
        return TensorDtype.F32
    if np_dtype.kind == "u" and np_dtype.itemsize == 1:
        return TensorDtype.U8
    if np_dtype.kind == "i" and np_dtype.itemsize == 4:
        return TensorDtype.I32
    if np_dtype.kind == "i" and np_dtype.itemsize == 8:
        return TensorDtype.I64
    if np_dtype.kind == "f" and np_dtype.itemsize == 8:
        return TensorDtype.F64
    raise UnknownDtype(f"no TensorDtype mapping for numpy dtype {np_dtype!r}")


__all__ = [
    "TENSOR_FRAME_VERSION",
    "TENSOR_HEADER_SIZE",
    "MAX_NDIM",
    "TensorDtype",
    "TensorLayout",
    "TensorFrameHeader",
    "TensorFrameError",
    "UnsupportedVersion",
    "UnknownDtype",
    "UnknownLayout",
    "PayloadLenMismatch",
    "TooManyDims",
    "decode_header",
    "decode_tensor",
    "encode_tensor",
]
