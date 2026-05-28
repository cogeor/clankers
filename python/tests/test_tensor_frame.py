"""Round-trip tests for the Python tensor-frame decoder (P4.3).

These exercise the encode/decode path in isolation. The wire format is
pinned byte-for-byte by the Rust ``tensor_frame_header_size_is_48_bytes``
test in ``crates/clankers-gym/src/tensor_frame.rs``; here we verify the
Python decoder agrees on layout, dtype handling, and the error paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from clankers.tensor_frame import (
    MAX_NDIM,
    TENSOR_FRAME_VERSION,
    TENSOR_HEADER_SIZE,
    PayloadLenMismatch,
    TensorDtype,
    TensorFrameError,
    TensorLayout,
    TooManyDims,
    UnknownDtype,
    UnsupportedVersion,
    decode_header,
    decode_tensor,
    encode_tensor,
)


def test_header_size_constant_is_48():
    assert TENSOR_HEADER_SIZE == 48


def test_dtype_size_bytes_table():
    assert TensorDtype.F32.size_bytes == 4
    assert TensorDtype.U8.size_bytes == 1
    assert TensorDtype.I32.size_bytes == 4
    assert TensorDtype.I64.size_bytes == 8
    assert TensorDtype.F64.size_bytes == 8


def test_roundtrip_f32_matrix():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    buf = encode_tensor(arr)
    assert len(buf) == TENSOR_HEADER_SIZE + 24

    header, decoded = decode_tensor(buf)
    assert header.version == TENSOR_FRAME_VERSION
    assert header.dtype == TensorDtype.F32
    assert header.ndim == 2
    assert header.active_shape == (2, 3)
    assert header.layout == TensorLayout.ROW_MAJOR
    np.testing.assert_array_equal(decoded, arr)
    assert decoded.dtype == np.float32


def test_roundtrip_u8_image():
    arr = (np.arange(32 * 32 * 3, dtype=np.int64) % 251).astype(np.uint8).reshape(32, 32, 3)
    buf = encode_tensor(arr)
    header, decoded = decode_tensor(buf)
    assert header.active_shape == (32, 32, 3)
    assert header.dtype == TensorDtype.U8
    np.testing.assert_array_equal(decoded, arr)


def test_roundtrip_i64_vector():
    arr = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int64)
    buf = encode_tensor(arr)
    header, decoded = decode_tensor(buf)
    assert header.dtype == TensorDtype.I64
    assert header.active_shape == (7,)
    np.testing.assert_array_equal(decoded, arr)


def test_decode_header_rejects_short_buffer():
    with pytest.raises(PayloadLenMismatch):
        decode_header(b"\x00" * (TENSOR_HEADER_SIZE - 1))


def test_decode_rejects_truncated_payload():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    buf = encode_tensor(arr)
    # Drop the last byte so payload_end > len(buf).
    truncated = buf[:-1]
    with pytest.raises(PayloadLenMismatch):
        decode_tensor(truncated)


def test_decode_rejects_unknown_version():
    arr = np.zeros(4, dtype=np.float32)
    buf = bytearray(encode_tensor(arr))
    # Overwrite version field (offset 0..4, LE u32).
    buf[0:4] = (99).to_bytes(4, "little")
    with pytest.raises(UnsupportedVersion):
        decode_header(bytes(buf))


def test_decode_rejects_unknown_dtype():
    arr = np.zeros(4, dtype=np.float32)
    buf = bytearray(encode_tensor(arr))
    # Dtype byte lives at offset 4.
    buf[4] = 250
    with pytest.raises(UnknownDtype):
        decode_header(bytes(buf))


def test_decode_rejects_too_many_dims():
    arr = np.zeros(4, dtype=np.float32)
    buf = bytearray(encode_tensor(arr))
    # ndim byte lives at offset 5.
    buf[5] = MAX_NDIM + 1
    with pytest.raises(TooManyDims):
        decode_header(bytes(buf))


def test_decode_detects_shape_payload_mismatch():
    # Build a buffer by hand: header claims shape (2, 3) f32 -> 24B
    # payload, but we tamper the shape to (2, 4) -> 32B which disagrees
    # with the declared payload_len 24.
    arr = np.zeros((2, 3), dtype=np.float32)
    buf = bytearray(encode_tensor(arr))
    # shape[1] is at offset 8 + 1*4 = 12.
    buf[12:16] = (4).to_bytes(4, "little")
    with pytest.raises(PayloadLenMismatch):
        decode_header(bytes(buf))


def test_col_major_roundtrip_preserves_values():
    arr = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    buf = encode_tensor(arr, layout=TensorLayout.COL_MAJOR)
    header, decoded = decode_tensor(buf)
    assert header.layout == TensorLayout.COL_MAJOR
    np.testing.assert_array_equal(decoded, arr)


def test_encode_promotes_non_native_dtype():
    arr = np.array([1, 2, 3], dtype=np.float64)
    buf = encode_tensor(arr, dtype=TensorDtype.F32)
    header, decoded = decode_tensor(buf)
    assert header.dtype == TensorDtype.F32
    np.testing.assert_array_equal(decoded, arr.astype(np.float32))


def test_error_subclasses_are_tensor_frame_errors():
    assert issubclass(UnsupportedVersion, TensorFrameError)
    assert issubclass(UnknownDtype, TensorFrameError)
    assert issubclass(PayloadLenMismatch, TensorFrameError)
    assert issubclass(TooManyDims, TensorFrameError)
