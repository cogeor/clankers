"""Tests for clanker_gym.client framing and encoding."""

import json
import struct

import pytest

from clanker_gym.client import GymClient, ProtocolError


class TestFraming:
    """Test the length-prefixed JSON framing encode/decode."""

    def test_encode_decode_roundtrip(self):
        """Verify that _send_raw encoding matches _recv_raw decoding."""
        import io
        import socket
        from unittest.mock import MagicMock

        msg = {"type": "ping", "timestamp": 12345}
        payload = json.dumps(msg).encode("utf-8")
        header = struct.pack("<I", len(payload))
        wire = header + payload

        client = GymClient.__new__(GymClient)
        client._sock = MagicMock(spec=socket.socket)

        # Test send encoding
        sent_data = bytearray()

        def capture_sendall(data):
            sent_data.extend(data)

        client._sock.sendall = capture_sendall
        client._send_raw(msg)
        assert bytes(sent_data) == wire

        # Test recv decoding
        buf = io.BytesIO(wire)

        def mock_recv(n):
            return buf.read(n)

        client._sock.recv = mock_recv
        result = client._recv_raw()
        assert result == msg

    def test_payload_too_large(self):
        import socket
        from unittest.mock import MagicMock

        client = GymClient.__new__(GymClient)
        client._sock = MagicMock(spec=socket.socket)

        # Create a message that would exceed MAX_MESSAGE_SIZE
        big_msg = {"data": "x" * (17 * 1024 * 1024)}
        with pytest.raises(ProtocolError, match="too large"):
            client._send_raw(big_msg)

    def test_recv_truncated_header(self):
        import socket
        from unittest.mock import MagicMock

        client = GymClient.__new__(GymClient)
        client._sock = MagicMock(spec=socket.socket)
        client._sock.recv.return_value = b""  # EOF

        with pytest.raises(ProtocolError, match="closed"):
            client._recv_raw()

    def test_little_endian_encoding(self):
        """Verify 4-byte LE u32 length prefix."""
        import socket
        from unittest.mock import MagicMock

        client = GymClient.__new__(GymClient)
        client._sock = MagicMock(spec=socket.socket)

        sent_data = bytearray()
        client._sock.sendall = lambda d: sent_data.extend(d)

        msg = {"type": "close"}
        client._send_raw(msg)

        # First 4 bytes should be little-endian length
        length = struct.unpack("<I", bytes(sent_data[:4]))[0]
        payload = sent_data[4:]
        assert length == len(payload)
        assert json.loads(payload.decode("utf-8")) == msg


class TestGymClientContext:
    def test_context_manager(self):
        client = GymClient.__new__(GymClient)
        client._sock = None
        with client:
            pass  # Should not raise
