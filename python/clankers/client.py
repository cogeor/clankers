"""Low-level TCP client with length-prefixed JSON framing.

Handles the wire protocol: 4-byte little-endian u32 length prefix
followed by a JSON payload. Provides ``send`` / ``recv`` for typed
request/response dictionaries.

Binary observation protocol
---------------------------
When ``binary_obs=True`` is requested (via capabilities), the server may
respond to step requests with an ``obs_encoding`` field in the JSON.  If
``obs_encoding["type"] == "RawU8"``, a second framed binary message
immediately follows the JSON frame.  The helper :meth:`recv_binary_frame`
reads that follow-up frame (4-byte LE u32 length + raw bytes).
"""

from __future__ import annotations

import contextlib
import json
import socket
import struct
from typing import Any

import numpy as np

PROTOCOL_VERSION = "1.0.0"
MAX_MESSAGE_SIZE = 16 * 1024 * 1024  # 16 MiB


class ProtocolError(Exception):
    """Raised when a protocol-level error occurs."""


class GymClient:
    """TCP client for the Clankers gym protocol.

    Manages a persistent TCP connection with length-prefixed JSON framing.
    Handles the Init handshake automatically on connect.

    Parameters
    ----------
    host : str
        Server hostname or IP.
    port : int
        Server port.
    client_name : str
        Identifier sent during handshake.
    capabilities : dict[str, bool] | None
        Requested capability flags.  Pass ``{"binary_obs": True}`` to
        enable the binary image observation path.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9876,
        client_name: str = "clankers_py",
        capabilities: dict[str, bool] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.client_name = client_name
        self.capabilities = capabilities or {}
        self._sock: socket.socket | None = None
        self._negotiated_capabilities: dict[str, bool] = {}
        self._env_info: dict[str, Any] = {}
        # Set when binary_obs is negotiated and the server sends RawU8 frames.
        self._binary_obs_active: bool = False

    @property
    def negotiated_capabilities(self) -> dict[str, bool]:
        return self._negotiated_capabilities

    @property
    def env_info(self) -> dict[str, Any]:
        return self._env_info

    def connect(self, seed: int | None = None) -> dict[str, Any]:
        """Connect to the server and perform the Init handshake.

        Always requests ``binary_obs: true`` in capabilities so the server
        can send image observations as raw binary frames instead of base64
        JSON.  If the server does not support the capability, the negotiated
        flag will be ``False`` and the client falls back to JSON observations.

        Returns the InitResponse dict.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))

        # Merge caller-supplied capabilities with binary_obs request.
        caps = dict(self.capabilities)
        caps.setdefault("binary_obs", True)

        init_msg: dict[str, Any] = {
            "type": "init",
            "protocol_version": PROTOCOL_VERSION,
            "client_name": self.client_name,
            "client_version": "0.1.0",
            "capabilities": caps,
        }
        if seed is not None:
            init_msg["seed"] = seed

        self._send_raw(init_msg)
        resp = self._recv_raw()

        if resp.get("type") == "error":
            msg = resp.get("message", "unknown error")
            raise ProtocolError(f"Handshake failed: {msg}")

        self._negotiated_capabilities = resp.get("capabilities", {})
        self._env_info = resp.get("env_info", {})
        # Track whether the server agreed to binary obs so step() knows to
        # read an extra binary frame when obs_encoding is present.
        self._binary_obs_active = bool(self._negotiated_capabilities.get("binary_obs", False))
        return resp

    def send(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a request and return the response.

        For step requests, if the server negotiated ``binary_obs`` and responds
        with ``obs_encoding.type == "RawU8"``, the observation is decoded from
        a follow-up binary frame and stored under ``"_image_obs"`` in the
        returned dict as a ``np.ndarray`` of shape ``(H, W, C)`` and
        dtype ``uint8``.  The original ``observation`` sentinel field is still
        present for backward compatibility but its data should be ignored.
        """
        if self._sock is None:
            msg = "Not connected. Call connect() first."
            raise ProtocolError(msg)
        self._send_raw(request)
        resp = self._recv_raw()

        # Binary obs path: read extra binary frame if server sends RawU8.
        obs_encoding = resp.get("obs_encoding")
        if obs_encoding is not None and obs_encoding.get("type") == "RawU8":
            width = int(obs_encoding["width"])
            height = int(obs_encoding["height"])
            channels = int(obs_encoding["channels"])
            raw = self.recv_binary_frame()
            image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, channels))
            resp["_image_obs"] = image

        return resp

    def recv_binary_frame(self) -> bytes:
        """Read a length-prefixed binary frame from the socket.

        Wire format: 4-byte little-endian u32 length prefix followed by
        exactly that many raw bytes.  Matches the Rust ``write_binary_frame``
        helper in ``clankers-gym/src/framing.rs``.
        """
        header = self._recv_exact(4)
        if len(header) < 4:
            raise ProtocolError("Connection closed while reading binary frame length")
        (length,) = struct.unpack("<I", header)
        if length > MAX_MESSAGE_SIZE:
            raise ProtocolError(f"Binary frame too large: {length} bytes")
        data = self._recv_exact(length)
        if len(data) < length:
            raise ProtocolError("Connection closed while reading binary frame payload")
        return data

    def close(self) -> None:
        """Send Close request and disconnect."""
        if self._sock is not None:
            try:
                self._send_raw({"type": "close"})
                self._recv_raw()
            except (OSError, ProtocolError):
                pass
            finally:
                self._sock.close()
                self._sock = None

    def _send_raw(self, msg: dict[str, Any]) -> None:
        """Encode and send a length-prefixed JSON message."""
        assert self._sock is not None
        payload = json.dumps(msg).encode("utf-8")
        if len(payload) > MAX_MESSAGE_SIZE:
            msg_str = f"Payload too large: {len(payload)} bytes"
            raise ProtocolError(msg_str)
        header = struct.pack("<I", len(payload))
        self._sock.sendall(header + payload)

    def _recv_raw(self) -> dict[str, Any]:
        """Read a length-prefixed JSON message."""
        assert self._sock is not None
        header = self._recv_exact(4)
        if len(header) < 4:
            raise ProtocolError("Connection closed during read")
        (length,) = struct.unpack("<I", header)
        if length > MAX_MESSAGE_SIZE:
            raise ProtocolError(f"Message too large: {length} bytes")
        payload = self._recv_exact(length)
        if len(payload) < length:
            raise ProtocolError("Connection closed during payload read")
        return json.loads(payload.decode("utf-8"))  # type: ignore[no-any-return]

    def _recv_exact(self, n: int) -> bytes:
        """Read exactly n bytes from the socket."""
        assert self._sock is not None
        data = bytearray()
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    def __enter__(self) -> GymClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
