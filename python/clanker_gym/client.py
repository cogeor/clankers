"""Low-level TCP client with length-prefixed JSON framing.

Handles the wire protocol: 4-byte little-endian u32 length prefix
followed by a JSON payload. Provides ``send`` / ``recv`` for typed
request/response dictionaries.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

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
        Requested capability flags.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9876,
        client_name: str = "clanker_gym_py",
        capabilities: dict[str, bool] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.client_name = client_name
        self.capabilities = capabilities or {}
        self._sock: socket.socket | None = None
        self._negotiated_capabilities: dict[str, bool] = {}
        self._env_info: dict[str, Any] = {}

    @property
    def negotiated_capabilities(self) -> dict[str, bool]:
        return self._negotiated_capabilities

    @property
    def env_info(self) -> dict[str, Any]:
        return self._env_info

    def connect(self, seed: int | None = None) -> dict[str, Any]:
        """Connect to the server and perform the Init handshake.

        Returns the InitResponse dict.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))

        init_msg: dict[str, Any] = {
            "type": "init",
            "protocol_version": PROTOCOL_VERSION,
            "client_name": self.client_name,
            "client_version": "0.1.0",
            "capabilities": self.capabilities,
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
        return resp

    def send(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a request and return the response."""
        if self._sock is None:
            msg = "Not connected. Call connect() first."
            raise ProtocolError(msg)
        self._send_raw(request)
        return self._recv_raw()

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
            try:
                self._sock.close()
            except OSError:
                pass
