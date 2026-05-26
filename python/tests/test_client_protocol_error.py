"""Pins the contract that the disconnected client raises ProtocolError.

Replaces the legacy ``assert self._sock is not None`` behaviour in
``_send_raw`` / ``_recv_raw`` / ``_recv_exact`` and harmonises the
existing ``send()`` error message with the three new sites (all four
raise ``ProtocolError("not connected: call connect() first")``).
"""

from __future__ import annotations

import pytest

from clankers import ProtocolError
from clankers.client import GymClient


def test_client_raises_protocol_error_when_not_connected() -> None:
    """``send()`` on a disconnected client raises ProtocolError."""
    client = GymClient.__new__(GymClient)
    client._sock = None
    with pytest.raises(ProtocolError, match="not connected"):
        client.send({"type": "ping", "timestamp": 0})


def test_send_raw_raises_protocol_error_when_not_connected() -> None:
    """``_send_raw`` raises ProtocolError instead of AssertionError."""
    client = GymClient.__new__(GymClient)
    client._sock = None
    with pytest.raises(ProtocolError, match="not connected"):
        client._send_raw({"type": "ping"})


def test_recv_raw_raises_protocol_error_when_not_connected() -> None:
    """``_recv_raw`` raises ProtocolError instead of AssertionError."""
    client = GymClient.__new__(GymClient)
    client._sock = None
    with pytest.raises(ProtocolError, match="not connected"):
        client._recv_raw()


def test_recv_exact_raises_protocol_error_when_not_connected() -> None:
    """``_recv_exact`` raises ProtocolError instead of AssertionError."""
    client = GymClient.__new__(GymClient)
    client._sock = None
    with pytest.raises(ProtocolError, match="not connected"):
        client._recv_exact(4)


def test_protocol_error_is_reexported_from_package_root() -> None:
    """``ProtocolError`` is importable from ``clankers`` and ``clankers.client``."""
    from clankers import ProtocolError as PkgProtocolError
    from clankers._errors import ProtocolError as ErrModProtocolError
    from clankers.client import ProtocolError as ClientProtocolError

    assert PkgProtocolError is ErrModProtocolError
    assert ClientProtocolError is ErrModProtocolError
