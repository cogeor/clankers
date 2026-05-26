"""Protocol-level error types for the clankers Python client."""

from __future__ import annotations


class ProtocolError(Exception):
    """Raised when the gym wire protocol contract is violated.

    Examples
    --------
    - The TCP socket is closed or never opened.
    - A response's observation shape or dtype does not match the
      negotiated ObservationSpace.
    - A length-prefixed frame is truncated or oversize.
    """
