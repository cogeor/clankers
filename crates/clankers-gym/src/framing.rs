//! Length-prefixed JSON framing for the wire protocol.
//!
//! Every message on the wire is a 4-byte **little-endian** `u32` length prefix
//! followed by that many bytes of UTF-8 JSON payload. See `PROTOCOL_SPEC.md`
//! Section 1 for the full specification.
//!
//! # Wire format
//!
//! ```text
//! +----------------+------------------+
//! | Length (4B LE) | JSON Payload     |
//! +----------------+------------------+
//! ```

use std::io::{Read, Write};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::protocol::{MAX_MESSAGE_SIZE, ProtocolError};

/// Read a length-prefixed JSON message from a stream.
///
/// Returns `Ok(None)` if the stream reaches EOF before any bytes are read
/// (clean disconnect). Returns an error if the length prefix or payload
/// cannot be read, the payload exceeds [`MAX_MESSAGE_SIZE`], or the JSON
/// is invalid.
pub fn read_message<T: DeserializeOwned>(
    reader: &mut impl Read,
) -> Result<Option<T>, ProtocolError> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
        Err(e) => return Err(ProtocolError::Io(e)),
    }

    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_MESSAGE_SIZE {
        return Err(ProtocolError::PayloadTooLarge {
            size: len,
            max: MAX_MESSAGE_SIZE,
        });
    }

    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload)?;

    let msg: T = serde_json::from_slice(&payload)?;
    Ok(Some(msg))
}

/// Write a length-prefixed JSON message to a stream.
///
/// Serialises `msg` to JSON, writes the 4-byte little-endian length prefix,
/// then writes the JSON payload. The stream is flushed after writing.
pub fn write_message<T: Serialize>(writer: &mut impl Write, msg: &T) -> Result<(), ProtocolError> {
    let payload = serde_json::to_vec(msg)?;

    if payload.len() > MAX_MESSAGE_SIZE {
        return Err(ProtocolError::PayloadTooLarge {
            size: payload.len(),
            max: MAX_MESSAGE_SIZE,
        });
    }

    // Safe: MAX_MESSAGE_SIZE (16 MiB) fits in u32.
    let len = u32::try_from(payload.len())
        .map_err(|_| ProtocolError::PayloadTooLarge {
            size: payload.len(),
            max: MAX_MESSAGE_SIZE,
        })?
        .to_le_bytes();
    writer.write_all(&len)?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
}

/// Write a length-prefixed binary frame to a stream.
///
/// Writes a 4-byte little-endian `u32` length prefix followed by the raw
/// bytes. Flushes the writer after writing.
///
/// # Errors
///
/// Returns an `io::Error` if writing or flushing fails.
pub fn write_binary_frame<W: Write>(writer: &mut W, data: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(data.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "data too large for u32 length prefix"))?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(data)?;
    writer.flush()
}

/// Read a length-prefixed binary frame from a stream.
///
/// Reads the 4-byte little-endian `u32` length prefix, then reads exactly
/// that many bytes of payload.
///
/// # Errors
///
/// Returns an `io::Error` if reading fails or if the stream ends prematurely.
pub fn read_binary_frame<R: Read>(reader: &mut R) -> std::io::Result<Vec<u8>> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    let mut data = vec![0u8; len];
    reader.read_exact(&mut data)?;
    Ok(data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Request, Response};
    use clankers_core::types::Action;
    use std::io::Cursor;

    #[test]
    fn roundtrip_request() {
        let req = Request::Reset { seed: Some(42) };
        let mut buf = Vec::new();
        write_message(&mut buf, &req).unwrap();

        let mut cursor = Cursor::new(&buf);
        let req2: Request = read_message(&mut cursor).unwrap().unwrap();
        if let Request::Reset { seed } = req2 {
            assert_eq!(seed, Some(42));
        } else {
            panic!("expected Reset");
        }
    }

    #[test]
    fn roundtrip_response() {
        let resp = Response::Close;
        let mut buf = Vec::new();
        write_message(&mut buf, &resp).unwrap();

        let mut cursor = Cursor::new(&buf);
        let resp2: Response = read_message(&mut cursor).unwrap().unwrap();
        assert!(matches!(resp2, Response::Close));
    }

    #[test]
    fn length_prefix_is_little_endian() {
        let req = Request::Spaces;
        let mut buf = Vec::new();
        write_message(&mut buf, &req).unwrap();

        // First 4 bytes are the length prefix (little-endian u32)
        let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        assert_eq!(len, buf.len() - 4);
    }

    #[test]
    fn eof_returns_none() {
        let buf: Vec<u8> = Vec::new();
        let mut cursor = Cursor::new(&buf);
        let result: Result<Option<Request>, _> = read_message(&mut cursor);
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn payload_too_large_is_rejected() {
        // Craft a length prefix claiming a huge payload
        let fake_len = (u32::try_from(MAX_MESSAGE_SIZE).unwrap() + 1).to_le_bytes();
        let mut cursor = Cursor::new(fake_len.to_vec());
        let result: Result<Option<Request>, _> = read_message(&mut cursor);
        let err = result.unwrap_err();
        assert!(matches!(err, ProtocolError::PayloadTooLarge { .. }));
    }

    #[test]
    fn multiple_messages_in_sequence() {
        let mut buf = Vec::new();
        let req1 = Request::Spaces;
        let req2 = Request::Reset { seed: None };
        let req3 = Request::Step {
            action: Action::Discrete(1),
        };
        write_message(&mut buf, &req1).unwrap();
        write_message(&mut buf, &req2).unwrap();
        write_message(&mut buf, &req3).unwrap();

        let mut cursor = Cursor::new(&buf);
        let r1: Request = read_message(&mut cursor).unwrap().unwrap();
        let r2: Request = read_message(&mut cursor).unwrap().unwrap();
        let r3: Request = read_message(&mut cursor).unwrap().unwrap();
        assert!(matches!(r1, Request::Spaces));
        assert!(matches!(r2, Request::Reset { .. }));
        assert!(matches!(r3, Request::Step { .. }));

        // No more messages
        let r4: Result<Option<Request>, _> = read_message(&mut cursor);
        assert!(r4.unwrap().is_none());
    }

    #[test]
    fn invalid_json_returns_error() {
        let garbage = b"not json at all";
        let len = u32::try_from(garbage.len()).unwrap().to_le_bytes();
        let mut data = len.to_vec();
        data.extend_from_slice(garbage);

        let mut cursor = Cursor::new(&data);
        let result: Result<Option<Request>, _> = read_message(&mut cursor);
        assert!(matches!(result, Err(ProtocolError::Json(_))));
    }

    #[test]
    fn pong_roundtrip_via_framing() {
        let resp = Response::Pong {
            timestamp: 12345,
            server_time: 12346,
        };
        let mut buf = Vec::new();
        write_message(&mut buf, &resp).unwrap();

        let mut cursor = Cursor::new(&buf);
        let resp2: Response = read_message(&mut cursor).unwrap().unwrap();
        if let Response::Pong {
            timestamp,
            server_time,
        } = resp2
        {
            assert_eq!(timestamp, 12345);
            assert_eq!(server_time, 12346);
        } else {
            panic!("expected Pong");
        }
    }

    #[test]
    fn binary_frame_roundtrip_empty() {
        let data: &[u8] = &[];
        let mut buf = Vec::new();
        write_binary_frame(&mut buf, data).unwrap();

        let mut cursor = Cursor::new(&buf);
        let result = read_binary_frame(&mut cursor).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn binary_frame_roundtrip_small() {
        let data: &[u8] = &[0u8, 1, 2, 3, 255, 128, 64];
        let mut buf = Vec::new();
        write_binary_frame(&mut buf, data).unwrap();

        let mut cursor = Cursor::new(&buf);
        let result = read_binary_frame(&mut cursor).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn binary_frame_roundtrip_image_like() {
        // Simulate a 4x4 RGB image (48 bytes)
        let data: Vec<u8> = (0u8..48).collect();
        let mut buf = Vec::new();
        write_binary_frame(&mut buf, &data).unwrap();

        let mut cursor = Cursor::new(&buf);
        let result = read_binary_frame(&mut cursor).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn binary_frame_length_prefix_is_little_endian() {
        let data = b"hello";
        let mut buf = Vec::new();
        write_binary_frame(&mut buf, data).unwrap();

        // First 4 bytes are LE u32 length
        let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        assert_eq!(len, data.len());
        assert_eq!(&buf[4..], data);
    }

    #[test]
    fn binary_frame_multiple_sequential() {
        let frame1 = vec![1u8, 2, 3];
        let frame2 = vec![10u8, 20, 30, 40];
        let mut buf = Vec::new();
        write_binary_frame(&mut buf, &frame1).unwrap();
        write_binary_frame(&mut buf, &frame2).unwrap();

        let mut cursor = Cursor::new(&buf);
        let r1 = read_binary_frame(&mut cursor).unwrap();
        let r2 = read_binary_frame(&mut cursor).unwrap();
        assert_eq!(r1, frame1);
        assert_eq!(r2, frame2);
    }
}
