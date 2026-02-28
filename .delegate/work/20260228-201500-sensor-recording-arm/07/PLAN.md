# Loop 07: Binary image TCP protocol

## Goal
Extend gym TCP protocol with binary observation channel for efficient image transfer.

## Changes
- `crates/clankers-gym/src/protocol.rs`: Add binary_obs capability, ObsEncoding enum
- `crates/clankers-gym/src/framing.rs`: Add write_binary_frame/read_binary_frame
- `crates/clankers-gym/src/server.rs`: After JSON response, send binary image if negotiated
- Unit tests for framing roundtrip
