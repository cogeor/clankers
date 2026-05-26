//! Workspace developer tooling library.
//!
//! Exposes the subcommand implementations consumed by `xtask/src/main.rs`
//! so integration tests under `xtask/tests/` can call into the same
//! logic without re-spawning the binary.

pub mod line_count;
