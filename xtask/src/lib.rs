//! Workspace developer tooling library.
//!
//! Exposes the subcommand implementations consumed by `xtask/src/main.rs`
//! so integration tests under `xtask/tests/` can call into the same
//! logic without re-spawning the binary, plus workspace-meta audit
//! catalogues (see [`audit`]) that pin invariants about the workspace's
//! own structure (stability tiers, panic sites, release checklist).

pub mod audit;
pub mod line_count;
