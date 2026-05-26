//! Per-subcommand implementations for the `clankers-app` binary.
//!
//! Each `pub mod` here exposes a `pub fn execute(...) -> std::process::ExitCode`
//! that `main.rs` calls from a `match` arm on the parsed [`crate::Commands`]
//! enum. The split keeps `main.rs` short (≤120 lines per the W5 PR1 gate)
//! and lets each subcommand grow independently.
//!
//! In W5 PR1, only `info`, `validate`, and `inspect` ship real
//! implementations. The remaining modules (`run`, `serve`, `record`,
//! `replay`, `bench`, `viz`) ship stubs or legacy shims and gain their
//! intended bodies in subsequent W5 PRs.

pub mod bench;
pub mod info;
pub mod inspect;
pub mod record;
pub mod replay;
pub mod run;
pub mod serve;
pub mod validate;
pub mod viz;
