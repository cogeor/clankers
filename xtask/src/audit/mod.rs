//! Workspace-meta audit catalogues.
//!
//! Each submodule pins one slice of the
//! `CODE_QUALITY_REVIEW.md` audit as typed Rust data with unit-test
//! invariants. Modules live in `xtask` (and only in `xtask`) so the
//! workspace's spine crate (`clankers-core`) stays free of meta
//! concerns it would otherwise force on every downstream consumer.
//!
//! Submodules:
//!
//! - [`code_quality`] — outcome per numbered review item.
//! - [`panic_audit`] — every audited panic site + category.
//! - [`stability`] — declared stability tier per public module.
//! - [`test_layers`] — non-unit tests inventoried by layer.
//! - [`release`] — published artifacts + release checklist.

pub mod code_quality;
pub mod panic_audit;
pub mod release;
pub mod stability;
pub mod test_layers;
