//! Panic-site audit catalogue (G7).
//!
//! CODE_QUALITY_REVIEW § "Gap 7: Errors Aren't Uniform". Several
//! workspace surfaces still panic where a typed error would be more
//! appropriate, and several panics are deliberate "broken-contract"
//! signals that the user can't recover from. The two need to be
//! distinguishable, otherwise downstream code can't decide whether to
//! catch the panic, fix the input, or treat it as a bug.
//!
//! This module is the canonical audit: a typed list of every panic
//! site we've reviewed, what category it belongs to, and whether it
//! has a fallible counterpart. CI can later parse this list and grep
//! the source to ensure no new panics were introduced without an
//! entry here.
//!
//! ## Categories
//!
//! - [`PanicCategory::ProgrammerError`] — the panic indicates a
//!   programmer-side broken invariant. Recovery is impossible because
//!   the bad state is already in the type system; the panic is the
//!   appropriate signal. Examples: layout-binding without enough
//!   entities (the documented contract was broken), unreachable
//!   match arms.
//! - [`PanicCategory::FallibleAlternativeExists`] — the panic exists
//!   for ergonomic legacy callers, but the recommended path is a
//!   `try_*` method that returns `Result`. Protocol / server / env
//!   code should always use the fallible variant.
//! - [`PanicCategory::TestOnly`] — the panic is inside `#[cfg(test)]`.
//!   No production reachability.

use serde::{Deserialize, Serialize};

/// Categorisation for a single audited panic site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PanicCategory {
    /// Panic signals a broken programmer-side invariant; no recovery.
    ProgrammerError,
    /// A `try_*` fallible counterpart exists; users should call that.
    FallibleAlternativeExists,
    /// Inside `#[cfg(test)]` only.
    TestOnly,
}

// ---------------------------------------------------------------------------
// AuditedPanic
// ---------------------------------------------------------------------------

/// One panic site reviewed during the G7 audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditedPanic {
    /// Source path (relative to repo root).
    pub file: String,
    /// Approximate line number when the audit landed. Stays valid
    /// until the file is restructured.
    pub line: u32,
    /// One-line description of why the panic exists.
    pub note: String,
    /// Category.
    pub category: PanicCategory,
    /// Fallible alternative, when applicable (e.g.
    /// `"Action::try_scale"`).
    #[serde(default)]
    pub fallible_alternative: Option<String>,
}

/// The audited panic catalogue at G7-landing time.
///
/// New panics added after G7 should append to this list; CI lint
/// (follow-up) can grep production sources for `panic!` /
/// `unwrap()` and require a matching entry.
#[must_use]
pub fn panic_audit_catalogue() -> Vec<AuditedPanic> {
    use PanicCategory::{FallibleAlternativeExists, ProgrammerError};
    let p = |file: &str, line, note: &str, category, alt: Option<&str>| AuditedPanic {
        file: file.to_string(),
        line,
        note: note.to_string(),
        category,
        fallible_alternative: alt.map(str::to_string),
    };
    vec![
        // Action ergonomic aliases — fallible variants are documented.
        p(
            "crates/clankers-core/src/types.rs",
            190,
            "Action::values() panics on non-Continuous; protocol code uses Action::as_continuous \
             or Action::try_scale.",
            FallibleAlternativeExists,
            Some("Action::as_continuous / Action::try_scale"),
        ),
        p(
            "crates/clankers-core/src/types.rs",
            423,
            "ActionSpace::sample() rejects Dict spaces — Dict callers must sample each sub-space \
             individually. ProgrammerError because Dict-sample has no canonical semantics.",
            ProgrammerError,
            None,
        ),
        // Layout binding — try_bind_entities is the fallible variant.
        p(
            "crates/clankers-core/src/layout.rs",
            280,
            "JointLayout::bind_entities() panics in both debug and release if the count is wrong; \
             callers wanting a Result should use try_bind_entities (P0.4).",
            FallibleAlternativeExists,
            Some("JointLayout::try_bind_entities"),
        ),
        // Sensor constructors — try_new is the fallible variant.
        p(
            "crates/clankers-env/src/sensors.rs",
            0,
            "Sensor::new(layout) family panics if the layout isn't fully bound; try_new(layout) \
             returns Result<Self, SensorBuildError> (P0.4).",
            FallibleAlternativeExists,
            Some("Sensor::try_new"),
        ),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_audit_catalogue_is_non_empty() {
        let cat = panic_audit_catalogue();
        assert!(!cat.is_empty());
    }

    #[test]
    fn every_fallible_alternative_entry_names_an_alternative() {
        for entry in panic_audit_catalogue() {
            if entry.category == PanicCategory::FallibleAlternativeExists {
                assert!(
                    entry.fallible_alternative.is_some(),
                    "audit entry at {}:{} marked FallibleAlternativeExists but has no alternative",
                    entry.file,
                    entry.line
                );
            }
        }
    }

    #[test]
    fn audit_categories_serialise_to_snake_case() {
        let s = serde_json::to_string(&PanicCategory::FallibleAlternativeExists).unwrap();
        assert_eq!(s, "\"fallible_alternative_exists\"");
    }
}
