//! Code-quality review audit bookkeeping.
//!
//! Records the workspace's pass over `CODE_QUALITY_REVIEW.md`. The
//! audit catalogue is closed: every item the review called out has
//! either landed (PASS) or has a foundation commit + recorded
//! follow-up. Re-running the review is the way to start a fresh
//! generation; bump [`AUDIT_GENERATION`] and re-list the
//! [`AuditItem`]s when that happens.
//!
//! This module pairs with the typed inventories at
//! [`crate::stability`], [`crate::panic_audit`], [`crate::test_layers`],
//! [`crate::release`], [`crate::user_journeys`] — each of those
//! captures one slice of the review; this module captures the
//! pass-or-fail outcome per numbered item.

use serde::{Deserialize, Serialize};

/// Bumped each time the review is re-run end-to-end. Stays at `1` for
/// the spring-2026 audit; future re-runs increment.
pub const AUDIT_GENERATION: u32 = 1;

// ---------------------------------------------------------------------------
// AuditOutcome
// ---------------------------------------------------------------------------

/// Per-item outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditOutcome {
    /// Item is fully implemented and tested at landing time.
    Pass,
    /// Foundation only: data shapes / typed surfaces defined + tested,
    /// consumer wiring is a follow-up loop. Tracked as PASS for the
    /// review's "what's the state of the contract" question, separately
    /// for the "how widely is it used" question.
    Foundation,
}

// ---------------------------------------------------------------------------
// AuditItem
// ---------------------------------------------------------------------------

/// One review item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditItem {
    /// Item id (e.g. `"P0.1"`, `"G3"`).
    pub id: String,
    /// Outcome.
    pub outcome: AuditOutcome,
    /// Landing commit short hash.
    pub commit: String,
}

/// The canonical audit catalogue at AUDIT_GENERATION = 1.
#[must_use]
pub fn audit_catalogue() -> Vec<AuditItem> {
    use AuditOutcome::{Foundation, Pass};
    let item = |id: &str, outcome, commit: &str| AuditItem {
        id: id.to_string(),
        outcome,
        commit: commit.to_string(),
    };
    vec![
        item("P0.1", Pass, "e5f2a0b"),
        item("P0.2", Pass, "7c5dfd3"),
        item("P0.3", Pass, "9e85fd3"),
        item("P0.4", Pass, "d22d230"),
        item("P1.1", Pass, "a500ebe"),
        item("P1.2", Pass, "eeb9c5b"),
        item("P1.3", Pass, "4daf61a"),
        item("P1.4", Pass, "521c484"),
        item("P1.5", Pass, "202e75a"),
        item("P1.6", Pass, "10b2777"),
        item("P1.7", Pass, "e7417c3"),
        item("P1.8", Pass, "5efdef0"),
        item("P1.9", Pass, "67115d8"),
        item("P1.10", Pass, "fbd1cf7"),
        item("P1.11", Pass, "bde8187"),
        item("P1.12", Pass, "7ff2389"),
        item("P1.13", Pass, "55deebb"),
        item("P1.14", Pass, "d92d5bb"),
        item("P2.1", Pass, "9bd87bd"),
        item("P2.2", Pass, "d5aa51b"),
        item("P2.3", Pass, "faeb7f3"),
        item("P2.4", Pass, "8bd04d4"),
        item("P3.1", Pass, "7364430"),
        item("P3.2", Foundation, "f5b8e6c"),
        item("P3.3", Foundation, "ad11736"),
        item("P3.4", Foundation, "ad11736"),
        item("P4.1", Pass, "b3435d3"),
        item("P4.2", Pass, "a36b1b2"),
        item("P4.3", Pass, "d890e8c"),
        item("P4.4", Pass, "931765f"),
        item("G1", Foundation, "c3adc1a"),
        item("G2", Pass, "77ecf45"),
        item("G3", Foundation, "6df7f51"),
        item("G4", Pass, "0b98efc"),
        item("G5", Pass, "fd75b5d"),
        item("G6", Pass, "914d02d"),
        item("G7", Pass, "248ca48"),
        item("G8", Pass, "30dd871"),
        item("G9", Foundation, "8c0a81e"),
        item("G10", Pass, "2b041e4"),
        item("G11", Foundation, "827cb0a"),
        item("G12", Foundation, "dfbebb2"),
        item("G13", Pass, "4a5eb61"),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn audit_catalogue_has_no_duplicate_ids() {
        let cat = audit_catalogue();
        let ids: HashSet<&str> = cat.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids.len(), cat.len());
    }

    #[test]
    fn audit_catalogue_covers_every_numbered_review_item() {
        // Phase 0 (P0.1–P0.4)  – 4 items
        // Phase 1 (P1.1–P1.14) – 14 items
        // Phase 2 (P2.1–P2.4)  – 4 items
        // Phase 3 (P3.1–P3.4)  – 4 items
        // Phase 4 (P4.1–P4.4)  – 4 items
        // Gap analysis (G1–G13) – 13 items
        // total: 43 numbered items.
        let cat = audit_catalogue();
        assert_eq!(cat.len(), 43);
    }

    #[test]
    fn audit_outcomes_serialise_snake_case() {
        let s = serde_json::to_string(&AuditOutcome::Foundation).unwrap();
        assert_eq!(s, "\"foundation\"");
    }
}
