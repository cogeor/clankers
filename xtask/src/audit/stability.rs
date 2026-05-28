//! Declared API stability tiers.
//!
//! The workspace ships `pub` everywhere with no signal to downstream
//! about which surfaces are committed-to vs. internal helpers. This
//! module pins the canonical tiers.
//!
//! ## Tiers
//!
//! - [`StabilityTier::Stable`] — committed for the 1.x line. Breaking
//!   changes require a major bump + migration notes.
//! - [`StabilityTier::Experimental`] — public surface but reserves the
//!   right to break inside a major. Always flagged in the module-
//!   level rustdoc.
//! - [`StabilityTier::Internal`] — exists only for testing or
//!   cross-crate plumbing. Downstream should never depend on these
//!   symbols directly; we keep them `pub` because Rust's
//!   `pub(workspace)` doesn't exist yet.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// StabilityTier
// ---------------------------------------------------------------------------

/// The three tier values clankers commits to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StabilityTier {
    /// Committed for the 1.x line — breaking changes require a major
    /// version bump and a migration note. New code should aspire here
    /// once the API has been validated by at least one downstream.
    Stable,
    /// Public surface that explicitly reserves the right to break.
    /// Used while the design is being validated.
    Experimental,
    /// Exists for tests / cross-crate plumbing only. Downstream code
    /// should not import these symbols.
    Internal,
}

impl StabilityTier {
    /// Short slug for the tier (CLI flags, docs anchors).
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Experimental => "experimental",
            Self::Internal => "internal",
        }
    }

    /// All tiers, in declaration order.
    pub const ALL: [Self; 3] = [Self::Stable, Self::Experimental, Self::Internal];
}

// ---------------------------------------------------------------------------
// ModuleTier
// ---------------------------------------------------------------------------

/// Declared tier for a single module path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleTier {
    /// Crate-relative module path (e.g. `"clankers_core::env_spec"`).
    pub module: String,
    /// Declared tier.
    pub tier: StabilityTier,
}

// ---------------------------------------------------------------------------
// Canonical tier table
// ---------------------------------------------------------------------------

/// The tiers we currently commit to.
///
/// Keep this list authoritative — CI can grep for module paths
/// against this table to detect undeclared modules. New modules
/// should be added (or marked `Internal` and revisited later).
#[must_use]
pub fn canonical_tier_table() -> Vec<ModuleTier> {
    use StabilityTier::{Experimental, Internal, Stable};
    let t = |module: &str, tier| ModuleTier {
        module: module.to_string(),
        tier,
    };
    vec![
        // clankers-core ---------------------------------------------------
        t("clankers_core::config", Stable),
        t("clankers_core::env_spec", Experimental),
        t("clankers_core::layout", Stable),
        t("clankers_core::manifest", Experimental),
        t("clankers_core::baselines", Experimental),
        t("clankers_core::types", Stable),
        t("clankers_core::seed", Stable),
        t("clankers_core::time", Stable),
        t("clankers_core::traits", Stable),
        t("clankers_core::termination", Experimental),
        t("clankers_core::unified_config", Experimental),
        t("clankers_core::validators", Stable),
        t("clankers_core::view", Stable),
        t("clankers_core::schema", Stable),
        t("clankers_core::physics", Stable),
        t("clankers_core::error", Stable),
        // clankers-physics ------------------------------------------------
        t("clankers_physics::backend", Stable),
        t("clankers_physics::buffers", Experimental),
        t("clankers_physics::components", Stable),
        t("clankers_physics::mirror", Experimental),
        t("clankers_physics::neutral", Experimental),
        t("clankers_physics::plugin", Stable),
        t("clankers_physics::rapier", Internal),
        t("clankers_physics::readback", Experimental),
        t("clankers_physics::scene_objects", Experimental),
        t("clankers_physics::step", Experimental),
        t("clankers_physics::systems", Internal),
        // clankers-gym ----------------------------------------------------
        t("clankers_gym::binary_frame", Stable),
        t("clankers_gym::encoding", Stable),
        t("clankers_gym::env", Stable),
        t("clankers_gym::protocol", Stable),
        t("clankers_gym::server", Stable),
        t("clankers_gym::state_machine", Internal),
        t("clankers_gym::tensor_frame", Experimental),
        t("clankers_gym::vec_env", Stable),
        // clankers-domain-rand -------------------------------------------
        t("clankers_domain_rand::randomizers", Stable),
        t("clankers_domain_rand::ranges", Stable),
        t("clankers_domain_rand::spec", Experimental),
        // clankers-sim ----------------------------------------------------
        t("clankers_sim::builder", Stable),
        t("clankers_sim::scenarios", Stable),
        t("clankers_sim::stats", Stable),
        t("clankers_sim::telemetry", Experimental),
        // clankers-record -------------------------------------------------
        t("clankers_record::async_writer", Experimental),
        // xtask audit -----------------------------------------------------
        t("xtask::audit::code_quality", Internal),
        t("xtask::audit::panic_audit", Internal),
        t("xtask::audit::release", Internal),
        t("xtask::audit::stability", Internal),
        t("xtask::audit::test_layers", Internal),
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
    fn stability_tier_slug_roundtrip() {
        for t in StabilityTier::ALL {
            let s = serde_json::to_string(&t).unwrap();
            let back: StabilityTier = serde_json::from_str(&s).unwrap();
            assert_eq!(t, back);
            assert!(s.contains(t.slug()));
        }
    }

    #[test]
    fn canonical_tier_table_has_no_duplicates() {
        let table = canonical_tier_table();
        let mut seen: HashSet<&str> = HashSet::new();
        for entry in &table {
            assert!(
                seen.insert(entry.module.as_str()),
                "duplicate tier entry for {}",
                entry.module
            );
        }
    }

    #[test]
    fn at_least_a_handful_of_each_tier_is_declared() {
        let table = canonical_tier_table();
        let mut counts = [0usize; 3];
        for entry in &table {
            let idx = match entry.tier {
                StabilityTier::Stable => 0,
                StabilityTier::Experimental => 1,
                StabilityTier::Internal => 2,
            };
            counts[idx] += 1;
        }
        for (i, c) in counts.iter().enumerate() {
            assert!(*c > 0, "tier idx {i} has zero declared modules");
        }
    }
}
