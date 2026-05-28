//! Layered test taxonomy.
//!
//! The workspace has plenty of tests, but they're labelled only by
//! `#[test]` — there's no signal at the crate level distinguishing
//! happy-path unit tests from boundary contract tests from
//! golden-fixture regression tests. This module defines the taxonomy.
//!
//! ## Layers
//!
//! - [`TestLayer::Unit`] — pure-function logic, single module.
//!   No I/O, no async, no Bevy world.
//! - [`TestLayer::Contract`] — exercises a documented protocol
//!   invariant by feeding malformed input across a real boundary
//!   (e.g. a bad `BatchReset.seeds` length to `dispatch_vec`).
//! - [`TestLayer::Golden`] — pinned fixture comparison
//!   (e.g. recorded MCAP byte-equality with a checked-in reference).
//!   Failures indicate behaviour drift even when the unit tests pass.
//! - [`TestLayer::Architecture`] — assertions on workspace
//!   structure rather than runtime behaviour (e.g. "clankers-core
//!   does not depend on rapier3d"). These run as `#[test]` but
//!   examine the cargo graph or module structure.

use serde::{Deserialize, Serialize};

/// One of the four canonical test layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestLayer {
    /// Pure-function logic, single module.
    Unit,
    /// Cross-boundary contract test (malformed input across a real
    /// API boundary).
    Contract,
    /// Pinned fixture comparison (byte-equality, snapshot).
    Golden,
    /// Workspace / module structure assertion.
    Architecture,
}

impl TestLayer {
    /// Short slug for naming conventions (`contract_*`, `golden_*`).
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unit => "unit",
            Self::Contract => "contract",
            Self::Golden => "golden",
            Self::Architecture => "architecture",
        }
    }

    /// All layers in declaration order.
    pub const ALL: [Self; 4] = [Self::Unit, Self::Contract, Self::Golden, Self::Architecture];
}

// ---------------------------------------------------------------------------
// Canonical layered-test inventory
// ---------------------------------------------------------------------------

/// One entry in the layered-test inventory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayeredTest {
    /// Path to the test file.
    pub file: String,
    /// Layer.
    pub layer: TestLayer,
    /// One-line description of what the test pins.
    pub purpose: String,
}

/// The non-Unit tests we've identified in the workspace today.
///
/// Unit tests are not enumerated here — they're the default and don't
/// benefit from inventorying. This list is the spine of Contract /
/// Golden / Architecture coverage; new layered tests should append.
#[must_use]
pub fn layered_test_inventory() -> Vec<LayeredTest> {
    use TestLayer::{Architecture, Contract, Golden};
    let t = |file: &str, layer, purpose: &str| LayeredTest {
        file: file.to_string(),
        layer,
        purpose: purpose.to_string(),
    };
    vec![
        // Contract layer — feed malformed input across boundaries.
        t(
            "crates/clankers-gym/tests/it/batch_reset_validation.rs",
            Contract,
            "BatchReset rejects mismatched seeds/env_ids lengths (P0.2).",
        ),
        t(
            "crates/clankers-env/tests/it/auto_reset_parity.rs",
            Contract,
            "Sequential vs parallel auto-reset produce identical obs sequences (P1.8).",
        ),
        t(
            "crates/clankers-physics/src/scene_objects.rs",
            Contract,
            "RapierContext::resolve_object_body rejects unknown handles (P2.3).",
        ),
        t(
            "crates/clankers-physics/src/step.rs",
            Contract,
            "step_with_buffers rejects buffer/runtime length mismatches (P3.2).",
        ),
        t(
            "crates/clankers-physics/src/mirror.rs",
            Contract,
            "mirror_joint_state_to_ecs / snapshot_joint_torque_from_ecs reject length mismatch (P3.3).",
        ),
        t(
            "crates/clankers-gym/src/tensor_frame.rs",
            Contract,
            "TensorFrameHeader decoder rejects every malformed variant: short buffer, unknown \
             version/dtype/layout, payload-length mismatch (P4.2).",
        ),
        t(
            "python/tests/test_tensor_frame.py",
            Contract,
            "Python tensor-frame decoder rejects every malformed variant in lockstep with the \
             Rust producer (P4.3).",
        ),
        // Golden layer — byte-pinned fixtures.
        t(
            "crates/clankers-gym/src/tensor_frame.rs",
            Golden,
            "TensorFrameHeader byte-size pinned at 48 bytes (`tensor_frame_header_size_is_48_bytes`) \
             so Python and Rust never disagree on offsets.",
        ),
        t(
            "crates/clankers-gym/src/binary_frame.rs",
            Golden,
            "BinaryFrameHeader byte-size pinned at 24 bytes (`binary_frame_header_size_is_24_bytes`).",
        ),
        t(
            "crates/clankers-physics/src/rapier/runtime.rs",
            Golden,
            "JointRuntime struct size pinned <= 256 B to flag inflations that would slow the hot \
             path.",
        ),
        // Architecture layer — workspace structure.
        t(
            "crates/clankers-physics/src/lib.rs",
            Architecture,
            "Prelude re-exports compile (no accidental private types in public surface).",
        ),
        t(
            "xtask/src/audit/stability.rs",
            Architecture,
            "Canonical tier table has no duplicates and covers each tier with at least one entry.",
        ),
        t(
            "xtask/src/audit/panic_audit.rs",
            Architecture,
            "Every FallibleAlternativeExists entry in the panic audit names an alternative.",
        ),
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
    fn slugs_are_unique() {
        let slugs: HashSet<&str> = TestLayer::ALL.iter().map(|l| l.slug()).collect();
        assert_eq!(slugs.len(), TestLayer::ALL.len());
    }

    #[test]
    fn inventory_has_each_non_unit_layer_represented() {
        let inv = layered_test_inventory();
        let layers: HashSet<TestLayer> = inv.iter().map(|t| t.layer).collect();
        assert!(layers.contains(&TestLayer::Contract));
        assert!(layers.contains(&TestLayer::Golden));
        assert!(layers.contains(&TestLayer::Architecture));
    }

    #[test]
    fn inventory_has_no_duplicate_entries() {
        let inv = layered_test_inventory();
        let mut seen: HashSet<(String, TestLayer, String)> = HashSet::new();
        for entry in &inv {
            assert!(
                seen.insert((entry.file.clone(), entry.layer, entry.purpose.clone())),
                "duplicate inventory entry: {entry:?}"
            );
        }
    }
}
