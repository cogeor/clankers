//! Domain-randomisation specification (G11).
//!
//! CODE_QUALITY_REVIEW § "Gap 11: Domain Randomisation Lacks A First-
//! Class Contract". The `clankers-domain-rand` crate ships range types
//! today, but each task wires randomisers ad-hoc; there's no
//! declaration tied to [`crate::env_spec::EnvSpec`] that says "this
//! task randomises gravity over [9.0, 10.5] m/s² and friction over
//! [0.4, 0.6]".
//!
//! [`DomainRandomizationSpec`] is the declaration. Producers stamp it
//! into the [`crate::manifest::RunManifest`] so consumers can know
//! exactly which randomisation distribution generated the data.
//!
//! The spec is intentionally narrow today — covers the parameters
//! we randomise in-tree (gravity, mass, friction, restitution,
//! joint damping). Adding a parameter is a single struct field
//! + a serde-default; downstream consumers absorbing the change
//! get the new field as `None` (= "not randomised") without breaking.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Range
// ---------------------------------------------------------------------------

/// Closed sampling range `[low, high]`. `low <= high` is enforced by
/// the [`Range::new`] constructor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Range {
    /// Low (inclusive) end of the range.
    pub low: f32,
    /// High (inclusive) end of the range.
    pub high: f32,
}

impl Range {
    /// Build a range, swapping `low` and `high` if the caller supplied
    /// them out of order. Use [`Self::try_new`] to surface the error
    /// instead of silently fixing it.
    #[must_use]
    pub fn new(low: f32, high: f32) -> Self {
        if low <= high {
            Self { low, high }
        } else {
            Self {
                low: high,
                high: low,
            }
        }
    }

    /// Strict constructor. Returns `None` when `low > high`.
    #[must_use]
    pub const fn try_new(low: f32, high: f32) -> Option<Self> {
        if low <= high {
            Some(Self { low, high })
        } else {
            None
        }
    }

    /// Centre of the range.
    #[must_use]
    pub fn centre(&self) -> f32 {
        0.5 * (self.low + self.high)
    }

    /// Half-width of the range (radius).
    #[must_use]
    pub fn half_width(&self) -> f32 {
        0.5 * (self.high - self.low)
    }
}

// ---------------------------------------------------------------------------
// DomainRandomizationSpec
// ---------------------------------------------------------------------------

/// Per-episode parameter randomisation declaration.
///
/// All fields are `Option<Range>`; `None` means "not randomised — use
/// the env default". The wire shape stays additive across stack
/// versions because new fields default to `None`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DomainRandomizationSpec {
    /// Per-axis gravity randomisation in m/s². Typical:
    /// `Range::new(9.5, 10.2)`.
    #[serde(default)]
    pub gravity_z: Option<Range>,
    /// Global mass multiplier applied to every dynamic body.
    #[serde(default)]
    pub mass_multiplier: Option<Range>,
    /// Global friction coefficient range.
    #[serde(default)]
    pub friction: Option<Range>,
    /// Global restitution (bounciness) range.
    #[serde(default)]
    pub restitution: Option<Range>,
    /// Multiplier on joint damping (PD `kd`) per episode.
    #[serde(default)]
    pub joint_damping_multiplier: Option<Range>,
    /// Per-joint per-step actuation noise std (in N or Nm).
    #[serde(default)]
    pub actuation_noise_std: Option<Range>,
}

impl DomainRandomizationSpec {
    /// Whether this spec randomises anything. Manifest-stampers use
    /// this to skip writing a no-op section.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.gravity_z.is_some()
            || self.mass_multiplier.is_some()
            || self.friction.is_some()
            || self.restitution.is_some()
            || self.joint_damping_multiplier.is_some()
            || self.actuation_noise_std.is_some()
    }

    /// Count of fields actively randomised. Useful for inspect /
    /// summary surfaces.
    #[must_use]
    pub const fn active_field_count(&self) -> usize {
        let mut n = 0;
        if self.gravity_z.is_some() {
            n += 1;
        }
        if self.mass_multiplier.is_some() {
            n += 1;
        }
        if self.friction.is_some() {
            n += 1;
        }
        if self.restitution.is_some() {
            n += 1;
        }
        if self.joint_damping_multiplier.is_some() {
            n += 1;
        }
        if self.actuation_noise_std.is_some() {
            n += 1;
        }
        n
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_new_swaps_inverted_bounds() {
        let r = Range::new(5.0, 1.0);
        assert!((r.low - 1.0).abs() < 1e-6);
        assert!((r.high - 5.0).abs() < 1e-6);
    }

    #[test]
    fn range_try_new_rejects_inverted_bounds() {
        assert!(Range::try_new(5.0, 1.0).is_none());
        assert!(Range::try_new(1.0, 5.0).is_some());
    }

    #[test]
    fn range_centre_and_half_width() {
        let r = Range::new(0.0, 10.0);
        assert!((r.centre() - 5.0).abs() < 1e-6);
        assert!((r.half_width() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn spec_default_is_inactive() {
        let s = DomainRandomizationSpec::default();
        assert!(!s.is_active());
        assert_eq!(s.active_field_count(), 0);
    }

    #[test]
    fn spec_with_some_field_is_active() {
        let s = DomainRandomizationSpec {
            gravity_z: Some(Range::new(9.5, 10.2)),
            ..DomainRandomizationSpec::default()
        };
        assert!(s.is_active());
        assert_eq!(s.active_field_count(), 1);
    }

    #[test]
    fn spec_roundtrips_through_json() {
        let s = DomainRandomizationSpec {
            gravity_z: Some(Range::new(9.5, 10.2)),
            friction: Some(Range::new(0.4, 0.6)),
            ..DomainRandomizationSpec::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: DomainRandomizationSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
        assert_eq!(back.active_field_count(), 2);
    }

    #[test]
    fn spec_with_only_active_fields_serialised() {
        let s = DomainRandomizationSpec {
            friction: Some(Range::new(0.4, 0.6)),
            ..DomainRandomizationSpec::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        // Inactive fields are serialised as `null`; new producers can
        // strip them with `skip_serializing_if = "Option::is_none"`
        // once we land the manifest stamper.
        assert!(json.contains("\"friction\""));
    }
}
