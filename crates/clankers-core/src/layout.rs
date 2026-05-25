//! `JointLayout` — ordered, hashable, deterministic joint layout.
//!
//! The layout is the single source of truth for joint ordering across
//! every workspace process: simulator, sensors, recorder, Python client.
//!
//! Construction must be deterministic — the same URDF in must produce the
//! same hash out, regardless of `HashMap` iteration order or thread
//! scheduling. See `docs/plans/WS1-plan.md` and the quality report
//! finding #1 ("Deterministic Joint Layout Is Promised But Not
//! Guaranteed") for the original motivation.
//!
//! ## Hashing & equality
//!
//! [`JointLayout`] implements [`std::hash::Hash`] and [`Eq`] structurally
//! over `(version, [(name, joint_type), ..])`. Floating-point limits and
//! axes are deliberately **not** folded into the structural hash — they
//! can wobble across parser revisions even when the joint topology is
//! identical. Limits are exposed via the separate [`JointLayout::limits_hash`]
//! helper, which uses [`f32::to_bits`] for bitwise stability and is
//! documented as stable only for the same URDF parsed by the same parser
//! version.
//!
//! `PartialEq` is derived and compares all fields including limits, so
//! `a == b` is *stricter* than `hash(a) == hash(b)`. This matches the
//! pattern recommended by the standard library — `Hash` may be coarser
//! than `Eq` but never finer.

use std::hash::{Hash, Hasher};

use bevy::ecs::entity::Entity;
use serde::{Deserialize, Serialize};

use crate::schema::SchemaMismatch;

// ---------------------------------------------------------------------------
// JointKind
// ---------------------------------------------------------------------------

/// URDF-style joint kinds.
///
/// Mirrors `clankers_urdf::types::JointType` but lives in `clankers-core`
/// so the layout has no URDF dependency (avoids a reverse crate cycle,
/// since `clankers-urdf` depends on `clankers-core`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JointKind {
    /// Rotation about a single axis, with position limits.
    Revolute,
    /// Unlimited rotation about a single axis.
    Continuous,
    /// Translation along an axis, with position limits.
    Prismatic,
    /// No relative motion between parent and child.
    Fixed,
    /// Unconstrained 6-DOF joint (rarely used).
    Floating,
    /// Translation along one axis with no rotation (rarely used).
    Planar,
}

impl JointKind {
    /// Whether this joint kind contributes actuatable degrees of freedom.
    #[must_use]
    pub const fn is_actuated(self) -> bool {
        matches!(self, Self::Revolute | Self::Continuous | Self::Prismatic)
    }
}

// ---------------------------------------------------------------------------
// JointSpecLimits
// ---------------------------------------------------------------------------

/// Position / effort / velocity limits on a single joint.
///
/// Mirrors `clankers_urdf::types::JointLimits`. Held by value inside
/// [`JointSpec`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct JointSpecLimits {
    /// Lower position limit (rad or m). `None` means unbounded.
    pub lower: Option<f32>,
    /// Upper position limit (rad or m). `None` means unbounded.
    pub upper: Option<f32>,
    /// Maximum effort (Nm or N).
    pub effort: f32,
    /// Maximum velocity (rad/s or m/s).
    pub velocity: f32,
}

// ---------------------------------------------------------------------------
// JointSpec
// ---------------------------------------------------------------------------

/// One joint's identity within a layout.
///
/// `entity` is `Option` because the layout is built from a URDF (no
/// `Entity` yet) and only later bound to a spawned robot. The `entity`
/// field is **not** serialised — it is a runtime handle into the Bevy
/// world and has no meaning across processes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointSpec {
    /// Joint name. Unique within a layout.
    pub name: String,
    /// Bevy entity bound to this joint, once the robot has been spawned.
    /// `None` for layouts built from a URDF prior to spawning. Not
    /// serialised — see type docs.
    #[serde(skip)]
    pub entity: Option<Entity>,
    /// Kind of joint (revolute, fixed, ...).
    pub joint_type: JointKind,
    /// Motion limits.
    pub limits: JointSpecLimits,
    /// Joint axis (unit vector).
    pub axis: [f32; 3],
}

// ---------------------------------------------------------------------------
// JointLayout
// ---------------------------------------------------------------------------

/// Ordered, hashable, versioned joint layout.
///
/// Construction is deterministic — the same URDF in must produce the
/// same hash out. Use [`JointLayout::validate_against`] to confirm two
/// layouts are structurally compatible (e.g. between a Rust server and
/// a Python client).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointLayout {
    joints: Vec<JointSpec>,
    version: u32,
}

impl JointLayout {
    /// Schema version of the layout wire format. Bumped on breaking
    /// changes to the (name, kind) ordering invariant.
    pub const SCHEMA_VERSION: u32 = 1;

    /// Build a layout from an explicit ordered list of specs.
    ///
    /// Callers are responsible for sorting the input deterministically.
    /// Use [`JointLayoutBuilder`] for an alphabetic sort.
    #[must_use]
    pub const fn new(joints: Vec<JointSpec>) -> Self {
        Self {
            joints,
            version: Self::SCHEMA_VERSION,
        }
    }

    /// Start building a new layout with deterministic ordering.
    #[must_use]
    pub fn builder() -> JointLayoutBuilder {
        JointLayoutBuilder::default()
    }

    /// Schema version of this layout. See [`Self::SCHEMA_VERSION`].
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Borrow the ordered list of joint specs.
    #[must_use]
    pub fn joints(&self) -> &[JointSpec] {
        &self.joints
    }

    /// Iterate the joint names in layout order.
    pub fn joint_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.joints.iter().map(|j| j.name.as_str())
    }

    /// Number of actuated degrees of freedom (revolute / continuous /
    /// prismatic). Fixed and floating joints are excluded.
    #[must_use]
    pub fn dof(&self) -> usize {
        self.joints
            .iter()
            .filter(|j| j.joint_type.is_actuated())
            .count()
    }

    /// Total number of joint slots in the layout (actuated or not).
    ///
    /// This is the dimension callers should use to size sensor and
    /// action buffers indexed by layout slot. Contrast with
    /// [`Self::dof`], which counts only actuated joints.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.joints.len()
    }

    /// Whether the layout has zero joints.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    /// Iterate the per-slot Bevy entities, in layout order.
    ///
    /// Each yielded value is `Some(entity)` once the layout has been
    /// bound to a spawned robot via [`Self::bind_entities`], or `None`
    /// otherwise (e.g. for layouts built from a URDF prior to spawning).
    pub fn entities(&self) -> impl Iterator<Item = Option<Entity>> + '_ {
        self.joints.iter().map(|j| j.entity)
    }

    /// Iterate only the joint entities that have been bound, in layout
    /// order. Skips slots where `entity` is `None`.
    pub fn bound_entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.joints.iter().filter_map(|j| j.entity)
    }

    /// Bind a Bevy entity to each joint slot, in layout order.
    ///
    /// Writes `entities[i]` into the `entity` field of joint slot `i`
    /// for every slot. Callers are responsible for ensuring that the
    /// `entities` slice is in the same order as [`Self::joints`].
    ///
    /// # Panics
    ///
    /// Debug builds panic if `entities.len() != self.len()`. Release
    /// builds bind as many slots as are provided and ignore the
    /// remainder (caller error).
    pub fn bind_entities(&mut self, entities: &[Entity]) {
        debug_assert_eq!(
            entities.len(),
            self.joints.len(),
            "bind_entities: expected {} entities, got {}",
            self.joints.len(),
            entities.len()
        );
        for (slot, &entity) in self.joints.iter_mut().zip(entities.iter()) {
            slot.entity = Some(entity);
        }
    }

    /// Look up the index of a joint by name.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|j| j.name == name)
    }

    /// Validate that `other` is structurally compatible with `self`.
    ///
    /// Checks version, joint count, and per-index `(name, joint_type)`.
    /// Limits / axes are NOT compared — use [`Self::limits_hash`] for
    /// that.
    ///
    /// # Errors
    /// Returns a [`SchemaMismatch`] describing the first divergence.
    pub fn validate_against(&self, other: &Self) -> Result<(), SchemaMismatch> {
        if self.version != other.version {
            return Err(SchemaMismatch::VersionMismatch {
                expected: self.version,
                found: other.version,
            });
        }
        if self.joints.len() != other.joints.len() {
            return Err(SchemaMismatch::JointCountMismatch {
                expected: self.joints.len(),
                found: other.joints.len(),
            });
        }
        for (i, (a, b)) in self.joints.iter().zip(other.joints.iter()).enumerate() {
            if a.name != b.name {
                return Err(SchemaMismatch::JointNameMismatch {
                    index: i,
                    expected: a.name.clone(),
                    found: b.name.clone(),
                });
            }
            if a.joint_type != b.joint_type {
                return Err(SchemaMismatch::JointTypeMismatch {
                    name: a.name.clone(),
                    expected: a.joint_type,
                    found: b.joint_type,
                });
            }
        }
        Ok(())
    }

    /// `SipHasher24` digest over the float-valued fields of every joint.
    ///
    /// Stable for the same URDF parsed by the same parser version. Do
    /// not compare across major parser revisions — float bit patterns
    /// may shift even when the joint topology is identical.
    #[must_use]
    pub fn limits_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.version.hash(&mut hasher);
        for j in &self.joints {
            j.name.hash(&mut hasher);
            j.limits
                .lower
                .unwrap_or(f32::NAN)
                .to_bits()
                .hash(&mut hasher);
            j.limits
                .upper
                .unwrap_or(f32::NAN)
                .to_bits()
                .hash(&mut hasher);
            j.limits.effort.to_bits().hash(&mut hasher);
            j.limits.velocity.to_bits().hash(&mut hasher);
            for component in j.axis {
                component.to_bits().hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Test-only setter for the schema version. Used by the
    /// `layout_validate_against_version_mismatch_is_err` integration
    /// test; not part of the public contract.
    #[doc(hidden)]
    pub const fn set_version_for_test(&mut self, version: u32) {
        self.version = version;
    }
}

impl Hash for JointLayout {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.version.hash(state);
        self.joints.len().hash(state);
        for j in &self.joints {
            j.name.hash(state);
            j.joint_type.hash(state);
        }
    }
}

impl Eq for JointLayout {}

// ---------------------------------------------------------------------------
// JointLayoutBuilder
// ---------------------------------------------------------------------------

/// Deterministic builder for [`JointLayout`].
///
/// Joints pushed in arbitrary order are alphabetically sorted by name
/// on [`Self::build`], giving the same layout regardless of insertion
/// order.
#[derive(Debug, Default, Clone)]
pub struct JointLayoutBuilder {
    joints: Vec<JointSpec>,
}

impl JointLayoutBuilder {
    /// Append a spec. The final order is determined by [`Self::build`].
    #[must_use]
    pub fn push(mut self, spec: JointSpec) -> Self {
        self.joints.push(spec);
        self
    }

    /// Finalise into a layout, alphabetically sorted by joint name.
    #[must_use]
    pub fn build(mut self) -> JointLayout {
        self.joints.sort_by(|a, b| a.name.cmp(&b.name));
        JointLayout::new(self.joints)
    }
}

// ---------------------------------------------------------------------------
// Inline tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use super::*;

    fn spec(name: &str, kind: JointKind) -> JointSpec {
        JointSpec {
            name: name.into(),
            entity: None,
            joint_type: kind,
            limits: JointSpecLimits::default(),
            axis: [0.0, 0.0, 1.0],
        }
    }

    fn hash<T: Hash>(t: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        t.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn joint_spec_eq_and_hash() {
        // Two equal layouts hash equally.
        let a = JointLayout::new(vec![spec("a", JointKind::Revolute)]);
        let b = JointLayout::new(vec![spec("a", JointKind::Revolute)]);
        assert_eq!(a, b);
        assert_eq!(hash(&a), hash(&b));

        // Differing on joint_type produces different hashes.
        let c = JointLayout::new(vec![spec("a", JointKind::Prismatic)]);
        assert_ne!(a, c);
        assert_ne!(hash(&a), hash(&c));
    }

    #[test]
    fn joint_layout_index_of_round_trip() {
        let layout = JointLayout::builder()
            .push(spec("b", JointKind::Revolute))
            .push(spec("a", JointKind::Revolute))
            .push(spec("c", JointKind::Fixed))
            .build();
        for (i, name) in layout.joint_names().enumerate() {
            assert_eq!(layout.index_of(name), Some(i));
        }
        assert_eq!(layout.index_of("missing"), None);
    }

    #[test]
    fn joint_layout_dof_matches_actuated_count() {
        let layout = JointLayout::new(vec![
            spec("a", JointKind::Revolute),
            spec("b", JointKind::Fixed),
            spec("c", JointKind::Prismatic),
            spec("d", JointKind::Floating),
            spec("e", JointKind::Continuous),
        ]);
        assert_eq!(layout.dof(), 3);
    }

    #[test]
    fn joint_layout_len_and_is_empty() {
        let empty = JointLayout::new(vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let layout = JointLayout::new(vec![
            spec("a", JointKind::Revolute),
            spec("b", JointKind::Fixed),
        ]);
        assert_eq!(layout.len(), 2);
        assert!(!layout.is_empty());
    }

    #[test]
    fn joint_layout_entities_returns_options_in_order() {
        let layout = JointLayout::new(vec![
            spec("a", JointKind::Revolute),
            spec("b", JointKind::Revolute),
        ]);
        let entities: Vec<Option<Entity>> = layout.entities().collect();
        assert_eq!(entities.len(), 2);
        assert!(entities.iter().all(Option::is_none));
        // bound_entities yields nothing when nothing has been bound.
        assert_eq!(layout.bound_entities().count(), 0);
    }

    #[test]
    fn joint_layout_bind_entities_round_trips() {
        let mut layout = JointLayout::new(vec![
            spec("a", JointKind::Revolute),
            spec("b", JointKind::Revolute),
            spec("c", JointKind::Revolute),
        ]);
        let bound = [
            Entity::from_bits(11),
            Entity::from_bits(22),
            Entity::from_bits(33),
        ];
        layout.bind_entities(&bound);

        let entities: Vec<Option<Entity>> = layout.entities().collect();
        assert_eq!(
            entities,
            bound.iter().copied().map(Some).collect::<Vec<_>>()
        );

        let bound_only: Vec<Entity> = layout.bound_entities().collect();
        assert_eq!(bound_only, bound);
    }

    #[test]
    #[should_panic(expected = "bind_entities: expected 3 entities, got 2")]
    fn joint_layout_bind_entities_wrong_count_panics_in_debug() {
        let mut layout = JointLayout::new(vec![
            spec("a", JointKind::Revolute),
            spec("b", JointKind::Revolute),
            spec("c", JointKind::Revolute),
        ]);
        layout.bind_entities(&[Entity::from_bits(1), Entity::from_bits(2)]);
    }

    #[test]
    fn joint_layout_bind_entities_overwrites_previous_binding() {
        let mut layout = JointLayout::new(vec![spec("a", JointKind::Revolute)]);
        layout.bind_entities(&[Entity::from_bits(1)]);
        layout.bind_entities(&[Entity::from_bits(99)]);
        let bound: Vec<Entity> = layout.bound_entities().collect();
        assert_eq!(bound, vec![Entity::from_bits(99)]);
    }
}
