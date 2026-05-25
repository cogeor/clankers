//! Typed schema contracts shared across processes.
//!
//! Four contracts live here:
//!
//! - [`ObservationSchema`] / [`ObservationSlot`] — typed observation slot
//!   table.
//! - [`ActionSchema`] + [`ActionSemantics`] — action encoding.
//! - [`FrameSchema`] / [`RecorderSchema`] — recorder topic schema.
//!
//! Plus the shared [`SchemaMismatch`] error enum.
//!
//! Each schema exposes `version() -> u32` and
//! `validate_against(&other) -> Result<(), SchemaMismatch>` so the Rust
//! server and the Python client can negotiate compatibility before
//! exchanging data.
//!
//! See `docs/plans/WS1-plan.md` § 4-6 for the spec. Note that
//! [`ObservationSchema`] (a slot layout) is intentionally distinct from
//! `crate::types::ObservationSpace` (a Gymnasium-style box / discrete
//! contract); both exist and are exported.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::layout::JointKind;

// ---------------------------------------------------------------------------
// SchemaMismatch
// ---------------------------------------------------------------------------

/// Error returned by every `validate_against` helper on a schema or
/// layout. Variants are named after the divergence they describe so the
/// CLI / Python client can surface a precise message.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SchemaMismatch {
    /// Two schemas reported different `version()` values.
    #[error("schema version mismatch: expected {expected}, found {found}")]
    VersionMismatch {
        /// Version reported by `self` (the local expectation).
        expected: u32,
        /// Version reported by `other` (the remote value).
        found: u32,
    },
    /// Two layouts hold a different number of joints.
    #[error("joint count mismatch: expected {expected}, found {found}")]
    JointCountMismatch {
        /// Joint count reported by `self`.
        expected: usize,
        /// Joint count reported by `other`.
        found: usize,
    },
    /// Two layouts have different joint names at the same index.
    #[error("joint name mismatch at index {index}: expected {expected}, found {found}")]
    JointNameMismatch {
        /// Index of the first divergence.
        index: usize,
        /// Expected joint name (from `self`).
        expected: String,
        /// Joint name observed in `other`.
        found: String,
    },
    /// Two layouts agree on joint names but disagree on the joint kind.
    #[error("joint type mismatch for {name}: expected {expected:?}, found {found:?}")]
    JointTypeMismatch {
        /// Joint name where the kinds diverged.
        name: String,
        /// Joint kind expected by `self`.
        expected: JointKind,
        /// Joint kind observed in `other`.
        found: JointKind,
    },
    /// An observation slot has a different name / dtype / shape.
    #[error("observation slot mismatch ({name}): {reason}")]
    SlotMismatch {
        /// Slot name where the divergence was detected.
        name: String,
        /// Human-readable reason (e.g. "dtype F32 != F64").
        reason: String,
    },
    /// Two action schemas disagree on [`ActionSemantics`].
    #[error("action semantics mismatch: expected {expected:?}, found {found:?}")]
    ActionSemanticsMismatch {
        /// Semantics expected by `self`.
        expected: ActionSemantics,
        /// Semantics observed in `other`.
        found: ActionSemantics,
    },
    /// Two action schemas have different dimensions.
    #[error("action schema dim mismatch: expected {expected}, found {found}")]
    ActionDimMismatch {
        /// Action dimension expected by `self`.
        expected: usize,
        /// Action dimension observed in `other`.
        found: usize,
    },
    /// Two frame schemas disagree on payload encoding for the same channel.
    #[error("encoding mismatch on {channel}: expected {expected}, found {found}")]
    EncodingMismatch {
        /// Channel name where the divergence was detected.
        channel: String,
        /// Human-readable expected encoding (e.g. `"Json"`).
        expected: String,
        /// Human-readable observed encoding.
        found: String,
    },
    /// Two recorder schemas hold a different set of channels.
    #[error("channel set mismatch: missing {missing:?}, unexpected {unexpected:?}")]
    ChannelSetMismatch {
        /// Channels present in `self` but missing from `other`.
        missing: Vec<String>,
        /// Channels present in `other` but not in `self`.
        unexpected: Vec<String>,
    },
}

// ---------------------------------------------------------------------------
// SchemaDtype
// ---------------------------------------------------------------------------

/// Numeric / boolean element type for an [`ObservationSlot`].
///
/// Mirrors the dtype set used by the Python client. Extending this enum
/// is a breaking change — bump [`ObservationSchema::SCHEMA_VERSION`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SchemaDtype {
    /// 32-bit IEEE float.
    F32,
    /// 64-bit IEEE float.
    F64,
    /// 8-bit unsigned integer.
    U8,
    /// 32-bit signed integer.
    I32,
    /// Boolean (typically packed by the consumer).
    Bool,
}

// ---------------------------------------------------------------------------
// ObservationSlot / ObservationSchema
// ---------------------------------------------------------------------------

/// One typed slot in an [`ObservationSchema`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationSlot {
    /// Slot name. Unique within an [`ObservationSchema`].
    pub name: String,
    /// Element dtype.
    pub dtype: SchemaDtype,
    /// Tensor shape (row-major).
    pub shape: Vec<usize>,
    /// Human-readable physical units (e.g. `"rad"`, `"m/s"`).
    pub units: Option<String>,
    /// Sensor name that produces this slot (for traceability).
    pub source_sensor: String,
}

impl ObservationSlot {
    /// Number of scalar elements (product of `shape`).
    #[must_use]
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Typed observation slot table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationSchema {
    /// Slots in canonical order. Order is part of the contract.
    pub slots: Vec<ObservationSlot>,
    /// Schema version. See [`Self::SCHEMA_VERSION`].
    pub version: u32,
}

impl ObservationSchema {
    /// Schema version for the [`ObservationSchema`] wire format.
    pub const SCHEMA_VERSION: u32 = 1;

    /// Schema version of this instance.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Validate that `other` is compatible. Checks version, slot count,
    /// and per-slot name / dtype / shape.
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
        if self.slots.len() != other.slots.len() {
            return Err(SchemaMismatch::SlotMismatch {
                name: String::new(),
                reason: format!("slot count {} != {}", self.slots.len(), other.slots.len()),
            });
        }
        for (a, b) in self.slots.iter().zip(other.slots.iter()) {
            if a.name != b.name {
                return Err(SchemaMismatch::SlotMismatch {
                    name: a.name.clone(),
                    reason: format!("name {} != {}", a.name, b.name),
                });
            }
            if a.dtype != b.dtype {
                return Err(SchemaMismatch::SlotMismatch {
                    name: a.name.clone(),
                    reason: format!("dtype {:?} != {:?}", a.dtype, b.dtype),
                });
            }
            if a.shape != b.shape {
                return Err(SchemaMismatch::SlotMismatch {
                    name: a.name.clone(),
                    reason: format!("shape {:?} != {:?}", a.shape, b.shape),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ActionSchema / ActionSemantics
// ---------------------------------------------------------------------------

/// Interpretation of the f32 vector exchanged through an [`ActionSchema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionSemantics {
    /// Values in `[-1, 1]`, scaled to joint limits by the consumer.
    NormalizedPosition,
    /// Absolute joint positions (rad / m), already in physical units.
    AbsoluteJointPosition,
    /// Joint velocities (rad/s / m/s).
    JointVelocity,
    /// Torques / forces (Nm / N).
    Torque,
}

/// Typed action schema for an environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionSchema {
    /// Interpretation of the action vector.
    pub semantics: ActionSemantics,
    /// Action vector dimension.
    pub dim: usize,
    /// Optional per-dimension lower bounds.
    pub low: Option<Vec<f32>>,
    /// Optional per-dimension upper bounds.
    pub high: Option<Vec<f32>>,
    /// Schema version. See [`Self::SCHEMA_VERSION`].
    pub version: u32,
}

impl ActionSchema {
    /// Schema version for the [`ActionSchema`] wire format.
    pub const SCHEMA_VERSION: u32 = 1;

    /// Schema version of this instance.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Validate compatibility. Checks version, semantics, and dim.
    /// Bounds (`low` / `high`) are not compared because cross-process
    /// float wobble would produce false negatives.
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
        if self.semantics != other.semantics {
            return Err(SchemaMismatch::ActionSemanticsMismatch {
                expected: self.semantics,
                found: other.semantics,
            });
        }
        if self.dim != other.dim {
            return Err(SchemaMismatch::ActionDimMismatch {
                expected: self.dim,
                found: other.dim,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FrameSchema / FrameEncoding / RecorderSchema
// ---------------------------------------------------------------------------

/// Payload encoding for a recorder [`FrameSchema`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrameEncoding {
    /// JSON text payload.
    Json,
    /// ROS 2 CDR-serialised payload.
    Cdr,
    /// Opaque byte blob (e.g. JPEG, custom binary).
    RawBytes,
    /// Protobuf message, fully-qualified name carried inline.
    ProtobufFqn(String),
}

impl FrameEncoding {
    /// Short human-readable tag for error messages.
    #[must_use]
    pub fn tag(&self) -> String {
        match self {
            Self::Json => "Json".into(),
            Self::Cdr => "Cdr".into(),
            Self::RawBytes => "RawBytes".into(),
            Self::ProtobufFqn(name) => format!("ProtobufFqn({name})"),
        }
    }
}

/// Per-channel recorder frame schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameSchema {
    /// Channel name (e.g. `"/joints"`).
    pub channel: String,
    /// Logical message type (e.g. `"JointState"`).
    pub message_type: String,
    /// Payload encoding.
    pub encoding: FrameEncoding,
    /// Schema version. See [`Self::SCHEMA_VERSION`].
    pub version: u32,
}

impl FrameSchema {
    /// Schema version for the [`FrameSchema`] wire format.
    pub const SCHEMA_VERSION: u32 = 1;

    /// Schema version of this instance.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Validate compatibility. Checks version, message type, and encoding.
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
        if self.message_type != other.message_type || self.encoding != other.encoding {
            return Err(SchemaMismatch::EncodingMismatch {
                channel: self.channel.clone(),
                expected: format!("{}/{}", self.message_type, self.encoding.tag()),
                found: format!("{}/{}", other.message_type, other.encoding.tag()),
            });
        }
        Ok(())
    }
}

/// Recorder-wide schema. Holds an ordered set of per-channel frame
/// schemas plus a version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecorderSchema {
    /// Channels in canonical order. Order is part of the contract.
    pub channels: Vec<FrameSchema>,
    /// Schema version. See [`Self::SCHEMA_VERSION`].
    pub version: u32,
}

impl RecorderSchema {
    /// Schema version for the [`RecorderSchema`] wire format.
    pub const SCHEMA_VERSION: u32 = 1;

    /// Schema version of this instance.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Validate compatibility. Checks version, then channel set equality,
    /// then runs [`FrameSchema::validate_against`] per channel.
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
        let self_names: Vec<&str> = self.channels.iter().map(|c| c.channel.as_str()).collect();
        let other_names: Vec<&str> = other.channels.iter().map(|c| c.channel.as_str()).collect();
        let missing: Vec<String> = self_names
            .iter()
            .filter(|n| !other_names.contains(n))
            .map(|s| (*s).to_owned())
            .collect();
        let unexpected: Vec<String> = other_names
            .iter()
            .filter(|n| !self_names.contains(n))
            .map(|s| (*s).to_owned())
            .collect();
        if !missing.is_empty() || !unexpected.is_empty() {
            return Err(SchemaMismatch::ChannelSetMismatch {
                missing,
                unexpected,
            });
        }
        for a in &self.channels {
            let Some(b) = other.channels.iter().find(|c| c.channel == a.channel) else {
                // Already covered by the set-equality check above; defensive.
                return Err(SchemaMismatch::ChannelSetMismatch {
                    missing: vec![a.channel.clone()],
                    unexpected: vec![],
                });
            };
            a.validate_against(b)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Inline tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_action(dim: usize) -> ActionSchema {
        ActionSchema {
            semantics: ActionSemantics::NormalizedPosition,
            dim,
            low: None,
            high: None,
            version: ActionSchema::SCHEMA_VERSION,
        }
    }

    #[test]
    fn observation_slot_shape_size() {
        let slot = ObservationSlot {
            name: "image".into(),
            dtype: SchemaDtype::U8,
            shape: vec![64, 64, 3],
            units: None,
            source_sensor: "cam".into(),
        };
        assert_eq!(slot.size(), 64 * 64 * 3);
        assert_eq!(slot.size(), slot.shape.iter().product::<usize>());
    }

    #[test]
    fn action_schema_validate_against_dim_mismatch_is_err() {
        let a = make_action(6);
        let b = make_action(7);
        let err = a.validate_against(&b).unwrap_err();
        assert!(matches!(err, SchemaMismatch::ActionDimMismatch { .. }));
    }

    #[test]
    fn frame_schema_validate_against_encoding_mismatch_is_err() {
        let a = FrameSchema {
            channel: "/joints".into(),
            message_type: "JointState".into(),
            encoding: FrameEncoding::Json,
            version: FrameSchema::SCHEMA_VERSION,
        };
        let mut b = a.clone();
        b.encoding = FrameEncoding::Cdr;
        let err = a.validate_against(&b).unwrap_err();
        assert!(matches!(err, SchemaMismatch::EncodingMismatch { .. }));
    }
}
