//! Teleop configuration types.
//!
//! [`TeleopConfig`] defines how input channels map to joint commands,
//! including scaling and dead-zone parameters.

use std::collections::HashMap;

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// JointMapping
// ---------------------------------------------------------------------------

/// Maps a named input channel to a joint entity with scaling.
#[derive(Clone, Debug)]
pub struct JointMapping {
    /// Target joint entity.
    pub entity: Entity,
    /// Scale factor applied to the raw input value.
    pub scale: f32,
    /// Values below this threshold are treated as zero.
    pub dead_zone: f32,
}

impl JointMapping {
    /// Create a mapping with default scale (1.0) and no dead zone.
    #[must_use]
    pub const fn new(entity: Entity) -> Self {
        Self {
            entity,
            scale: 1.0,
            dead_zone: 0.0,
        }
    }

    /// Set the scale factor.
    #[must_use]
    pub const fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set the dead zone threshold.
    #[must_use]
    pub const fn with_dead_zone(mut self, dead_zone: f32) -> Self {
        self.dead_zone = dead_zone;
        self
    }

    /// Apply dead zone and scaling to a raw input value.
    #[must_use]
    pub fn apply(&self, raw: f32) -> f32 {
        if raw.abs() < self.dead_zone {
            0.0
        } else {
            raw * self.scale
        }
    }
}

// ---------------------------------------------------------------------------
// TeleopConfig
// ---------------------------------------------------------------------------

/// Configuration resource mapping named input channels to joint commands.
///
/// Input channels are arbitrary string names (e.g., `"axis_0"`, `"key_up"`).
/// External code writes raw values to [`TeleopCommander`](super::commander::TeleopCommander),
/// and the teleop system applies them to the mapped joints.
#[derive(Resource, Clone, Debug, Default)]
pub struct TeleopConfig {
    /// Map from input channel name to joint mapping.
    pub mappings: HashMap<String, JointMapping>,
    /// Whether teleop is active (commands are applied).
    pub enabled: bool,
}

impl TeleopConfig {
    /// Create an enabled config with no mappings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
            enabled: true,
        }
    }

    /// Add a mapping from a named channel to a joint entity.
    #[must_use]
    pub fn with_mapping(mut self, channel: impl Into<String>, mapping: JointMapping) -> Self {
        self.mappings.insert(channel.into(), mapping);
        self
    }

    /// Set enabled state.
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_entity() -> Entity {
        let mut world = World::new();
        world.spawn_empty().id()
    }

    #[test]
    fn joint_mapping_default_scale() {
        let mapping = JointMapping::new(dummy_entity());
        assert!((mapping.scale - 1.0).abs() < f32::EPSILON);
        assert!((mapping.dead_zone - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn joint_mapping_apply_no_dead_zone() {
        let mapping = JointMapping::new(dummy_entity()).with_scale(2.0);
        assert!((mapping.apply(0.5) - 1.0).abs() < f32::EPSILON);
        assert!((mapping.apply(-0.5) - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn joint_mapping_apply_with_dead_zone() {
        let mapping = JointMapping::new(dummy_entity()).with_dead_zone(0.1);
        assert!((mapping.apply(0.05)).abs() < f32::EPSILON);
        assert!((mapping.apply(0.5) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn teleop_config_builder() {
        let config = TeleopConfig::new()
            .with_mapping("axis_0", JointMapping::new(dummy_entity()).with_scale(5.0))
            .with_enabled(true);

        assert!(config.enabled);
        assert_eq!(config.mappings.len(), 1);
        let m = &config.mappings["axis_0"];
        assert!((m.scale - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn teleop_config_default_disabled() {
        let config = TeleopConfig::default();
        assert!(!config.enabled);
        assert!(config.mappings.is_empty());
    }

    #[test]
    fn joint_mapping_dead_zone_negative_input() {
        let mapping = JointMapping::new(dummy_entity()).with_dead_zone(0.2);
        assert!((mapping.apply(-0.1)).abs() < f32::EPSILON);
        assert!((mapping.apply(-0.5) - (-0.5)).abs() < f32::EPSILON);
    }
}
