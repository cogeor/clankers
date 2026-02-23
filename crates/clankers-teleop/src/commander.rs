//! Teleop command buffer.
//!
//! [`TeleopCommander`] is a Bevy resource that stores raw input values
//! from any input source. The teleop system reads these values and applies
//! them to joint commands via the configured mappings.

use std::collections::HashMap;

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// TeleopCommander
// ---------------------------------------------------------------------------

/// Resource that buffers raw input values from any source.
///
/// External code (keyboard handlers, gamepad readers, network input, etc.)
/// writes values here using [`set`](Self::set). The teleop system reads
/// and applies them each frame.
///
/// # Example
///
/// ```
/// use clankers_teleop::TeleopCommander;
///
/// let mut commander = TeleopCommander::new();
/// commander.set("axis_0", 0.5);
/// commander.set("axis_1", -1.0);
///
/// assert!((commander.get("axis_0") - 0.5).abs() < f32::EPSILON);
/// assert!((commander.get("missing")).abs() < f32::EPSILON);
/// ```
#[derive(Resource, Clone, Debug, Default)]
pub struct TeleopCommander {
    values: HashMap<String, f32>,
}

impl TeleopCommander {
    /// Create an empty commander.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a raw input value for a named channel.
    pub fn set(&mut self, channel: impl Into<String>, value: f32) {
        self.values.insert(channel.into(), value);
    }

    /// Get the current value for a channel (0.0 if unset).
    #[must_use]
    pub fn get(&self, channel: &str) -> f32 {
        self.values.get(channel).copied().unwrap_or(0.0)
    }

    /// Clear all input values to zero.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Number of active channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.values.len()
    }

    /// Iterator over all (channel, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, f32)> {
        self.values.iter().map(|(k, &v)| (k.as_str(), v))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commander_default_empty() {
        let commander = TeleopCommander::new();
        assert_eq!(commander.channel_count(), 0);
        assert!((commander.get("anything")).abs() < f32::EPSILON);
    }

    #[test]
    fn commander_set_and_get() {
        let mut commander = TeleopCommander::new();
        commander.set("axis_0", 0.75);
        commander.set("axis_1", -0.5);

        assert!((commander.get("axis_0") - 0.75).abs() < f32::EPSILON);
        assert!((commander.get("axis_1") - (-0.5)).abs() < f32::EPSILON);
        assert_eq!(commander.channel_count(), 2);
    }

    #[test]
    fn commander_overwrite() {
        let mut commander = TeleopCommander::new();
        commander.set("axis_0", 1.0);
        commander.set("axis_0", 2.0);
        assert!((commander.get("axis_0") - 2.0).abs() < f32::EPSILON);
        assert_eq!(commander.channel_count(), 1);
    }

    #[test]
    fn commander_clear() {
        let mut commander = TeleopCommander::new();
        commander.set("a", 1.0);
        commander.set("b", 2.0);
        commander.clear();
        assert_eq!(commander.channel_count(), 0);
        assert!((commander.get("a")).abs() < f32::EPSILON);
    }

    #[test]
    fn commander_iter() {
        let mut commander = TeleopCommander::new();
        commander.set("x", 1.0);
        commander.set("y", 2.0);

        assert_eq!(commander.iter().count(), 2);
    }
}
