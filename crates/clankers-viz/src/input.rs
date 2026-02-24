//! Keyboard-to-teleop input mapping.
//!
//! Maps keyboard keys to [`TeleopCommander`] channels so that
//! pressing keys drives joints.

use bevy::prelude::*;

use clankers_teleop::TeleopCommander;

/// A single key-to-joint channel binding.
#[derive(Clone, Debug)]
pub struct KeyboardJointBinding {
    /// [`TeleopCommander`] channel name (e.g., `"joint_0"`).
    pub channel: String,
    /// Key that increases the channel value.
    pub key_positive: KeyCode,
    /// Key that decreases the channel value.
    pub key_negative: KeyCode,
}

/// Resource mapping keyboard keys to teleop channels.
#[derive(Resource, Clone, Debug)]
pub struct KeyboardTeleopMap {
    /// Per-joint bindings.
    pub bindings: Vec<KeyboardJointBinding>,
    /// Value added per frame while a key is held.
    pub increment: f32,
}

impl Default for KeyboardTeleopMap {
    fn default() -> Self {
        Self::six_joint_default()
    }
}

impl KeyboardTeleopMap {
    /// Create a mapping for exactly `n` joints (clamped to 0..=6).
    ///
    /// Uses the same key pairs as [`six_joint_default`](Self::six_joint_default):
    /// Q/A, W/S, E/D, R/F, T/G, Y/H -- truncated to the first `n`.
    #[must_use]
    pub fn for_joint_count(n: usize) -> Self {
        let pairs = [
            (KeyCode::KeyQ, KeyCode::KeyA),
            (KeyCode::KeyW, KeyCode::KeyS),
            (KeyCode::KeyE, KeyCode::KeyD),
            (KeyCode::KeyR, KeyCode::KeyF),
            (KeyCode::KeyT, KeyCode::KeyG),
            (KeyCode::KeyY, KeyCode::KeyH),
        ];

        let count = n.min(pairs.len());
        let bindings = pairs[..count]
            .iter()
            .enumerate()
            .map(|(i, &(pos, neg))| KeyboardJointBinding {
                channel: format!("joint_{i}"),
                key_positive: pos,
                key_negative: neg,
            })
            .collect();

        Self {
            bindings,
            increment: 0.05,
        }
    }

    /// Default mapping for up to 6 joints.
    ///
    /// Q/A → `joint_0`, W/S → `joint_1`, E/D → `joint_2`,
    /// R/F → `joint_3`, T/G → `joint_4`, Y/H → `joint_5`.
    #[must_use]
    pub fn six_joint_default() -> Self {
        Self::for_joint_count(6)
    }
}

/// System that reads keyboard input and writes to [`TeleopCommander`].
///
/// Always active — keyboard input is captured regardless of mode.
/// The downstream `apply_teleop_commands` system decides whether to
/// actually apply the buffered values to joints.
#[allow(clippy::needless_pass_by_value)]
pub fn keyboard_teleop_system(
    keys: Res<ButtonInput<KeyCode>>,
    map: Res<KeyboardTeleopMap>,
    mut commander: ResMut<TeleopCommander>,
) {
    for binding in &map.bindings {
        let current = commander.get(&binding.channel);
        let mut delta = 0.0;

        if keys.pressed(binding.key_positive) {
            delta += map.increment;
        }
        if keys.pressed(binding.key_negative) {
            delta -= map.increment;
        }

        if delta.abs() > f32::EPSILON {
            let value = (current + delta).clamp(-1.0, 1.0);
            commander.set(binding.channel.clone(), value);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clankers_core::ClankersSet;

    fn build_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(clankers_teleop::ClankersTeleopPlugin);
        app.init_resource::<KeyboardTeleopMap>();
        app.init_resource::<ButtonInput<KeyCode>>();
        app.add_systems(Update, keyboard_teleop_system.in_set(ClankersSet::Decide));
        app.finish();
        app.cleanup();
        app
    }

    #[test]
    fn default_map_has_six_bindings() {
        let map = KeyboardTeleopMap::six_joint_default();
        assert_eq!(map.bindings.len(), 6);
        assert_eq!(map.bindings[0].channel, "joint_0");
        assert_eq!(map.bindings[5].channel, "joint_5");
    }

    #[test]
    fn keyboard_input_always_captured() {
        let mut app = build_test_app();

        // Pressing Q updates commander regardless of mode.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyQ);
        app.update();

        let commander = app.world().resource::<TeleopCommander>();
        assert!(commander.get("joint_0") > 0.0, "keyboard input should always be captured");
    }

    #[test]
    fn pressing_key_updates_commander() {
        let mut app = build_test_app();

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyQ);
        app.update();

        let commander = app.world().resource::<TeleopCommander>();
        let value = commander.get("joint_0");
        assert!(value > 0.0, "expected positive value, got {value}");
    }

    #[test]
    fn negative_key_decrements() {
        let mut app = build_test_app();

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyA);
        app.update();

        let commander = app.world().resource::<TeleopCommander>();
        let value = commander.get("joint_0");
        assert!(value < 0.0, "expected negative value, got {value}");
    }

    #[test]
    fn for_joint_count_zero() {
        let map = KeyboardTeleopMap::for_joint_count(0);
        assert_eq!(map.bindings.len(), 0);
    }

    #[test]
    fn for_joint_count_three() {
        let map = KeyboardTeleopMap::for_joint_count(3);
        assert_eq!(map.bindings.len(), 3);
        assert_eq!(map.bindings[0].channel, "joint_0");
        assert_eq!(map.bindings[2].channel, "joint_2");
        assert_eq!(map.bindings[0].key_positive, KeyCode::KeyQ);
        assert_eq!(map.bindings[2].key_positive, KeyCode::KeyE);
    }

    #[test]
    fn for_joint_count_clamps_at_six() {
        let map = KeyboardTeleopMap::for_joint_count(10);
        assert_eq!(map.bindings.len(), 6);
    }

    #[test]
    fn six_joint_default_equals_for_joint_count_six() {
        let a = KeyboardTeleopMap::six_joint_default();
        let b = KeyboardTeleopMap::for_joint_count(6);
        assert_eq!(a.bindings.len(), b.bindings.len());
        for (ba, bb) in a.bindings.iter().zip(b.bindings.iter()) {
            assert_eq!(ba.channel, bb.channel);
            assert_eq!(ba.key_positive, bb.key_positive);
            assert_eq!(ba.key_negative, bb.key_negative);
        }
    }

    #[test]
    fn value_clamps_to_range() {
        let mut app = build_test_app();

        // Set near the limit.
        app.world_mut()
            .resource_mut::<TeleopCommander>()
            .set("joint_0", 0.99);

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyQ);
        app.update();

        let commander = app.world().resource::<TeleopCommander>();
        let value = commander.get("joint_0");
        assert!((value - 1.0).abs() < f32::EPSILON, "expected clamped to 1.0, got {value}");
    }
}
