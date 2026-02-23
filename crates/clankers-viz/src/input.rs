//! Keyboard-to-teleop input mapping.
//!
//! Maps keyboard keys to [`TeleopCommander`] channels so that
//! pressing keys drives joints during teleop mode.

use bevy::prelude::*;

use clankers_teleop::TeleopCommander;

use crate::mode::VizMode;

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
    /// Default mapping for up to 6 joints.
    ///
    /// Q/A → `joint_0`, W/S → `joint_1`, E/D → `joint_2`,
    /// R/F → `joint_3`, T/G → `joint_4`, Y/H → `joint_5`.
    #[must_use]
    pub fn six_joint_default() -> Self {
        let pairs = [
            (KeyCode::KeyQ, KeyCode::KeyA),
            (KeyCode::KeyW, KeyCode::KeyS),
            (KeyCode::KeyE, KeyCode::KeyD),
            (KeyCode::KeyR, KeyCode::KeyF),
            (KeyCode::KeyT, KeyCode::KeyG),
            (KeyCode::KeyY, KeyCode::KeyH),
        ];

        let bindings = pairs
            .into_iter()
            .enumerate()
            .map(|(i, (pos, neg))| KeyboardJointBinding {
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
}

/// System that reads keyboard input and writes to [`TeleopCommander`].
///
/// Only active when [`VizMode::Teleop`] is selected.
#[allow(clippy::needless_pass_by_value)]
pub fn keyboard_teleop_system(
    keys: Res<ButtonInput<KeyCode>>,
    map: Res<KeyboardTeleopMap>,
    mut commander: ResMut<TeleopCommander>,
    mode: Res<VizMode>,
) {
    if *mode != VizMode::Teleop {
        return;
    }

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
