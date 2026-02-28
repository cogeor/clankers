//! [`RecorderPlugin`] — registers all recording systems with the Bevy app.

use bevy::prelude::*;

use crate::recorder::{
    PendingAction, PendingReward, RecordingConfig, record_action_system, record_joint_states_system,
    record_reward_system, setup_channels, setup_recorder,
};

// ---------------------------------------------------------------------------
// RecorderPlugin
// ---------------------------------------------------------------------------

/// Bevy plugin that sets up MCAP episode recording.
///
/// # Usage
///
/// ```no_run
/// use bevy::prelude::*;
/// use clankers_record::plugin::RecorderPlugin;
/// use clankers_record::recorder::RecordingConfig;
///
/// let mut app = App::new();
/// app.insert_resource(RecordingConfig::default());
/// app.add_plugins(RecorderPlugin);
/// ```
///
/// The plugin inserts default [`PendingAction`] and [`PendingReward`]
/// resources.  Update them from your own systems before `PostUpdate` runs so
/// the recording systems pick up the latest values.
pub struct RecorderPlugin;

impl Plugin for RecorderPlugin {
    fn build(&self, app: &mut App) {
        // Ensure configuration resource is present.
        app.init_resource::<RecordingConfig>();
        app.init_resource::<PendingReward>();
        app.init_resource::<PendingAction>();

        // setup_recorder takes &mut World (exclusive system) — add to Startup.
        app.add_systems(Startup, setup_recorder);

        // Register MCAP channels on the first PostUpdate frame after startup.
        // setup_channels uses NonSendMut<Recorder> so it implicitly runs on
        // the main thread. The recording systems are after it.
        app.add_systems(
            PostUpdate,
            (
                setup_channels,
                record_joint_states_system,
                record_action_system,
                record_reward_system,
            )
                .chain(),
        );

        // Optional camera recording.
        #[cfg(feature = "camera")]
        {
            use crate::recorder::camera::{CameraChannelIds, record_image_system};
            app.init_resource::<CameraChannelIds>();
            app.add_systems(
                PostUpdate,
                record_image_system.after(setup_channels),
            );
        }
    }
}
