//! Top-level Bevy plugin integrating all core Clankers modules.
//!
//! [`ClankersSimPlugin`] is a convenience meta-plugin that adds the core,
//! actuator, and environment plugins in one call, plus episode statistics
//! tracking.
//!
//! # Example
//!
//! ```no_run
//! use bevy::prelude::*;
//! use clankers_sim::ClankersSimPlugin;
//!
//! App::new()
//!     .add_plugins(ClankersSimPlugin)
//!     .run();
//! ```

pub mod builder;
pub mod stats;

#[cfg(test)]
mod headless;
#[cfg(test)]
mod integration;

use bevy::prelude::*;
use clankers_core::ClankersSet;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use builder::{SceneBuilder, SpawnedScene};
pub use stats::EpisodeStats;

// ---------------------------------------------------------------------------
// ClankersSimPlugin
// ---------------------------------------------------------------------------

/// Meta-plugin that adds the full Clankers simulation stack.
///
/// Includes:
/// - [`ClankersCorePlugin`](clankers_core::ClankersCorePlugin) — system ordering and `SimTime`
/// - [`ClankersActuatorPlugin`](clankers_actuator::ClankersActuatorPlugin) — motor/joint dynamics
/// - [`ClankersEnvPlugin`](clankers_env::ClankersEnvPlugin) — observation, episode lifecycle
/// - [`EpisodeStats`] resource and tracking system
///
/// Does NOT include policy, domain-rand, URDF, or render plugins — add those
/// manually based on your application needs.
pub struct ClankersSimPlugin;

impl Plugin for ClankersSimPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(clankers_core::ClankersCorePlugin)
            .add_plugins(clankers_actuator::ClankersActuatorPlugin)
            .add_plugins(clankers_env::ClankersEnvPlugin)
            .init_resource::<EpisodeStats>()
            .add_systems(
                Update,
                stats::episode_stats_system.in_set(ClankersSet::Communicate),
            );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_builds_without_panic() {
        let mut app = App::new();
        app.add_plugins(ClankersSimPlugin);
        app.finish();
        app.cleanup();
        app.update();

        assert!(app.world().get_resource::<EpisodeStats>().is_some());
        assert!(
            app.world()
                .get_resource::<clankers_core::time::SimTime>()
                .is_some()
        );
        assert!(
            app.world()
                .get_resource::<clankers_env::episode::Episode>()
                .is_some()
        );
    }

    #[test]
    fn plugin_can_run_episode() {
        let mut app = App::new();
        app.add_plugins(ClankersSimPlugin);
        app.finish();
        app.cleanup();

        // Configure short episode
        app.world_mut()
            .resource_mut::<clankers_env::episode::EpisodeConfig>()
            .max_episode_steps = 5;

        app.world_mut()
            .resource_mut::<clankers_env::episode::Episode>()
            .reset(None);

        for _ in 0..5 {
            app.update();
        }

        let stats = app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 1);
        assert_eq!(stats.total_steps, 5);
    }

    #[test]
    fn multiple_episodes_tracked() {
        let mut app = App::new();
        app.add_plugins(ClankersSimPlugin);
        app.finish();
        app.cleanup();

        app.world_mut()
            .resource_mut::<clankers_env::episode::EpisodeConfig>()
            .max_episode_steps = 3;

        for _ in 0..3 {
            app.world_mut()
                .resource_mut::<clankers_env::episode::Episode>()
                .reset(None);
            for _ in 0..3 {
                app.update();
            }
        }

        let stats = app.world().resource::<EpisodeStats>();
        assert_eq!(stats.episodes_completed, 3);
        assert_eq!(stats.total_steps, 9);
    }
}
