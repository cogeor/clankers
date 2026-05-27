//! Scenario smoke matrix.
//!
//! Iterates every entry of [`clankers_sim::scenarios::REGISTRY`] and
//! asserts each one builds into a fresh Bevy `App`, advances 12 frames
//! without panicking, and records at least 10 steps in
//! [`clankers_sim::EpisodeStats`](clankers_sim::EpisodeStats).
//!
//! Per the WS8-plan § 6 test contract:
//! - `pub const REGISTRY: &[(&str, fn(&mut App, ScenarioConfig))]`
//! - 12-frame advance per scenario.
//! - `EpisodeStats::total_steps >= 10`.
//!
//! Loop 7 (W8 PR1) shipped this for the arm family; loop 8 (W8 PR2)
//! extends to iterate the full REGISTRY (8 entries after this loop).
//!
//! # Note on quadruped (not present yet)
//!
//! Loop 8 does NOT add a `quadruped_trot` scenario; the quadruped
//! scenario lift is deferred to a follow-up (see IMPLEMENTATION.md
//! "Scope deferred"). When it lands, this test picks it up
//! automatically.

use bevy::prelude::App;
use clankers_env::episode::{Episode, EpisodeConfig};
use clankers_sim::{ClankersSimPlugin, EpisodeStats, REGISTRY, ScenarioConfig};

#[test]
fn each_first_class_scenario_builds() {
    assert!(!REGISTRY.is_empty(), "REGISTRY must not be empty");

    for (name, builder) in REGISTRY {
        let mut app = App::new();
        app.add_plugins(ClankersSimPlugin);

        let cfg = ScenarioConfig {
            max_steps: 10,
            ..ScenarioConfig::default()
        };
        builder(&mut app, cfg);

        // Pin EpisodeConfig regardless of any per-scenario clamps so
        // the 10-step episode terminates exactly when expected.
        app.world_mut()
            .resource_mut::<EpisodeConfig>()
            .max_episode_steps = 10;

        app.finish();
        app.cleanup();

        app.world_mut().resource_mut::<Episode>().reset(None);

        // Advance enough frames to drive the episode to termination
        // and let `episode_stats_system` (in ClankersSet::Communicate)
        // record the completion. Extra frames are no-ops thanks to the
        // edge-detection logic in stats.rs.
        for _ in 0..12 {
            app.update();
        }

        let stats = app.world().resource::<EpisodeStats>();
        assert!(
            stats.episodes_completed >= 1,
            "scenario {name}: expected >= 1 episode completed, got {}",
            stats.episodes_completed,
        );
        assert!(
            stats.total_steps >= 10,
            "scenario {name}: expected >= 10 total_steps, got {}",
            stats.total_steps,
        );
    }
}
