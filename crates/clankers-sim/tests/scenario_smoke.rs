//! W8 PR1 scenario smoke matrix.
//!
//! Iterates the four arm-family entries of
//! [`clankers_sim::scenarios::REGISTRY`] and asserts each one builds
//! into a fresh Bevy `App`, advances 10 frames without panicking, and
//! records at least 10 steps in
//! [`clankers_sim::EpisodeStats`](clankers_sim::EpisodeStats).
//!
//! Per the WS8-plan § 6 test contract:
//! - `pub const REGISTRY: &[(&str, fn(&mut App, ScenarioConfig))]`
//! - 10-frame advance per scenario.
//! - `EpisodeStats::total_steps >= 10`.
//!
//! Cartpole is intentionally skipped here — loop 8 will add it to the
//! matrix once the cartpole bins also use the const registry.

use bevy::prelude::App;
use clankers_env::episode::{Episode, EpisodeConfig};
use clankers_sim::{ClankersSimPlugin, EpisodeStats, REGISTRY, ScenarioConfig};

/// Names exercised by [`each_arm_scenario_builds`]. Sourced from the
/// PLAN.md "Loop 7" scope step 4.
const ARM_SCENARIO_NAMES: &[&str] = &["arm_bench", "arm_ik", "arm_pick", "arm_two_link"];

#[test]
fn each_arm_scenario_builds() {
    for name in ARM_SCENARIO_NAMES {
        let (_, builder) = REGISTRY
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("scenario {name} missing from REGISTRY"));

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
