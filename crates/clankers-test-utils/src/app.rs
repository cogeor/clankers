//! Bevy test app builders with various plugin combinations.

use bevy::prelude::*;

/// Create a minimal test app with only the core plugin.
///
/// Provides `ClankersSet` system ordering and core resources
/// but no actuator, environment, or policy systems.
pub fn minimal_test_app() -> App {
    let mut app = App::new();
    app.add_plugins(clankers_core::ClankersCorePlugin);
    app.finish();
    app.cleanup();
    app
}

/// Create a full-stack test app with core, actuator, and environment plugins.
///
/// Provides all standard simulation systems: observe, act, evaluate.
/// Does NOT include policy or domain-rand plugins — add those manually
/// if needed for your test.
pub fn full_test_app() -> App {
    let mut app = App::new();
    app.add_plugins(clankers_core::ClankersCorePlugin);
    app.add_plugins(clankers_actuator::ClankersActuatorPlugin);
    app.add_plugins(clankers_env::ClankersEnvPlugin);
    app.finish();
    app.cleanup();
    app
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clankers_env::episode::Episode;

    #[test]
    fn minimal_app_builds() {
        let app = minimal_test_app();
        // Core plugin registers ClankersSet ordering
        assert!(
            app.world()
                .get_resource::<clankers_core::time::SimTime>()
                .is_some()
        );
    }

    #[test]
    fn full_app_builds() {
        let app = full_test_app();
        assert!(app.world().get_resource::<Episode>().is_some());
    }

    #[test]
    fn full_app_can_update() {
        let mut app = full_test_app();
        app.update();
        app.update();
    }
}
