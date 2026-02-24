//! The main physics plugin that delegates to a concrete backend.

use bevy::app::{App, Plugin};

use crate::backend::PhysicsBackend;

/// Bevy plugin that wires a [`PhysicsBackend`] into the app.
///
/// # Usage
///
/// ```ignore
/// app.add_plugins(ClankersPhysicsPlugin::new(RapierBackend::default()));
/// ```
///
/// The plugin delegates all setup to the backend's [`build`](PhysicsBackend::build)
/// method, which inserts engine-specific resources and registers systems in
/// [`ClankersSet::Simulate`](clankers_core::ClankersSet::Simulate).
pub struct ClankersPhysicsPlugin {
    backend: Box<dyn PhysicsBackend>,
}

impl ClankersPhysicsPlugin {
    /// Create a new physics plugin with the given backend.
    pub fn new(backend: impl PhysicsBackend + 'static) -> Self {
        Self {
            backend: Box::new(backend),
        }
    }

    /// The name of the active physics backend.
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
}

impl Plugin for ClankersPhysicsPlugin {
    fn build(&self, app: &mut App) {
        self.backend.build(app);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct TestBackend {
        name: &'static str,
    }

    impl PhysicsBackend for TestBackend {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            self.name
        }
    }

    #[test]
    fn plugin_delegates_name() {
        let plugin = ClankersPhysicsPlugin::new(TestBackend { name: "test" });
        assert_eq!(plugin.backend_name(), "test");
    }

    #[test]
    fn plugin_builds_without_panic() {
        let plugin = ClankersPhysicsPlugin::new(TestBackend { name: "test" });
        let mut app = App::new();
        plugin.build(&mut app);
    }
}
