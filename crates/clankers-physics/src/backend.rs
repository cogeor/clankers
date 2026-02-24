//! Engine-agnostic physics backend trait.
//!
//! Any physics engine (Rapier, XPBD, custom) implements [`PhysicsBackend`]
//! and passes it to [`ClankersPhysicsPlugin::new`](super::ClankersPhysicsPlugin::new).

use bevy::app::App;

/// Trait that concrete physics engines must implement.
///
/// The backend is responsible for:
/// - Inserting engine-specific resources (rigid body sets, pipelines, etc.)
/// - Registering systems in [`ClankersSet::Simulate`](clankers_core::ClankersSet::Simulate)
/// - Reading [`JointTorque`](clankers_actuator::components::JointTorque) and writing
///   [`JointState`](clankers_actuator::components::JointState)
pub trait PhysicsBackend: Send + Sync + 'static {
    /// Called once during plugin build to insert engine-specific resources
    /// and register systems.
    fn build(&self, app: &mut App);

    /// Human-readable engine name (e.g., "rapier3d").
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the trait is object-safe (can be used as `dyn PhysicsBackend`).
    #[test]
    fn trait_is_object_safe() {
        fn _accepts_boxed(_: Box<dyn PhysicsBackend>) {}
    }

    /// Verify the trait bound includes Send + Sync.
    #[test]
    fn trait_is_send_sync() {
        fn _assert_send_sync<T: Send + Sync>() {}
        // A boxed trait object should be Send + Sync because the trait requires it.
        _assert_send_sync::<Box<dyn PhysicsBackend>>();
    }

    /// Minimal backend for testing.
    struct DummyBackend;

    impl PhysicsBackend for DummyBackend {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "dummy"
        }
    }

    #[test]
    fn dummy_backend_name() {
        let b = DummyBackend;
        assert_eq!(b.name(), "dummy");
    }

    #[test]
    fn dummy_backend_can_be_boxed() {
        let b: Box<dyn PhysicsBackend> = Box::new(DummyBackend);
        assert_eq!(b.name(), "dummy");
    }
}
