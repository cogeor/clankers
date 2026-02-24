// clankers-physics: Engine-agnostic physics abstraction layer for Clankers.
//
// Provides a `PhysicsBackend` trait so the concrete engine (rapier3d, XPBD,
// etc.) can be swapped without changing the rest of the system. Marker
// components tag entities that participate in physics. The plugin delegates
// all setup to the chosen backend.

pub mod backend;
pub mod components;
pub mod plugin;
pub mod systems;

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::{
        backend::PhysicsBackend,
        components::{GroundPlane, PhysicsBody, PhysicsJoint},
        plugin::ClankersPhysicsPlugin,
    };
}

// Re-export the plugin at crate root for convenience.
pub use plugin::ClankersPhysicsPlugin;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the prelude re-exports compile.
    #[test]
    fn prelude_exports() {
        use prelude::*;

        // PhysicsBackend trait is usable
        fn _accepts_backend(_: &dyn PhysicsBackend) {}

        // Components construct
        let _body = PhysicsBody::Fixed;
        let _gp = GroundPlane::default();
    }
}
