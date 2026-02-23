//! Policy implementations and inference runner for Clankers.
//!
//! Provides basic policies (zero, constant, random, scripted) and a
//! [`PolicyRunner`](runner::PolicyRunner) resource that drives the decide phase of the simulation loop.
//!
//! # Example
//!
//! ```no_run
//! use bevy::prelude::*;
//! use clankers_policy::prelude::*;
//!
//! let runner = PolicyRunner::new(Box::new(ZeroPolicy::new(4)), 4);
//! App::new()
//!     .add_plugins(clankers_core::ClankersCorePlugin)
//!     .add_plugins(clankers_env::ClankersEnvPlugin)
//!     .add_plugins(ClankersPolicyPlugin)
//!     .insert_resource(runner)
//!     .run();
//! ```

pub mod policies;
pub mod runner;

use bevy::prelude::*;
use clankers_core::ClankersSet;

// ---------------------------------------------------------------------------
// ClankersPolicyPlugin
// ---------------------------------------------------------------------------

/// Bevy plugin that adds the policy decide system.
///
/// Requires a [`PolicyRunner`](runner::PolicyRunner) resource to be inserted
/// by the user (since the policy choice is application-specific).
///
/// Adds [`policy_decide_system`](runner::policy_decide_system) to
/// [`ClankersSet::Decide`].
pub struct ClankersPolicyPlugin;

impl Plugin for ClankersPolicyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            runner::policy_decide_system.in_set(ClankersSet::Decide),
        );
    }
}

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::{
        ClankersPolicyPlugin,
        policies::{ConstantPolicy, RandomPolicy, ScriptedPolicy, ZeroPolicy},
        runner::{PolicyRunner, policy_decide_system},
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::PolicyRunner;

    #[test]
    fn plugin_builds_with_runner() {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(clankers_env::ClankersEnvPlugin);
        app.add_plugins(ClankersPolicyPlugin);
        app.insert_resource(PolicyRunner::new(Box::new(policies::ZeroPolicy::new(2)), 2));
        app.finish();
        app.cleanup();
        app.update();

        let runner = app.world().resource::<PolicyRunner>();
        assert_eq!(runner.policy_name(), "ZeroPolicy");
    }
}
