use crate::types::{Action, ActionSpace, Observation, ObservationSpace, RobotId};
use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

/// Core trait for a complete simulation environment.
///
/// Provides the full observation/action loop: observe the world, act on it,
/// and reset to an initial state.
pub trait Simulation: Send + Sync + 'static {
    /// Describes the shape and bounds of observations this simulation produces.
    fn observation_space(&self) -> ObservationSpace;

    /// Describes the shape and bounds of actions this simulation accepts.
    fn action_space(&self) -> ActionSpace;

    /// Read the current state of the world and produce an observation.
    fn observe(&self, world: &World) -> Observation;

    /// Apply an action to the world.
    fn act(&self, world: &mut World, action: &Action);

    /// Reset the simulation to an initial state, optionally with a seed.
    fn reset(&self, world: &mut World, seed: Option<u64>);

    /// Human-readable name for this simulation.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

// ---------------------------------------------------------------------------
// Sensor
// ---------------------------------------------------------------------------

/// A sensor that reads data from the world.
pub trait Sensor: Send + Sync + 'static {
    /// The type of data this sensor produces.
    type Output;

    /// Read sensor data from the world.
    fn read(&self, world: &mut World) -> Self::Output;

    /// Human-readable name for this sensor.
    fn name(&self) -> &str;

    /// Optional update rate in Hz. `None` means every frame.
    fn rate_hz(&self) -> Option<f64> {
        None
    }
}

/// A sensor whose output is an [`Observation`] vector.
pub trait ObservationSensor: Sensor<Output = Observation> {
    /// Dimensionality of the observation vector this sensor produces.
    fn observation_dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// A policy that maps observations to actions.
pub trait Policy: Send + Sync + 'static {
    /// Given an observation, produce an action.
    fn get_action(&self, obs: &Observation) -> Action;

    /// Human-readable name for this policy.
    fn name(&self) -> &str;

    /// Whether this policy is deterministic (no randomness).
    fn is_deterministic(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// ActionApplicator
// ---------------------------------------------------------------------------

/// Applies an action to the world.
pub trait ActionApplicator: Send + Sync + 'static {
    /// Apply the given action to the world.
    fn apply(&self, world: &mut World, action: &Action);

    /// Human-readable name for this applicator.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// MultiRobotActionApplicator
// ---------------------------------------------------------------------------

/// Applies actions to multiple robots in a single world.
///
/// The `actions` slice is indexed by robot order (matching allocation order in
/// [`RobotGroup`](crate::types::RobotGroup)). Each element corresponds to one
/// robot's action.
pub trait MultiRobotActionApplicator: Send + Sync + 'static {
    /// Apply one action per robot. `actions` is indexed by robot order.
    fn apply_multi(&self, world: &mut World, robot_ids: &[RobotId], actions: &[Action]);

    /// Human-readable name for this applicator.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MultiRobotActionApplicator ----

    struct MockMultiApplicator;

    impl MultiRobotActionApplicator for MockMultiApplicator {
        fn apply_multi(&self, _world: &mut World, robot_ids: &[RobotId], actions: &[Action]) {
            assert_eq!(robot_ids.len(), actions.len());
        }

        #[allow(clippy::unnecessary_literal_bound)]
        fn name(&self) -> &str {
            "MockMultiApplicator"
        }
    }

    fn test_world() -> World {
        World::new()
    }

    #[test]
    fn multi_robot_applicator_dispatches() {
        let applicator = MockMultiApplicator;
        let mut world = test_world();
        let ids = [RobotId(0), RobotId(1)];
        let actions = [Action::zeros(2), Action::zeros(2)];
        applicator.apply_multi(&mut world, &ids, &actions);
        assert_eq!(applicator.name(), "MockMultiApplicator");
    }

    #[test]
    fn multi_robot_applicator_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockMultiApplicator>();
    }
}
