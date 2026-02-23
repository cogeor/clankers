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
// RewardFunction
// ---------------------------------------------------------------------------

/// Computes a scalar reward from the current world state.
pub trait RewardFunction: Send + Sync + 'static {
    /// Compute the reward value.
    fn compute(&self, world: &World) -> f32;

    /// Human-readable name for this reward function.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// TerminationCondition
// ---------------------------------------------------------------------------

/// Determines whether an episode should terminate.
pub trait TerminationCondition: Send + Sync + 'static {
    /// Returns `true` if the episode should end.
    fn is_terminated(&self, world: &World) -> bool;

    /// Human-readable name for this termination condition.
    fn name(&self) -> &str;
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
// CompositeReward
// ---------------------------------------------------------------------------

/// A weighted combination of multiple reward functions.
///
/// The total reward is the sum of each component reward multiplied by its
/// weight. Use [`breakdown`](Self::breakdown) to inspect individual
/// contributions.
pub struct CompositeReward {
    rewards: Vec<(Box<dyn RewardFunction>, f32)>,
}

impl CompositeReward {
    /// Create an empty composite reward.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rewards: Vec::new(),
        }
    }

    /// Add a reward function with the given weight. Returns `self` for chaining.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, reward: Box<dyn RewardFunction>, weight: f32) -> Self {
        self.rewards.push((reward, weight));
        self
    }

    /// Compute each component reward and return `(name, weighted_value)` pairs.
    pub fn breakdown(&self, world: &World) -> Vec<(&str, f32)> {
        self.rewards
            .iter()
            .map(|(reward, weight)| (reward.name(), reward.compute(world) * weight))
            .collect()
    }
}

impl Default for CompositeReward {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardFunction for CompositeReward {
    fn compute(&self, world: &World) -> f32 {
        self.rewards
            .iter()
            .map(|(reward, weight)| reward.compute(world) * weight)
            .sum()
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "CompositeReward"
    }
}

// ---------------------------------------------------------------------------
// CompositeTermination
// ---------------------------------------------------------------------------

/// OR-composition of multiple termination conditions.
///
/// Returns `true` if **any** contained condition is satisfied.
pub struct CompositeTermination {
    conditions: Vec<Box<dyn TerminationCondition>>,
}

impl CompositeTermination {
    /// Create an empty composite termination.
    #[must_use]
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Add a termination condition. Returns `self` for chaining.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, condition: Box<dyn TerminationCondition>) -> Self {
        self.conditions.push(condition);
        self
    }
}

impl Default for CompositeTermination {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminationCondition for CompositeTermination {
    fn is_terminated(&self, world: &World) -> bool {
        self.conditions
            .iter()
            .any(|condition| condition.is_terminated(world))
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "CompositeTermination"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Mock reward functions --

    struct ConstantReward {
        value: f32,
        label: &'static str,
    }

    impl ConstantReward {
        fn new(value: f32, label: &'static str) -> Self {
            Self { value, label }
        }
    }

    impl RewardFunction for ConstantReward {
        fn compute(&self, _world: &World) -> f32 {
            self.value
        }

        fn name(&self) -> &str {
            self.label
        }
    }

    // -- Mock termination conditions --

    struct AlwaysTerminate;

    impl TerminationCondition for AlwaysTerminate {
        fn is_terminated(&self, _world: &World) -> bool {
            true
        }

        #[allow(clippy::unnecessary_literal_bound)]
        fn name(&self) -> &str {
            "AlwaysTerminate"
        }
    }

    struct NeverTerminate;

    impl TerminationCondition for NeverTerminate {
        fn is_terminated(&self, _world: &World) -> bool {
            false
        }

        #[allow(clippy::unnecessary_literal_bound)]
        fn name(&self) -> &str {
            "NeverTerminate"
        }
    }

    /// Helper to create a minimal Bevy World for testing.
    fn test_world() -> World {
        World::new()
    }

    // ---- CompositeReward tests ----

    #[test]
    fn composite_reward_empty_returns_zero() {
        let reward = CompositeReward::new();
        let world = test_world();
        assert!((reward.compute(&world) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn composite_reward_single() {
        let reward = CompositeReward::new().add(Box::new(ConstantReward::new(5.0, "five")), 1.0);
        let world = test_world();
        assert!((reward.compute(&world) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn composite_reward_weighted_sum() {
        let reward = CompositeReward::new()
            .add(Box::new(ConstantReward::new(10.0, "distance")), 0.5)
            .add(Box::new(ConstantReward::new(2.0, "penalty")), -1.0);
        let world = test_world();
        // 10.0 * 0.5 + 2.0 * (-1.0) = 5.0 - 2.0 = 3.0
        assert!((reward.compute(&world) - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn composite_reward_breakdown() {
        let reward = CompositeReward::new()
            .add(Box::new(ConstantReward::new(10.0, "distance")), 0.5)
            .add(Box::new(ConstantReward::new(2.0, "penalty")), -1.0);
        let world = test_world();
        let breakdown = reward.breakdown(&world);
        assert_eq!(breakdown.len(), 2);
        assert_eq!(breakdown[0].0, "distance");
        assert!((breakdown[0].1 - 5.0).abs() < f32::EPSILON);
        assert_eq!(breakdown[1].0, "penalty");
        assert!((breakdown[1].1 - (-2.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn composite_reward_default() {
        let reward = CompositeReward::default();
        let world = test_world();
        assert!((reward.compute(&world) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn composite_reward_name() {
        let reward = CompositeReward::new();
        assert_eq!(reward.name(), "CompositeReward");
    }

    // ---- CompositeTermination tests ----

    #[test]
    fn composite_termination_empty_returns_false() {
        let term = CompositeTermination::new();
        let world = test_world();
        assert!(!term.is_terminated(&world));
    }

    #[test]
    fn composite_termination_single_true() {
        let term = CompositeTermination::new().add(Box::new(AlwaysTerminate));
        let world = test_world();
        assert!(term.is_terminated(&world));
    }

    #[test]
    fn composite_termination_single_false() {
        let term = CompositeTermination::new().add(Box::new(NeverTerminate));
        let world = test_world();
        assert!(!term.is_terminated(&world));
    }

    #[test]
    fn composite_termination_or_logic_any_true() {
        let term = CompositeTermination::new()
            .add(Box::new(NeverTerminate))
            .add(Box::new(AlwaysTerminate))
            .add(Box::new(NeverTerminate));
        let world = test_world();
        assert!(term.is_terminated(&world));
    }

    #[test]
    fn composite_termination_or_logic_all_false() {
        let term = CompositeTermination::new()
            .add(Box::new(NeverTerminate))
            .add(Box::new(NeverTerminate));
        let world = test_world();
        assert!(!term.is_terminated(&world));
    }

    #[test]
    fn composite_termination_default() {
        let term = CompositeTermination::default();
        let world = test_world();
        assert!(!term.is_terminated(&world));
    }

    #[test]
    fn composite_termination_name() {
        let term = CompositeTermination::new();
        assert_eq!(term.name(), "CompositeTermination");
    }

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
