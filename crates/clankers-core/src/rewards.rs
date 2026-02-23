//! Standard reward function implementations for common robotics tasks.

use crate::traits::RewardFunction;
use bevy::prelude::*;

// ---------------------------------------------------------------------------
// DistanceReward
// ---------------------------------------------------------------------------

/// Reward based on the L2 distance between two entities.
///
/// Returns `-distance` (negative so maximizing reward minimizes distance).
/// The entity positions are read from Bevy [`Transform`] components.
///
/// If either entity is missing a `Transform`, returns `0.0`.
pub struct DistanceReward {
    entity_a: Entity,
    entity_b: Entity,
}

impl DistanceReward {
    /// Create a distance reward between two entities.
    #[must_use]
    pub const fn new(entity_a: Entity, entity_b: Entity) -> Self {
        Self { entity_a, entity_b }
    }
}

impl RewardFunction for DistanceReward {
    fn compute(&self, world: &World) -> f32 {
        let pos_a = world.get::<Transform>(self.entity_a).map(|t| t.translation);
        let pos_b = world.get::<Transform>(self.entity_b).map(|t| t.translation);

        match (pos_a, pos_b) {
            (Some(a), Some(b)) => -a.distance(b),
            _ => 0.0,
        }
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "DistanceReward"
    }
}

// ---------------------------------------------------------------------------
// SparseReward
// ---------------------------------------------------------------------------

/// Binary reward: `1.0` when distance is below a threshold, `0.0` otherwise.
///
/// Useful for goal-reaching tasks where only success matters.
pub struct SparseReward {
    entity_a: Entity,
    entity_b: Entity,
    threshold: f32,
}

impl SparseReward {
    /// Create a sparse reward that fires when the two entities are within `threshold`.
    #[must_use]
    pub const fn new(entity_a: Entity, entity_b: Entity, threshold: f32) -> Self {
        Self {
            entity_a,
            entity_b,
            threshold,
        }
    }
}

impl RewardFunction for SparseReward {
    fn compute(&self, world: &World) -> f32 {
        let pos_a = world.get::<Transform>(self.entity_a).map(|t| t.translation);
        let pos_b = world.get::<Transform>(self.entity_b).map(|t| t.translation);

        match (pos_a, pos_b) {
            (Some(a), Some(b)) if a.distance(b) < self.threshold => 1.0,
            _ => 0.0,
        }
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "SparseReward"
    }
}

// ---------------------------------------------------------------------------
// ActionPenaltyReward
// ---------------------------------------------------------------------------

/// Penalty proportional to the L2 norm of the last action.
///
/// Returns `-scale * ||action||^2`, encouraging minimal control effort.
/// The action is stored as a resource and must be set externally each step.
#[derive(Resource, Clone, Debug, Default)]
pub struct LastAction(pub Vec<f32>);

pub struct ActionPenaltyReward {
    scale: f32,
}

impl ActionPenaltyReward {
    /// Create with a penalty scale factor.
    #[must_use]
    pub const fn new(scale: f32) -> Self {
        Self { scale }
    }
}

impl RewardFunction for ActionPenaltyReward {
    fn compute(&self, world: &World) -> f32 {
        let Some(action) = world.get_resource::<LastAction>() else {
            return 0.0;
        };
        let norm_sq: f32 = action.0.iter().map(|v| v * v).sum();
        -self.scale * norm_sq
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "ActionPenaltyReward"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn world_with_transforms(pos_a: [f32; 3], pos_b: [f32; 3]) -> (World, Entity, Entity) {
        let mut world = World::new();
        let a = world
            .spawn(Transform::from_translation(Vec3::from_array(pos_a)))
            .id();
        let b = world
            .spawn(Transform::from_translation(Vec3::from_array(pos_b)))
            .id();
        (world, a, b)
    }

    // -- DistanceReward --

    #[test]
    fn distance_reward_zero_when_same_position() {
        let (world, a, b) = world_with_transforms([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let reward = DistanceReward::new(a, b);
        assert!(reward.compute(&world).abs() < f32::EPSILON);
    }

    #[test]
    fn distance_reward_negative_when_apart() {
        let (world, a, b) = world_with_transforms([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
        let reward = DistanceReward::new(a, b);
        assert!((reward.compute(&world) - (-5.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn distance_reward_missing_entity() {
        let mut world = World::new();
        let a = world.spawn_empty().id();
        let b = world.spawn_empty().id();
        let reward = DistanceReward::new(a, b);
        assert!(reward.compute(&world).abs() < f32::EPSILON);
    }

    #[test]
    fn distance_reward_name() {
        let mut world = World::new();
        let a = world.spawn_empty().id();
        let b = world.spawn_empty().id();
        let reward = DistanceReward::new(a, b);
        assert_eq!(reward.name(), "DistanceReward");
    }

    // -- SparseReward --

    #[test]
    fn sparse_reward_success() {
        let (world, a, b) = world_with_transforms([0.0, 0.0, 0.0], [0.01, 0.0, 0.0]);
        let reward = SparseReward::new(a, b, 0.1);
        assert!((reward.compute(&world) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sparse_reward_failure() {
        let (world, a, b) = world_with_transforms([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let reward = SparseReward::new(a, b, 0.1);
        assert!(reward.compute(&world).abs() < f32::EPSILON);
    }

    #[test]
    fn sparse_reward_name() {
        let mut world = World::new();
        let a = world.spawn_empty().id();
        let b = world.spawn_empty().id();
        let reward = SparseReward::new(a, b, 0.5);
        assert_eq!(reward.name(), "SparseReward");
    }

    // -- ActionPenaltyReward --

    #[test]
    fn action_penalty_no_resource() {
        let world = World::new();
        let reward = ActionPenaltyReward::new(0.01);
        assert!(reward.compute(&world).abs() < f32::EPSILON);
    }

    #[test]
    fn action_penalty_computes_negative_norm() {
        let mut world = World::new();
        world.insert_resource(LastAction(vec![3.0, 4.0])); // norm^2 = 25
        let reward = ActionPenaltyReward::new(0.01);
        assert!((reward.compute(&world) - (-0.25)).abs() < f32::EPSILON);
    }

    #[test]
    fn action_penalty_zero_action() {
        let mut world = World::new();
        world.insert_resource(LastAction(vec![0.0, 0.0]));
        let reward = ActionPenaltyReward::new(1.0);
        assert!(reward.compute(&world).abs() < f32::EPSILON);
    }

    #[test]
    fn action_penalty_name() {
        let reward = ActionPenaltyReward::new(0.01);
        assert_eq!(reward.name(), "ActionPenaltyReward");
    }
}
