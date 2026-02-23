//! Standard termination condition implementations for common robotics tasks.

use crate::traits::TerminationCondition;
use bevy::prelude::*;

// ---------------------------------------------------------------------------
// SuccessTermination
// ---------------------------------------------------------------------------

/// Terminates when two entities are within a distance threshold.
///
/// Useful for goal-reaching tasks where getting close enough counts as success.
/// Reads entity positions from Bevy [`Transform`] components.
///
/// If either entity is missing a `Transform`, the episode does **not** terminate.
pub struct SuccessTermination {
    entity_a: Entity,
    entity_b: Entity,
    threshold: f32,
}

impl SuccessTermination {
    /// Create a success termination that fires when the two entities are within `threshold`.
    #[must_use]
    pub const fn new(entity_a: Entity, entity_b: Entity, threshold: f32) -> Self {
        Self {
            entity_a,
            entity_b,
            threshold,
        }
    }
}

impl TerminationCondition for SuccessTermination {
    fn is_terminated(&self, world: &World) -> bool {
        let pos_a = world.get::<Transform>(self.entity_a).map(|t| t.translation);
        let pos_b = world.get::<Transform>(self.entity_b).map(|t| t.translation);

        match (pos_a, pos_b) {
            (Some(a), Some(b)) => a.distance(b) < self.threshold,
            _ => false,
        }
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "SuccessTermination"
    }
}

// ---------------------------------------------------------------------------
// TimeoutTermination
// ---------------------------------------------------------------------------

/// Terminates when the step count stored in a resource exceeds a maximum.
///
/// Reads the current step from [`StepCounter`]. Insert this resource and
/// increment it each step for the condition to work.
#[derive(Resource, Clone, Debug, Default)]
pub struct StepCounter(pub u32);

pub struct TimeoutTermination {
    max_steps: u32,
}

impl TimeoutTermination {
    /// Create a timeout that fires after `max_steps`.
    #[must_use]
    pub const fn new(max_steps: u32) -> Self {
        Self { max_steps }
    }
}

impl TerminationCondition for TimeoutTermination {
    fn is_terminated(&self, world: &World) -> bool {
        world
            .get_resource::<StepCounter>()
            .is_some_and(|counter| counter.0 >= self.max_steps)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "TimeoutTermination"
    }
}

// ---------------------------------------------------------------------------
// FailureTermination
// ---------------------------------------------------------------------------

/// Terminates when an entity falls below a height threshold.
///
/// Useful for detecting when a robot has fallen over or dropped below
/// a workspace boundary.
pub struct FailureTermination {
    entity: Entity,
    min_height: f32,
}

impl FailureTermination {
    /// Create a failure termination that fires when `entity`'s Y translation
    /// drops below `min_height`.
    #[must_use]
    pub const fn new(entity: Entity, min_height: f32) -> Self {
        Self { entity, min_height }
    }
}

impl TerminationCondition for FailureTermination {
    fn is_terminated(&self, world: &World) -> bool {
        world
            .get::<Transform>(self.entity)
            .is_some_and(|t| t.translation.y < self.min_height)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "FailureTermination"
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

    // -- SuccessTermination --

    #[test]
    fn success_termination_fires_when_close() {
        let (world, a, b) = world_with_transforms([0.0, 0.0, 0.0], [0.01, 0.0, 0.0]);
        let term = SuccessTermination::new(a, b, 0.1);
        assert!(term.is_terminated(&world));
    }

    #[test]
    fn success_termination_does_not_fire_when_far() {
        let (world, a, b) = world_with_transforms([0.0, 0.0, 0.0], [5.0, 0.0, 0.0]);
        let term = SuccessTermination::new(a, b, 0.1);
        assert!(!term.is_terminated(&world));
    }

    #[test]
    fn success_termination_missing_entity() {
        let mut world = World::new();
        let a = world.spawn_empty().id();
        let b = world.spawn_empty().id();
        let term = SuccessTermination::new(a, b, 0.1);
        assert!(!term.is_terminated(&world));
    }

    #[test]
    fn success_termination_name() {
        let mut world = World::new();
        let a = world.spawn_empty().id();
        let b = world.spawn_empty().id();
        let term = SuccessTermination::new(a, b, 0.5);
        assert_eq!(term.name(), "SuccessTermination");
    }

    // -- TimeoutTermination --

    #[test]
    fn timeout_termination_fires_at_limit() {
        let mut world = World::new();
        world.insert_resource(StepCounter(100));
        let term = TimeoutTermination::new(100);
        assert!(term.is_terminated(&world));
    }

    #[test]
    fn timeout_termination_does_not_fire_before_limit() {
        let mut world = World::new();
        world.insert_resource(StepCounter(50));
        let term = TimeoutTermination::new(100);
        assert!(!term.is_terminated(&world));
    }

    #[test]
    fn timeout_termination_no_resource() {
        let world = World::new();
        let term = TimeoutTermination::new(100);
        assert!(!term.is_terminated(&world));
    }

    #[test]
    fn timeout_termination_name() {
        let term = TimeoutTermination::new(100);
        assert_eq!(term.name(), "TimeoutTermination");
    }

    // -- FailureTermination --

    #[test]
    fn failure_termination_fires_below_height() {
        let mut world = World::new();
        let e = world
            .spawn(Transform::from_translation(Vec3::new(0.0, -1.0, 0.0)))
            .id();
        let term = FailureTermination::new(e, 0.0);
        assert!(term.is_terminated(&world));
    }

    #[test]
    fn failure_termination_does_not_fire_above_height() {
        let mut world = World::new();
        let e = world
            .spawn(Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)))
            .id();
        let term = FailureTermination::new(e, 0.0);
        assert!(!term.is_terminated(&world));
    }

    #[test]
    fn failure_termination_missing_entity() {
        let mut world = World::new();
        let e = world.spawn_empty().id();
        let term = FailureTermination::new(e, 0.0);
        assert!(!term.is_terminated(&world));
    }

    #[test]
    fn failure_termination_name() {
        let mut world = World::new();
        let e = world.spawn_empty().id();
        let term = FailureTermination::new(e, 0.0);
        assert_eq!(term.name(), "FailureTermination");
    }
}
