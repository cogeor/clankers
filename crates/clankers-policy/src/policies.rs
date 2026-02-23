//! Basic policy implementations.
//!
//! All policies implement [`Policy`] from `clankers-core`.

use std::sync::Mutex;

use clankers_core::traits::Policy;
use clankers_core::types::{Action, ActionSpace, Observation};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// ZeroPolicy
// ---------------------------------------------------------------------------

/// Policy that always returns a zero-valued continuous action.
pub struct ZeroPolicy {
    dim: usize,
}

impl ZeroPolicy {
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Policy for ZeroPolicy {
    fn get_action(&self, _obs: &Observation) -> Action {
        Action::zeros(self.dim)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "ZeroPolicy"
    }

    fn is_deterministic(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// ConstantPolicy
// ---------------------------------------------------------------------------

/// Policy that always returns the same fixed action.
pub struct ConstantPolicy {
    action: Action,
}

impl ConstantPolicy {
    pub const fn new(action: Action) -> Self {
        Self { action }
    }
}

impl Policy for ConstantPolicy {
    fn get_action(&self, _obs: &Observation) -> Action {
        self.action.clone()
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "ConstantPolicy"
    }

    fn is_deterministic(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// RandomPolicy
// ---------------------------------------------------------------------------

/// Policy that samples random actions from the action space.
///
/// Uses a seeded RNG for determinism. Thread-safe via [`Mutex`].
pub struct RandomPolicy {
    space: ActionSpace,
    rng: Mutex<ChaCha8Rng>,
}

impl RandomPolicy {
    /// Create a random policy for the given action space and seed.
    pub fn new(space: ActionSpace, seed: u64) -> Self {
        Self {
            space,
            rng: Mutex::new(ChaCha8Rng::seed_from_u64(seed)),
        }
    }
}

impl Policy for RandomPolicy {
    fn get_action(&self, _obs: &Observation) -> Action {
        let mut rng = self.rng.lock().unwrap();
        self.space.sample(&mut *rng)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "RandomPolicy"
    }

    fn is_deterministic(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// ScriptedPolicy
// ---------------------------------------------------------------------------

/// Policy that replays a fixed sequence of actions, cycling when exhausted.
pub struct ScriptedPolicy {
    actions: Vec<Action>,
    index: Mutex<usize>,
}

impl ScriptedPolicy {
    /// Create a scripted policy from a sequence of actions.
    ///
    /// # Panics
    ///
    /// Panics if `actions` is empty.
    pub fn new(actions: Vec<Action>) -> Self {
        assert!(
            !actions.is_empty(),
            "ScriptedPolicy requires at least one action"
        );
        Self {
            actions,
            index: Mutex::new(0),
        }
    }
}

impl Policy for ScriptedPolicy {
    fn get_action(&self, _obs: &Observation) -> Action {
        let mut idx = self.index.lock().unwrap();
        let action = self.actions[*idx].clone();
        *idx = (*idx + 1) % self.actions.len();
        action
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "ScriptedPolicy"
    }

    fn is_deterministic(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_obs() -> Observation {
        Observation::new(vec![1.0, 2.0, 3.0])
    }

    // -- ZeroPolicy --

    #[test]
    fn zero_policy_returns_zeros() {
        let policy = ZeroPolicy::new(4);
        let action = policy.get_action(&dummy_obs());
        assert_eq!(action.as_slice().len(), 4);
        for &v in action.as_slice() {
            assert!((v).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn zero_policy_name() {
        let policy = ZeroPolicy::new(1);
        assert_eq!(policy.name(), "ZeroPolicy");
    }

    #[test]
    fn zero_policy_deterministic() {
        let policy = ZeroPolicy::new(1);
        assert!(policy.is_deterministic());
    }

    // -- ConstantPolicy --

    #[test]
    fn constant_policy_returns_fixed_action() {
        let action = Action::from(vec![1.0, 2.0, 3.0]);
        let policy = ConstantPolicy::new(action.clone());
        let result = policy.get_action(&dummy_obs());
        assert_eq!(result.as_slice(), action.as_slice());
    }

    #[test]
    fn constant_policy_name() {
        let policy = ConstantPolicy::new(Action::zeros(1));
        assert_eq!(policy.name(), "ConstantPolicy");
    }

    // -- RandomPolicy --

    #[test]
    fn random_policy_samples_from_space() {
        let space = ActionSpace::Box {
            low: vec![-1.0, -1.0],
            high: vec![1.0, 1.0],
        };
        let policy = RandomPolicy::new(space, 42);
        let action = policy.get_action(&dummy_obs());
        let vals = action.as_slice();
        assert_eq!(vals.len(), 2);
        for &v in vals {
            assert!((-1.0..1.0).contains(&v), "got {v}");
        }
    }

    #[test]
    fn random_policy_deterministic_with_same_seed() {
        let space = ActionSpace::Box {
            low: vec![-1.0],
            high: vec![1.0],
        };
        let p1 = RandomPolicy::new(space.clone(), 123);
        let p2 = RandomPolicy::new(space, 123);
        let a1 = p1.get_action(&dummy_obs());
        let a2 = p2.get_action(&dummy_obs());
        assert_eq!(a1.as_slice(), a2.as_slice());
    }

    #[test]
    fn random_policy_not_deterministic_flag() {
        let space = ActionSpace::Box {
            low: vec![-1.0],
            high: vec![1.0],
        };
        let policy = RandomPolicy::new(space, 0);
        assert!(!policy.is_deterministic());
    }

    #[test]
    fn random_policy_name() {
        let space = ActionSpace::Discrete { n: 2 };
        let policy = RandomPolicy::new(space, 0);
        assert_eq!(policy.name(), "RandomPolicy");
    }

    // -- ScriptedPolicy --

    #[test]
    fn scripted_policy_replays_sequence() {
        let actions = vec![
            Action::from(vec![1.0]),
            Action::from(vec![2.0]),
            Action::from(vec![3.0]),
        ];
        let policy = ScriptedPolicy::new(actions);

        assert_eq!(policy.get_action(&dummy_obs()).as_slice(), &[1.0]);
        assert_eq!(policy.get_action(&dummy_obs()).as_slice(), &[2.0]);
        assert_eq!(policy.get_action(&dummy_obs()).as_slice(), &[3.0]);
        // Cycles back
        assert_eq!(policy.get_action(&dummy_obs()).as_slice(), &[1.0]);
    }

    #[test]
    fn scripted_policy_name() {
        let policy = ScriptedPolicy::new(vec![Action::zeros(1)]);
        assert_eq!(policy.name(), "ScriptedPolicy");
    }

    #[test]
    #[should_panic(expected = "at least one action")]
    fn scripted_policy_panics_on_empty() {
        ScriptedPolicy::new(vec![]);
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn policy_types_are_send_sync() {
        assert_send_sync::<ZeroPolicy>();
        assert_send_sync::<ConstantPolicy>();
        assert_send_sync::<RandomPolicy>();
        assert_send_sync::<ScriptedPolicy>();
    }
}
