//! Deterministic seed hierarchy for reproducible simulation.
//!
//! [`SeedHierarchy`] provides a 5-level derivation tree:
//!
//! ```text
//! Run seed
//! └── Env seed (per environment in VecEnv)
//!     └── Episode seed (per episode within an env)
//!         └── Subsystem seed (domain-rand, reward, etc.)
//!             └── Sensor seed (per-sensor noise)
//! ```
//!
//! Child seeds are derived deterministically via hashing, ensuring that the
//! entire simulation is reproducible from a single root seed.

use std::hash::{DefaultHasher, Hash, Hasher};

use bevy::prelude::Resource;

/// Derive a child seed from a parent seed and a string key.
///
/// Uses `DefaultHasher` (SipHash-1-3) for fast, deterministic mixing.
///
/// # Example
///
/// ```
/// use clankers_core::seed::derive_seed;
///
/// let child = derive_seed(42, "env:0");
/// assert_ne!(child, 42); // derived, not identical
/// let child2 = derive_seed(42, "env:0");
/// assert_eq!(child, child2); // deterministic
/// ```
#[must_use]
pub fn derive_seed(parent: u64, key: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    parent.hash(&mut hasher);
    key.hash(&mut hasher);
    hasher.finish()
}

/// Derive a child seed from a parent seed and a numeric index.
///
/// Convenience wrapper for indexed children (env IDs, episode numbers).
///
/// # Example
///
/// ```
/// use clankers_core::seed::derive_seed_indexed;
///
/// let s0 = derive_seed_indexed(42, 0);
/// let s1 = derive_seed_indexed(42, 1);
/// assert_ne!(s0, s1);
/// ```
#[must_use]
pub fn derive_seed_indexed(parent: u64, index: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    parent.hash(&mut hasher);
    index.hash(&mut hasher);
    hasher.finish()
}

/// Hierarchical seed manager for reproducible simulation runs.
///
/// Stores the root (run-level) seed and provides methods to derive
/// deterministic child seeds at each level of the hierarchy.
///
/// # Example
///
/// ```
/// use clankers_core::seed::SeedHierarchy;
///
/// let seeds = SeedHierarchy::new(42);
/// let env_seed = seeds.env_seed(0);
/// let ep_seed = seeds.episode_seed(0, 5);
/// let sub_seed = seeds.subsystem_seed(0, 5, "domain_rand");
/// // All deterministic from root seed 42
/// ```
#[derive(Debug, Clone, Resource)]
pub struct SeedHierarchy {
    root: u64,
}

impl SeedHierarchy {
    /// Create a new hierarchy from a root seed.
    #[must_use]
    pub const fn new(root: u64) -> Self {
        Self { root }
    }

    /// The root (run-level) seed.
    #[must_use]
    pub const fn root(&self) -> u64 {
        self.root
    }

    /// Derive a seed for a specific environment index.
    #[must_use]
    pub fn env_seed(&self, env_index: u16) -> u64 {
        derive_seed_indexed(self.root, u64::from(env_index))
    }

    /// Derive a seed for a specific episode within an environment.
    #[must_use]
    pub fn episode_seed(&self, env_index: u16, episode_number: u64) -> u64 {
        derive_seed_indexed(self.env_seed(env_index), episode_number)
    }

    /// Derive a seed for a named subsystem within an episode.
    #[must_use]
    pub fn subsystem_seed(&self, env_index: u16, episode_number: u64, subsystem: &str) -> u64 {
        derive_seed(self.episode_seed(env_index, episode_number), subsystem)
    }

    /// Derive a seed for a sensor within a subsystem.
    #[must_use]
    pub fn sensor_seed(
        &self,
        env_index: u16,
        episode_number: u64,
        subsystem: &str,
        sensor_name: &str,
    ) -> u64 {
        derive_seed(
            self.subsystem_seed(env_index, episode_number, subsystem),
            sensor_name,
        )
    }

    /// Create a `ChaCha8Rng` from the root seed.
    #[must_use]
    pub fn root_rng(&self) -> rand_chacha::ChaCha8Rng {
        use rand::SeedableRng;
        rand_chacha::ChaCha8Rng::seed_from_u64(self.root)
    }

    /// Create a `ChaCha8Rng` from an env-level seed.
    #[must_use]
    pub fn env_rng(&self, env_index: u16) -> rand_chacha::ChaCha8Rng {
        use rand::SeedableRng;
        rand_chacha::ChaCha8Rng::seed_from_u64(self.env_seed(env_index))
    }

    /// Create a `ChaCha8Rng` from an episode-level seed.
    #[must_use]
    pub fn episode_rng(&self, env_index: u16, episode_number: u64) -> rand_chacha::ChaCha8Rng {
        use rand::SeedableRng;
        rand_chacha::ChaCha8Rng::seed_from_u64(self.episode_seed(env_index, episode_number))
    }
}

impl Default for SeedHierarchy {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn derive_seed_deterministic() {
        let a = derive_seed(42, "hello");
        let b = derive_seed(42, "hello");
        assert_eq!(a, b);
    }

    #[test]
    fn derive_seed_different_keys() {
        let a = derive_seed(42, "a");
        let b = derive_seed(42, "b");
        assert_ne!(a, b);
    }

    #[test]
    fn derive_seed_different_parents() {
        let a = derive_seed(1, "key");
        let b = derive_seed(2, "key");
        assert_ne!(a, b);
    }

    #[test]
    fn derive_seed_indexed_deterministic() {
        let a = derive_seed_indexed(42, 0);
        let b = derive_seed_indexed(42, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn derive_seed_indexed_different() {
        let a = derive_seed_indexed(42, 0);
        let b = derive_seed_indexed(42, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn hierarchy_root() {
        let h = SeedHierarchy::new(42);
        assert_eq!(h.root(), 42);
    }

    #[test]
    fn hierarchy_env_seeds_differ() {
        let h = SeedHierarchy::new(42);
        let s0 = h.env_seed(0);
        let s1 = h.env_seed(1);
        assert_ne!(s0, s1);
    }

    #[test]
    fn hierarchy_episode_seeds_differ() {
        let h = SeedHierarchy::new(42);
        let s0 = h.episode_seed(0, 0);
        let s1 = h.episode_seed(0, 1);
        assert_ne!(s0, s1);
    }

    #[test]
    fn hierarchy_subsystem_seeds_differ() {
        let h = SeedHierarchy::new(42);
        let a = h.subsystem_seed(0, 0, "domain_rand");
        let b = h.subsystem_seed(0, 0, "reward");
        assert_ne!(a, b);
    }

    #[test]
    fn hierarchy_sensor_seeds_differ() {
        let h = SeedHierarchy::new(42);
        let a = h.sensor_seed(0, 0, "obs", "joint_state");
        let b = h.sensor_seed(0, 0, "obs", "joint_torque");
        assert_ne!(a, b);
    }

    #[test]
    fn hierarchy_deterministic_across_instances() {
        let h1 = SeedHierarchy::new(100);
        let h2 = SeedHierarchy::new(100);
        assert_eq!(h1.env_seed(3), h2.env_seed(3));
        assert_eq!(h1.episode_seed(3, 10), h2.episode_seed(3, 10));
        assert_eq!(
            h1.subsystem_seed(3, 10, "foo"),
            h2.subsystem_seed(3, 10, "foo")
        );
    }

    #[test]
    fn hierarchy_rng_produces_values() {
        let h = SeedHierarchy::new(42);
        let mut rng = h.root_rng();
        let val: f64 = rng.r#gen::<f64>();
        assert!((0.0..1.0).contains(&val));
    }

    #[test]
    fn hierarchy_env_rng_deterministic() {
        let h = SeedHierarchy::new(42);
        let mut rng1 = h.env_rng(0);
        let mut rng2 = h.env_rng(0);
        let v1: f64 = rng1.r#gen::<f64>();
        let v2: f64 = rng2.r#gen::<f64>();
        assert!((v1 - v2).abs() < f64::EPSILON);
    }

    #[test]
    fn hierarchy_episode_rng_deterministic() {
        let h = SeedHierarchy::new(42);
        let mut rng1 = h.episode_rng(0, 5);
        let mut rng2 = h.episode_rng(0, 5);
        let v1: f64 = rng1.r#gen::<f64>();
        let v2: f64 = rng2.r#gen::<f64>();
        assert!((v1 - v2).abs() < f64::EPSILON);
    }

    #[test]
    fn hierarchy_default() {
        let h = SeedHierarchy::default();
        assert_eq!(h.root(), 0);
    }
}
