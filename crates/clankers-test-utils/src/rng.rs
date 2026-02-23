//! Deterministic RNG utilities for reproducible tests.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Create a deterministic `ChaCha8Rng` from a seed.
///
/// All test randomization should go through this to ensure reproducibility.
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Generate a deterministic `Vec<f32>` of length `dim` from a seed.
///
/// Useful for creating consistent test observations or actions.
pub fn deterministic_vec(dim: usize, seed: u64) -> Vec<f32> {
    use rand::Rng;
    let mut rng = seeded_rng(seed);
    (0..dim).map(|_| rng.r#gen::<f32>()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_rng_is_deterministic() {
        use rand::Rng;
        let mut rng1 = seeded_rng(42);
        let mut rng2 = seeded_rng(42);
        let v1: f32 = rng1.r#gen();
        let v2: f32 = rng2.r#gen();
        assert!((v1 - v2).abs() < f32::EPSILON);
    }

    #[test]
    fn deterministic_vec_reproducible() {
        let v1 = deterministic_vec(5, 99);
        let v2 = deterministic_vec(5, 99);
        assert_eq!(v1.len(), 5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn different_seeds_differ() {
        let v1 = deterministic_vec(3, 1);
        let v2 = deterministic_vec(3, 2);
        assert_ne!(v1, v2);
    }
}
