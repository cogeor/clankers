//! Multi-dimensional noise models for vector-valued sensors.
//!
//! [`VectorNoiseModel`] is a trait for noise over multiple axes.
//! [`IndependentAxesNoise`] is the primary implementation — one
//! [`NoiseModel`] per axis, sampled independently
//! (no cross-axis correlation).

use crate::model::NoiseModel;
use rand::Rng;

// ---------------------------------------------------------------------------
// VectorNoiseModel trait
// ---------------------------------------------------------------------------

/// Multi-dimensional noise model.
///
/// Each axis may have its own scalar [`NoiseModel`].
/// Implementations must be `Send + Sync` for Bevy compatibility.
pub trait VectorNoiseModel: Send + Sync {
    /// Sample a noise vector.  The returned `Vec` has length [`dim()`](Self::dim).
    fn sample_vec<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<f32>;

    /// Apply noise to a clean vector.  `values` must have length [`dim()`](Self::dim).
    ///
    /// Default implementation adds `sample_vec()` element-wise.
    fn apply_vec<R: Rng + ?Sized>(&mut self, values: &[f32], rng: &mut R) -> Vec<f32> {
        let noise = self.sample_vec(rng);
        values
            .iter()
            .zip(noise.iter())
            .map(|(v, n)| v + n)
            .collect()
    }

    /// Reset internal state for all axes.  Call at episode boundaries.
    fn reset<R: Rng + ?Sized>(&mut self, rng: &mut R);

    /// Number of dimensions (axes).
    fn dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// IndependentAxesNoise
// ---------------------------------------------------------------------------

/// Independent noise per axis — one [`NoiseModel`] per dimension, sampled
/// independently.
///
/// This is the standard implementation of [`VectorNoiseModel`].  No cross-axis
/// correlation is modeled.
#[derive(Clone, Debug)]
pub struct IndependentAxesNoise {
    models: Vec<NoiseModel>,
}

impl IndependentAxesNoise {
    /// Create from a list of per-axis noise models.
    pub const fn new(models: Vec<NoiseModel>) -> Self {
        Self { models }
    }

    /// Create from a single noise model cloned across `dim` axes.
    pub fn uniform_across(model: NoiseModel, dim: usize) -> Self {
        Self {
            models: vec![model; dim],
        }
    }

    /// Returns a reference to the per-axis models.
    pub fn models(&self) -> &[NoiseModel] {
        &self.models
    }

    /// Returns a mutable reference to the per-axis models.
    pub fn models_mut(&mut self) -> &mut [NoiseModel] {
        &mut self.models
    }
}

impl VectorNoiseModel for IndependentAxesNoise {
    fn sample_vec<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<f32> {
        self.models.iter_mut().map(|m| m.sample(rng)).collect()
    }

    fn apply_vec<R: Rng + ?Sized>(&mut self, values: &[f32], rng: &mut R) -> Vec<f32> {
        values
            .iter()
            .zip(self.models.iter_mut())
            .map(|(v, m)| m.apply(*v, rng))
            .collect()
    }

    fn reset<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        for model in &mut self.models {
            model.reset(rng);
        }
    }

    fn dim(&self) -> usize {
        self.models.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn independent_axes_correct_dim() {
        let noise = IndependentAxesNoise::new(vec![
            NoiseModel::gaussian_zero_mean(0.1).unwrap(),
            NoiseModel::gaussian_zero_mean(0.2).unwrap(),
            NoiseModel::gaussian_zero_mean(0.3).unwrap(),
        ]);
        assert_eq!(noise.dim(), 3);
    }

    #[test]
    fn independent_axes_sample_vec_length() {
        let mut rng = test_rng();
        let mut noise =
            IndependentAxesNoise::uniform_across(NoiseModel::gaussian_zero_mean(0.1).unwrap(), 5);
        let samples = noise.sample_vec(&mut rng);
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn independent_axes_apply_vec_length() {
        let mut rng = test_rng();
        let mut noise =
            IndependentAxesNoise::uniform_across(NoiseModel::gaussian_zero_mean(0.1).unwrap(), 3);
        let values = vec![1.0, 2.0, 3.0];
        let result = noise.apply_vec(&values, &mut rng);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn independent_axes_deterministic() {
        let samples_a = {
            let mut rng = test_rng();
            let mut noise = IndependentAxesNoise::uniform_across(
                NoiseModel::gaussian_zero_mean(1.0).unwrap(),
                3,
            );
            noise.sample_vec(&mut rng)
        };
        let samples_b = {
            let mut rng = test_rng();
            let mut noise = IndependentAxesNoise::uniform_across(
                NoiseModel::gaussian_zero_mean(1.0).unwrap(),
                3,
            );
            noise.sample_vec(&mut rng)
        };
        assert_eq!(samples_a, samples_b);
    }

    #[test]
    fn independent_axes_reset_propagates() {
        let mut rng = test_rng();
        let mut noise = IndependentAxesNoise::new(vec![
            NoiseModel::bias(10.0).unwrap(),
            NoiseModel::bias(10.0).unwrap(),
        ]);
        // Before reset: biases are 0.0
        let before = noise.sample_vec(&mut rng);
        assert!(before[0].abs() < f32::EPSILON);
        assert!(before[1].abs() < f32::EPSILON);
        // After reset: biases are non-zero
        noise.reset(&mut rng);
        let after = noise.sample_vec(&mut rng);
        assert!(after[0].abs() > f32::EPSILON || after[1].abs() > f32::EPSILON);
    }

    #[test]
    fn uniform_across_creates_clones() {
        let noise =
            IndependentAxesNoise::uniform_across(NoiseModel::gaussian(1.0, 0.5).unwrap(), 4);
        assert_eq!(noise.dim(), 4);
    }

    #[test]
    fn quantization_apply_vec_works() {
        let mut rng = test_rng();
        let mut noise =
            IndependentAxesNoise::uniform_across(NoiseModel::quantization(0.1).unwrap(), 2);
        let result = noise.apply_vec(&[0.15, 0.24], &mut rng);
        assert!((result[0] - 0.2).abs() < 1e-6);
        assert!((result[1] - 0.2).abs() < 1e-6);
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn independent_axes_noise_is_send_sync() {
        assert_send_sync::<IndependentAxesNoise>();
    }
}
