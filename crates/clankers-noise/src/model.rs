//! Scalar noise models for sensor simulation, action perturbation, and domain
//! randomization.
//!
//! [`NoiseModel`] is an enum with static dispatch — no trait objects, no vtable
//! overhead.  Composition uses the [`Chain`](NoiseModel::Chain) variant which
//! sums the outputs of its children.
//!
//! Every sampling method takes an explicit `&mut R: Rng` parameter so that
//! determinism is guaranteed when the same seed is provided.

use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform as UniformDist};
use std::fmt;

// ---------------------------------------------------------------------------
// NoiseError
// ---------------------------------------------------------------------------

/// Validation errors for noise model parameters.
///
/// Implements [`Copy`] for cheap propagation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseError {
    /// Standard deviation was negative, NaN, or infinite.
    InvalidStdDev { value: f32 },
    /// Range bounds are invalid: `low >= high`, NaN, or infinite.
    InvalidRange { low: f32, high: f32 },
    /// Timestep was `<= 0`, NaN, or infinite.
    InvalidTimestep { dt: f32 },
    /// Quantization step was `<= 0`, NaN, or infinite.
    InvalidStep { step: f32 },
}

impl fmt::Display for NoiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidStdDev { value } => {
                write!(f, "std_dev must be finite and >= 0, got {value}")
            }
            Self::InvalidRange { low, high } => {
                write!(
                    f,
                    "range must satisfy low < high with finite bounds, got [{low}, {high})"
                )
            }
            Self::InvalidTimestep { dt } => {
                write!(f, "dt must be finite and > 0, got {dt}")
            }
            Self::InvalidStep { step } => {
                write!(f, "quantization step must be finite and > 0, got {step}")
            }
        }
    }
}

impl std::error::Error for NoiseError {}

// ---------------------------------------------------------------------------
// NoiseModel
// ---------------------------------------------------------------------------

/// Scalar noise model.
///
/// All variants are statically dispatched via `match`.  Stateful variants
/// ([`Bias`](Self::Bias), [`Drift`](Self::Drift)) must be [`reset`](Self::reset)
/// before the first episode.  Before reset, they return `0.0`.
///
/// # Composition
///
/// Use [`Chain`](Self::Chain) to combine multiple noise models additively.
/// For per-axis noise on multi-dimensional sensors, see
/// [`IndependentAxesNoise`](crate::vector::IndependentAxesNoise).
#[derive(Clone, Debug)]
pub enum NoiseModel {
    /// Additive Gaussian: `N(mean, std²)`.
    Gaussian { mean: f32, std: f32 },
    /// Uniform random in `[low, high)`.
    Uniform { low: f32, high: f32 },
    /// Constant bias sampled once per episode from `N(0, std²)`.
    /// Returns `0.0` before the first [`reset`](Self::reset).
    Bias { std: f32, current: f32 },
    /// Random-walk (discrete Wiener process).
    ///
    /// Each step: `value += N(0, drift_std * sqrt(dt))`.
    /// On reset: `value = N(0, initial_std²)`.
    /// Returns `0.0` before the first [`reset`](Self::reset).
    Drift {
        initial_std: f32,
        drift_std: f32,
        dt: f32,
        current: f32,
        initialized: bool,
    },
    /// Rounds a value to the nearest multiple of `step`.
    ///
    /// Use via [`apply`](Self::apply), not [`sample`](Self::sample).
    /// `sample()` returns `0.0` (additive identity).
    Quantization { step: f32 },
    /// Additive chain — samples each child and sums the results.
    Chain(Vec<Self>),
}

// ---------------------------------------------------------------------------
// Constructors (all validate parameters)
// ---------------------------------------------------------------------------

impl NoiseModel {
    /// Create a Gaussian noise model with the given mean and standard deviation.
    ///
    /// # Errors
    ///
    /// Returns [`NoiseError::InvalidStdDev`] if `std` is negative, NaN, or
    /// infinite.
    pub fn gaussian(mean: f32, std: f32) -> Result<Self, NoiseError> {
        if !std.is_finite() || std < 0.0 {
            return Err(NoiseError::InvalidStdDev { value: std });
        }
        if !mean.is_finite() {
            return Err(NoiseError::InvalidStdDev { value: mean });
        }
        Ok(Self::Gaussian { mean, std })
    }

    /// Convenience constructor for zero-mean Gaussian noise.
    ///
    /// # Errors
    ///
    /// Returns [`NoiseError::InvalidStdDev`] if `std` is negative, NaN, or
    /// infinite.
    pub fn gaussian_zero_mean(std: f32) -> Result<Self, NoiseError> {
        Self::gaussian(0.0, std)
    }

    /// Create a uniform noise model sampling from `[low, high)`.
    ///
    /// # Errors
    ///
    /// Returns [`NoiseError::InvalidRange`] if `low >= high` or either bound
    /// is NaN/infinite.
    pub fn uniform(low: f32, high: f32) -> Result<Self, NoiseError> {
        if !low.is_finite() || !high.is_finite() || low >= high {
            return Err(NoiseError::InvalidRange { low, high });
        }
        Ok(Self::Uniform { low, high })
    }

    /// Convenience constructor for symmetric uniform noise `[-half_range,
    /// half_range)`.
    ///
    /// # Errors
    ///
    /// Returns [`NoiseError::InvalidRange`] if `half_range` is non-positive,
    /// NaN, or infinite.
    pub fn uniform_symmetric(half_range: f32) -> Result<Self, NoiseError> {
        Self::uniform(-half_range, half_range)
    }

    /// Create a constant-bias noise model.  On each [`reset`](Self::reset),
    /// the bias is resampled from `N(0, std²)`.
    ///
    /// # Errors
    ///
    /// Returns [`NoiseError::InvalidStdDev`] if `std` is negative, NaN, or
    /// infinite.
    pub fn bias(std: f32) -> Result<Self, NoiseError> {
        if !std.is_finite() || std < 0.0 {
            return Err(NoiseError::InvalidStdDev { value: std });
        }
        Ok(Self::Bias { std, current: 0.0 })
    }

    /// Create a drift (random-walk) noise model.
    ///
    /// - `initial_std`: standard deviation of the initial value on reset.
    /// - `drift_std`: drift rate in `[signal]/sqrt(s)`.
    /// - `dt`: simulation timestep in seconds.
    ///
    /// # Errors
    ///
    /// Returns [`NoiseError::InvalidStdDev`] if either std is invalid, or
    /// [`NoiseError::InvalidTimestep`] if `dt` is non-positive/non-finite.
    pub fn drift(initial_std: f32, drift_std: f32, dt: f32) -> Result<Self, NoiseError> {
        if !initial_std.is_finite() || initial_std < 0.0 {
            return Err(NoiseError::InvalidStdDev { value: initial_std });
        }
        if !drift_std.is_finite() || drift_std < 0.0 {
            return Err(NoiseError::InvalidStdDev { value: drift_std });
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(NoiseError::InvalidTimestep { dt });
        }
        Ok(Self::Drift {
            initial_std,
            drift_std,
            dt,
            current: 0.0,
            initialized: false,
        })
    }

    /// Create a quantization noise model that rounds values to the nearest
    /// multiple of `step`.
    ///
    /// # Errors
    ///
    /// Returns [`NoiseError::InvalidStep`] if `step` is non-positive, NaN, or
    /// infinite.
    pub fn quantization(step: f32) -> Result<Self, NoiseError> {
        if !step.is_finite() || step <= 0.0 {
            return Err(NoiseError::InvalidStep { step });
        }
        Ok(Self::Quantization { step })
    }

    /// Create a chain that sums the outputs of the given noise models.
    pub const fn chain(models: Vec<Self>) -> Self {
        Self::Chain(models)
    }
}

// ---------------------------------------------------------------------------
// Sampling, application, and reset
// ---------------------------------------------------------------------------

impl NoiseModel {
    /// Sample a single noise value.
    ///
    /// For [`Quantization`](Self::Quantization), returns `0.0` (use
    /// [`apply`](Self::apply) instead).
    #[allow(clippy::cast_possible_truncation)] // intentional f64→f32 for rand_distr
    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> f32 {
        match self {
            Self::Gaussian { mean, std } => {
                if *std == 0.0 {
                    return *mean;
                }
                let dist = Normal::new(f64::from(*mean), f64::from(*std))
                    .expect("validated in constructor");
                dist.sample(rng) as f32
            }
            Self::Uniform { low, high } => {
                let dist = UniformDist::new(*low, *high);
                dist.sample(rng)
            }
            Self::Bias { current, .. } => *current,
            Self::Drift {
                drift_std,
                dt,
                current,
                initialized,
                ..
            } => {
                if !*initialized {
                    return 0.0;
                }
                let step_std = *drift_std * dt.sqrt();
                if step_std > 0.0 {
                    let dist =
                        Normal::new(0.0, f64::from(step_std)).expect("validated in constructor");
                    *current += dist.sample(rng) as f32;
                }
                *current
            }
            Self::Quantization { .. } => 0.0,
            Self::Chain(models) => {
                let mut total = 0.0_f32;
                for model in models.iter_mut() {
                    total += model.sample(rng);
                }
                total
            }
        }
    }

    /// Apply noise to a clean value.
    ///
    /// For most variants this is `value + sample()`.
    /// For [`Quantization`](Self::Quantization), this rounds `value` to the
    /// nearest multiple of `step`.
    /// For [`Chain`](Self::Chain), applies each child sequentially.
    pub fn apply<R: Rng + ?Sized>(&mut self, value: f32, rng: &mut R) -> f32 {
        match self {
            Self::Quantization { step } => (value / *step).round() * *step,
            Self::Chain(models) => {
                let mut v = value;
                for model in models.iter_mut() {
                    v = model.apply(v, rng);
                }
                v
            }
            _ => value + self.sample(rng),
        }
    }

    /// Reset internal state.  Call at episode boundaries.
    ///
    /// - [`Bias`](Self::Bias): resamples the constant offset from `N(0, std²)`.
    /// - [`Drift`](Self::Drift): resamples the initial value from `N(0, initial_std²)`.
    /// - [`Chain`](Self::Chain): resets all children.
    /// - Stateless variants ([`Gaussian`](Self::Gaussian),
    ///   [`Uniform`](Self::Uniform), [`Quantization`](Self::Quantization)):
    ///   no-op.
    #[allow(clippy::cast_possible_truncation)] // intentional f64→f32 for rand_distr
    pub fn reset<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        match self {
            Self::Bias { std, current } => {
                if *std > 0.0 {
                    let dist = Normal::new(0.0, f64::from(*std)).expect("validated in constructor");
                    *current = dist.sample(rng) as f32;
                } else {
                    *current = 0.0;
                }
            }
            Self::Drift {
                initial_std,
                current,
                initialized,
                ..
            } => {
                *initialized = true;
                if *initial_std > 0.0 {
                    let dist = Normal::new(0.0, f64::from(*initial_std))
                        .expect("validated in constructor");
                    *current = dist.sample(rng) as f32;
                } else {
                    *current = 0.0;
                }
            }
            Self::Chain(models) => {
                for model in models.iter_mut() {
                    model.reset(rng);
                }
            }
            _ => {}
        }
    }

    /// Returns `true` if this noise model has internal state that requires
    /// [`reset`](Self::reset) at episode boundaries.
    pub fn is_stateful(&self) -> bool {
        match self {
            Self::Gaussian { .. } | Self::Uniform { .. } | Self::Quantization { .. } => false,
            Self::Bias { .. } | Self::Drift { .. } => true,
            Self::Chain(models) => models.iter().any(Self::is_stateful),
        }
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

    // -- Constructor validation --

    #[test]
    fn gaussian_rejects_negative_std() {
        assert!(NoiseModel::gaussian(0.0, -1.0).is_err());
    }

    #[test]
    fn gaussian_rejects_nan_std() {
        assert!(NoiseModel::gaussian(0.0, f32::NAN).is_err());
    }

    #[test]
    fn gaussian_rejects_inf_std() {
        assert!(NoiseModel::gaussian(0.0, f32::INFINITY).is_err());
    }

    #[test]
    fn gaussian_rejects_nan_mean() {
        assert!(NoiseModel::gaussian(f32::NAN, 1.0).is_err());
    }

    #[test]
    fn gaussian_accepts_zero_std() {
        assert!(NoiseModel::gaussian(1.0, 0.0).is_ok());
    }

    #[test]
    fn uniform_rejects_low_gte_high() {
        assert!(NoiseModel::uniform(1.0, 1.0).is_err());
        assert!(NoiseModel::uniform(2.0, 1.0).is_err());
    }

    #[test]
    fn uniform_rejects_nan_bounds() {
        assert!(NoiseModel::uniform(f32::NAN, 1.0).is_err());
        assert!(NoiseModel::uniform(0.0, f32::NAN).is_err());
    }

    #[test]
    fn uniform_rejects_inf_bounds() {
        assert!(NoiseModel::uniform(f32::NEG_INFINITY, 1.0).is_err());
    }

    #[test]
    fn bias_rejects_negative_std() {
        assert!(NoiseModel::bias(-0.1).is_err());
    }

    #[test]
    fn drift_rejects_negative_initial_std() {
        assert!(NoiseModel::drift(-1.0, 0.1, 0.01).is_err());
    }

    #[test]
    fn drift_rejects_negative_drift_std() {
        assert!(NoiseModel::drift(0.1, -1.0, 0.01).is_err());
    }

    #[test]
    fn drift_rejects_zero_dt() {
        assert!(NoiseModel::drift(0.1, 0.1, 0.0).is_err());
    }

    #[test]
    fn drift_rejects_negative_dt() {
        assert!(NoiseModel::drift(0.1, 0.1, -0.01).is_err());
    }

    #[test]
    fn quantization_rejects_zero_step() {
        assert!(NoiseModel::quantization(0.0).is_err());
    }

    #[test]
    fn quantization_rejects_negative_step() {
        assert!(NoiseModel::quantization(-0.1).is_err());
    }

    // -- Determinism --

    #[test]
    fn gaussian_is_deterministic_with_same_seed() {
        let samples_a: Vec<f32> = {
            let mut rng = test_rng();
            let mut m = NoiseModel::gaussian_zero_mean(1.0).unwrap();
            (0..100).map(|_| m.sample(&mut rng)).collect()
        };
        let samples_b: Vec<f32> = {
            let mut rng = test_rng();
            let mut m = NoiseModel::gaussian_zero_mean(1.0).unwrap();
            (0..100).map(|_| m.sample(&mut rng)).collect()
        };
        assert_eq!(samples_a, samples_b);
    }

    #[test]
    fn uniform_is_deterministic_with_same_seed() {
        let samples_a: Vec<f32> = {
            let mut rng = test_rng();
            let mut m = NoiseModel::uniform(-1.0, 1.0).unwrap();
            (0..100).map(|_| m.sample(&mut rng)).collect()
        };
        let samples_b: Vec<f32> = {
            let mut rng = test_rng();
            let mut m = NoiseModel::uniform(-1.0, 1.0).unwrap();
            (0..100).map(|_| m.sample(&mut rng)).collect()
        };
        assert_eq!(samples_a, samples_b);
    }

    // -- Sampling behavior --

    #[test]
    fn gaussian_zero_std_returns_mean() {
        let mut rng = test_rng();
        let mut m = NoiseModel::gaussian(5.0, 0.0).unwrap();
        for _ in 0..10 {
            assert!((m.sample(&mut rng) - 5.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn uniform_samples_within_range() {
        let mut rng = test_rng();
        let mut m = NoiseModel::uniform(-1.0, 1.0).unwrap();
        for _ in 0..1000 {
            let s = m.sample(&mut rng);
            assert!((-1.0..1.0).contains(&s), "sample {s} out of range");
        }
    }

    #[test]
    fn bias_returns_zero_before_reset() {
        let mut rng = test_rng();
        let mut m = NoiseModel::bias(1.0).unwrap();
        assert!((m.sample(&mut rng)).abs() < f32::EPSILON);
    }

    #[test]
    fn bias_returns_nonzero_after_reset() {
        let mut rng = test_rng();
        let mut m = NoiseModel::bias(10.0).unwrap();
        m.reset(&mut rng);
        // With std=10.0, the probability of sampling exactly 0.0 is negligible.
        let s = m.sample(&mut rng);
        assert!(s.abs() > f32::EPSILON);
    }

    #[test]
    fn bias_returns_constant_between_resets() {
        let mut rng = test_rng();
        let mut m = NoiseModel::bias(1.0).unwrap();
        m.reset(&mut rng);
        let first = m.sample(&mut rng);
        for _ in 0..100 {
            assert!((m.sample(&mut rng) - first).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn drift_returns_zero_before_reset() {
        let mut rng = test_rng();
        let mut m = NoiseModel::drift(1.0, 0.1, 0.01).unwrap();
        assert!((m.sample(&mut rng)).abs() < f32::EPSILON);
    }

    #[test]
    fn drift_accumulates_over_steps() {
        let mut rng = test_rng();
        let mut m = NoiseModel::drift(0.0, 1.0, 0.01).unwrap();
        m.reset(&mut rng); // initial_std=0 → starts at 0
        let mut prev = 0.0_f32;
        let mut changed = false;
        for _ in 0..100 {
            let s = m.sample(&mut rng);
            if (s - prev).abs() > f32::EPSILON {
                changed = true;
            }
            prev = s;
        }
        assert!(changed, "drift should change over steps");
    }

    #[test]
    fn quantization_sample_returns_zero() {
        let mut rng = test_rng();
        let mut m = NoiseModel::quantization(0.1).unwrap();
        assert!((m.sample(&mut rng)).abs() < f32::EPSILON);
    }

    #[test]
    fn quantization_apply_rounds_correctly() {
        let mut rng = test_rng();
        let mut m = NoiseModel::quantization(0.1).unwrap();
        assert!((m.apply(0.15, &mut rng) - 0.2).abs() < 1e-6);
        assert!((m.apply(0.14, &mut rng) - 0.1).abs() < 1e-6);
        // -0.35 / 0.1 = -3.5 → round half away from zero → -4.0 → -0.4
        assert!((m.apply(-0.35, &mut rng) - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn chain_sums_children() {
        let mut rng = test_rng();
        // Two zero-mean gaussians with zero std → both return 0.0
        let mut chain = NoiseModel::chain(vec![
            NoiseModel::gaussian(1.0, 0.0).unwrap(),
            NoiseModel::gaussian(2.0, 0.0).unwrap(),
        ]);
        let s = chain.sample(&mut rng);
        assert!((s - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn chain_reset_propagates() {
        let mut rng = test_rng();
        let mut chain = NoiseModel::chain(vec![
            NoiseModel::bias(10.0).unwrap(),
            NoiseModel::bias(10.0).unwrap(),
        ]);
        // Before reset: both biases are 0.0
        assert!((chain.sample(&mut rng)).abs() < f32::EPSILON);
        // After reset: both biases should be non-zero
        chain.reset(&mut rng);
        let s = chain.sample(&mut rng);
        assert!(s.abs() > f32::EPSILON);
    }

    // -- is_stateful --

    #[test]
    fn stateful_detection() {
        assert!(!NoiseModel::gaussian(0.0, 1.0).unwrap().is_stateful());
        assert!(!NoiseModel::uniform(-1.0, 1.0).unwrap().is_stateful());
        assert!(!NoiseModel::quantization(0.1).unwrap().is_stateful());
        assert!(NoiseModel::bias(1.0).unwrap().is_stateful());
        assert!(NoiseModel::drift(1.0, 0.1, 0.01).unwrap().is_stateful());
    }

    #[test]
    fn chain_stateful_if_any_child_stateful() {
        let chain_stateful = NoiseModel::chain(vec![
            NoiseModel::gaussian(0.0, 1.0).unwrap(),
            NoiseModel::bias(1.0).unwrap(),
        ]);
        assert!(chain_stateful.is_stateful());

        let chain_stateless = NoiseModel::chain(vec![
            NoiseModel::gaussian(0.0, 1.0).unwrap(),
            NoiseModel::uniform(-1.0, 1.0).unwrap(),
        ]);
        assert!(!chain_stateless.is_stateful());
    }

    // -- Error display --

    #[test]
    fn noise_error_display_messages() {
        assert_eq!(
            NoiseError::InvalidStdDev { value: -1.0 }.to_string(),
            "std_dev must be finite and >= 0, got -1"
        );
        assert_eq!(
            NoiseError::InvalidRange {
                low: 1.0,
                high: 0.0
            }
            .to_string(),
            "range must satisfy low < high with finite bounds, got [1, 0)"
        );
        assert_eq!(
            NoiseError::InvalidTimestep { dt: 0.0 }.to_string(),
            "dt must be finite and > 0, got 0"
        );
        assert_eq!(
            NoiseError::InvalidStep { step: -0.5 }.to_string(),
            "quantization step must be finite and > 0, got -0.5"
        );
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn noise_model_is_send_sync() {
        assert_send_sync::<NoiseModel>();
    }

    #[test]
    fn noise_error_is_send_sync() {
        assert_send_sync::<NoiseError>();
    }
}
