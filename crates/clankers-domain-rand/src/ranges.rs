//! Randomization ranges for parameter sampling.
//!
//! A [`RandomizationRange`] describes how a single scalar parameter should
//! be randomized.  Call [`sample`](RandomizationRange::sample) with an RNG
//! to draw a value.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from constructing a randomization range.
#[derive(Debug, Error)]
pub enum RangeError {
    #[error("invalid bounds: low ({low}) >= high ({high})")]
    InvalidBounds { low: f32, high: f32 },

    #[error("invalid standard deviation: {0} (must be >= 0 and finite)")]
    InvalidStd(f32),

    #[error("log-uniform bounds must be positive: low={low}, high={high}")]
    NonPositiveBounds { low: f32, high: f32 },

    #[error("value is not finite: {0}")]
    NonFinite(f32),
}

// ---------------------------------------------------------------------------
// RandomizationRange
// ---------------------------------------------------------------------------

/// Describes how a parameter should be randomized on episode reset.
#[derive(Clone, Debug)]
pub enum RandomizationRange {
    /// Always returns the same value.
    Fixed(f32),

    /// Uniform distribution over `[low, high)`.
    Uniform { low: f32, high: f32 },

    /// Gaussian distribution with given mean and standard deviation.
    Gaussian { mean: f32, std: f32 },

    /// Log-uniform distribution: `exp(Uniform(ln(low), ln(high)))`.
    ///
    /// Useful for parameters that span orders of magnitude (e.g., friction).
    LogUniform { low: f32, high: f32 },

    /// Multiplier: samples from `[1-fraction, 1+fraction]` and multiplies the
    /// nominal value. E.g., `Scaling { nominal: 10.0, fraction: 0.1 }` gives
    /// values in `[9.0, 11.0]`.
    Scaling { nominal: f32, fraction: f32 },
}

impl RandomizationRange {
    /// Create a fixed (constant) range.
    pub const fn fixed(value: f32) -> Result<Self, RangeError> {
        if !value.is_finite() {
            return Err(RangeError::NonFinite(value));
        }
        Ok(Self::Fixed(value))
    }

    /// Create a uniform range.
    pub fn uniform(low: f32, high: f32) -> Result<Self, RangeError> {
        if low >= high {
            return Err(RangeError::InvalidBounds { low, high });
        }
        if !low.is_finite() || !high.is_finite() {
            return Err(RangeError::InvalidBounds { low, high });
        }
        Ok(Self::Uniform { low, high })
    }

    /// Create a Gaussian range.
    pub fn gaussian(mean: f32, std: f32) -> Result<Self, RangeError> {
        if !std.is_finite() || std < 0.0 {
            return Err(RangeError::InvalidStd(std));
        }
        if !mean.is_finite() {
            return Err(RangeError::NonFinite(mean));
        }
        Ok(Self::Gaussian { mean, std })
    }

    /// Create a log-uniform range.
    pub fn log_uniform(low: f32, high: f32) -> Result<Self, RangeError> {
        if low <= 0.0 || high <= 0.0 {
            return Err(RangeError::NonPositiveBounds { low, high });
        }
        if low >= high {
            return Err(RangeError::InvalidBounds { low, high });
        }
        if !low.is_finite() || !high.is_finite() {
            return Err(RangeError::InvalidBounds { low, high });
        }
        Ok(Self::LogUniform { low, high })
    }

    /// Create a scaling range.
    pub fn scaling(nominal: f32, fraction: f32) -> Result<Self, RangeError> {
        if !nominal.is_finite() {
            return Err(RangeError::NonFinite(nominal));
        }
        if !fraction.is_finite() || fraction < 0.0 {
            return Err(RangeError::InvalidStd(fraction));
        }
        Ok(Self::Scaling { nominal, fraction })
    }

    /// Sample a value from this range using the given RNG.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        match self {
            Self::Fixed(v) => *v,
            Self::Uniform { low, high } => rng.gen_range(*low..*high),
            Self::Gaussian { mean, std } => {
                if *std == 0.0 {
                    return *mean;
                }
                let dist = Normal::new(f64::from(*mean), f64::from(*std)).unwrap();
                #[allow(clippy::cast_possible_truncation)]
                let val = dist.sample(rng) as f32;
                val
            }
            Self::LogUniform { low, high } => {
                let ln_low = low.ln();
                let ln_high = high.ln();
                let ln_val = rng.gen_range(ln_low..ln_high);
                ln_val.exp()
            }
            Self::Scaling { nominal, fraction } => {
                let scale = rng.gen_range(1.0 - fraction..=1.0 + fraction);
                nominal * scale
            }
        }
    }

    /// Return the nominal (center/expected) value.
    pub fn nominal(&self) -> f32 {
        match self {
            Self::Fixed(v) => *v,
            Self::Uniform { low, high } => (low + high) / 2.0,
            Self::Gaussian { mean, .. } => *mean,
            Self::LogUniform { low, high } => (low * high).sqrt(),
            Self::Scaling { nominal, .. } => *nominal,
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

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    // -- Fixed --

    #[test]
    fn fixed_always_returns_value() {
        let range = RandomizationRange::fixed(3.14).unwrap();
        let mut rng = rng();
        for _ in 0..10 {
            assert!((range.sample(&mut rng) - 3.14).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn fixed_rejects_nan() {
        assert!(RandomizationRange::fixed(f32::NAN).is_err());
    }

    #[test]
    fn fixed_nominal() {
        let r = RandomizationRange::fixed(5.0).unwrap();
        assert!((r.nominal() - 5.0).abs() < f32::EPSILON);
    }

    // -- Uniform --

    #[test]
    fn uniform_samples_in_range() {
        let range = RandomizationRange::uniform(1.0, 5.0).unwrap();
        let mut rng = rng();
        for _ in 0..100 {
            let v = range.sample(&mut rng);
            assert!((1.0..5.0).contains(&v), "got {v}");
        }
    }

    #[test]
    fn uniform_rejects_low_gte_high() {
        assert!(RandomizationRange::uniform(5.0, 5.0).is_err());
        assert!(RandomizationRange::uniform(6.0, 5.0).is_err());
    }

    #[test]
    fn uniform_rejects_inf() {
        assert!(RandomizationRange::uniform(0.0, f32::INFINITY).is_err());
    }

    #[test]
    fn uniform_nominal() {
        let r = RandomizationRange::uniform(2.0, 4.0).unwrap();
        assert!((r.nominal() - 3.0).abs() < f32::EPSILON);
    }

    // -- Gaussian --

    #[test]
    fn gaussian_samples_near_mean() {
        let range = RandomizationRange::gaussian(10.0, 0.1).unwrap();
        let mut rng = rng();
        for _ in 0..100 {
            let v = range.sample(&mut rng);
            assert!((v - 10.0).abs() < 3.0, "got {v}"); // 30 sigma
        }
    }

    #[test]
    fn gaussian_zero_std_returns_mean() {
        let range = RandomizationRange::gaussian(7.0, 0.0).unwrap();
        let mut rng = rng();
        assert!((range.sample(&mut rng) - 7.0).abs() < f32::EPSILON);
    }

    #[test]
    fn gaussian_rejects_negative_std() {
        assert!(RandomizationRange::gaussian(0.0, -1.0).is_err());
    }

    #[test]
    fn gaussian_rejects_nan_mean() {
        assert!(RandomizationRange::gaussian(f32::NAN, 1.0).is_err());
    }

    // -- LogUniform --

    #[test]
    fn log_uniform_samples_in_range() {
        let range = RandomizationRange::log_uniform(0.01, 10.0).unwrap();
        let mut rng = rng();
        for _ in 0..100 {
            let v = range.sample(&mut rng);
            assert!(v >= 0.01 && v < 10.0, "got {v}");
        }
    }

    #[test]
    fn log_uniform_rejects_non_positive() {
        assert!(RandomizationRange::log_uniform(-1.0, 10.0).is_err());
        assert!(RandomizationRange::log_uniform(0.0, 10.0).is_err());
    }

    #[test]
    fn log_uniform_rejects_low_gte_high() {
        assert!(RandomizationRange::log_uniform(5.0, 5.0).is_err());
    }

    #[test]
    fn log_uniform_nominal() {
        let r = RandomizationRange::log_uniform(1.0, 100.0).unwrap();
        assert!((r.nominal() - 10.0).abs() < f32::EPSILON);
    }

    // -- Scaling --

    #[test]
    fn scaling_samples_around_nominal() {
        let range = RandomizationRange::scaling(10.0, 0.1).unwrap();
        let mut rng = rng();
        for _ in 0..100 {
            let v = range.sample(&mut rng);
            assert!(v >= 9.0 && v <= 11.0, "got {v}");
        }
    }

    #[test]
    fn scaling_zero_fraction_returns_nominal() {
        let range = RandomizationRange::scaling(5.0, 0.0).unwrap();
        let mut rng = rng();
        assert!((range.sample(&mut rng) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn scaling_rejects_negative_fraction() {
        assert!(RandomizationRange::scaling(1.0, -0.1).is_err());
    }

    // -- Determinism --

    #[test]
    fn deterministic_with_same_seed() {
        let range = RandomizationRange::uniform(0.0, 100.0).unwrap();
        let mut rng1 = ChaCha8Rng::seed_from_u64(99);
        let mut rng2 = ChaCha8Rng::seed_from_u64(99);
        let v1 = range.sample(&mut rng1);
        let v2 = range.sample(&mut rng2);
        assert!((v1 - v2).abs() < f32::EPSILON);
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn range_is_send_sync() {
        assert_send_sync::<RandomizationRange>();
    }
}
