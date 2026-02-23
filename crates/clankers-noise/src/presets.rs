//! Pre-configured noise models for common sensors.
//!
//! Parameters are derived from typical MEMS datasheets and standard robotics
//! sensor specifications.  All presets return validated [`NoiseModel`]
//! instances wrapped in `Result`.
//!
//! # IMU Noise Conversion
//!
//! IMU datasheets specify noise in spectral density units.  Convert to
//! simulation parameters:
//!
//! ```text
//! white_noise_std = noise_density * sqrt(sample_rate_hz)
//! drift_std       = bias_random_walk * sqrt(sample_rate_hz)
//! ```

use crate::model::{NoiseError, NoiseModel};
use crate::vector::IndependentAxesNoise;

// ---------------------------------------------------------------------------
// IMU presets
// ---------------------------------------------------------------------------

/// MEMS gyroscope noise chain (white + bias + drift) for one axis.
///
/// Typical consumer MEMS gyroscope parameters:
/// - Noise density: `0.000_18 rad/s/sqrt(Hz)`
/// - Bias random walk: `0.000_04 rad/s²/sqrt(Hz)`
/// - Turn-on bias std: `0.01 rad/s`
///
/// # Errors
///
/// Returns [`NoiseError`] if `sample_rate_hz` produces invalid derived
/// parameters (e.g., zero or negative).
pub fn mems_gyro(sample_rate_hz: f32) -> Result<NoiseModel, NoiseError> {
    let dt = 1.0 / sample_rate_hz;
    let white_std = 0.000_18 * sample_rate_hz.sqrt();
    let drift_std_val = 0.000_04 * sample_rate_hz.sqrt();
    let bias_std = 0.01;

    Ok(NoiseModel::chain(vec![
        NoiseModel::gaussian_zero_mean(white_std)?,
        NoiseModel::bias(bias_std)?,
        NoiseModel::drift(bias_std, drift_std_val, dt)?,
    ]))
}

/// MEMS accelerometer noise chain (white + bias + drift) for one axis.
///
/// Typical consumer MEMS accelerometer parameters:
/// - Noise density: `0.003 m/s²/sqrt(Hz)`
/// - Bias random walk: `0.000_4 m/s³/sqrt(Hz)`
/// - Turn-on bias std: `0.05 m/s²`
///
/// # Errors
///
/// Returns [`NoiseError`] if `sample_rate_hz` produces invalid derived
/// parameters.
pub fn mems_accel(sample_rate_hz: f32) -> Result<NoiseModel, NoiseError> {
    let dt = 1.0 / sample_rate_hz;
    let white_std = 0.003 * sample_rate_hz.sqrt();
    let drift_std_val = 0.000_4 * sample_rate_hz.sqrt();
    let bias_std = 0.05;

    Ok(NoiseModel::chain(vec![
        NoiseModel::gaussian_zero_mean(white_std)?,
        NoiseModel::bias(bias_std)?,
        NoiseModel::drift(bias_std, drift_std_val, dt)?,
    ]))
}

/// 3-axis MEMS gyroscope with independent per-axis noise.
///
/// # Errors
///
/// Returns [`NoiseError`] if `sample_rate_hz` produces invalid derived
/// parameters.
pub fn mems_gyro_3axis(sample_rate_hz: f32) -> Result<IndependentAxesNoise, NoiseError> {
    Ok(IndependentAxesNoise::new(vec![
        mems_gyro(sample_rate_hz)?,
        mems_gyro(sample_rate_hz)?,
        mems_gyro(sample_rate_hz)?,
    ]))
}

/// 3-axis MEMS accelerometer with independent per-axis noise.
///
/// # Errors
///
/// Returns [`NoiseError`] if `sample_rate_hz` produces invalid derived
/// parameters.
pub fn mems_accel_3axis(sample_rate_hz: f32) -> Result<IndependentAxesNoise, NoiseError> {
    Ok(IndependentAxesNoise::new(vec![
        mems_accel(sample_rate_hz)?,
        mems_accel(sample_rate_hz)?,
        mems_accel(sample_rate_hz)?,
    ]))
}

// ---------------------------------------------------------------------------
// Encoder presets
// ---------------------------------------------------------------------------

/// Encoder position noise (zero-mean Gaussian, 1 mrad standard deviation).
///
/// # Errors
///
/// Returns [`NoiseError`] if the std parameter is invalid.
pub fn encoder_position() -> Result<NoiseModel, NoiseError> {
    NoiseModel::gaussian_zero_mean(0.001)
}

/// Encoder velocity noise (zero-mean Gaussian, 10 mrad/s standard deviation).
///
/// # Errors
///
/// Returns [`NoiseError`] if the std parameter is invalid.
pub fn encoder_velocity() -> Result<NoiseModel, NoiseError> {
    NoiseModel::gaussian_zero_mean(0.01)
}

/// Force/torque sensor noise (zero-mean Gaussian).
///
/// # Arguments
///
/// * `newton_std` — standard deviation in Newtons.
///
/// # Errors
///
/// Returns [`NoiseError`] if `newton_std` is invalid.
pub fn force_sensor(newton_std: f32) -> Result<NoiseModel, NoiseError> {
    NoiseModel::gaussian_zero_mean(newton_std)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::VectorNoiseModel;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    #[test]
    fn mems_gyro_produces_valid_chain() {
        let model = mems_gyro(200.0).unwrap();
        assert!(model.is_stateful());
    }

    #[test]
    fn mems_accel_produces_valid_chain() {
        let model = mems_accel(200.0).unwrap();
        assert!(model.is_stateful());
    }

    #[test]
    fn mems_gyro_3axis_has_3_dims() {
        let noise = mems_gyro_3axis(200.0).unwrap();
        assert_eq!(noise.dim(), 3);
    }

    #[test]
    fn mems_accel_3axis_has_3_dims() {
        let noise = mems_accel_3axis(200.0).unwrap();
        assert_eq!(noise.dim(), 3);
    }

    #[test]
    fn mems_gyro_3axis_produces_samples() {
        let mut rng = test_rng();
        let mut noise = mems_gyro_3axis(200.0).unwrap();
        noise.reset(&mut rng);
        let samples = noise.sample_vec(&mut rng);
        assert_eq!(samples.len(), 3);
        // After reset with bias, samples should be non-zero
        assert!(samples.iter().any(|s| s.abs() > f32::EPSILON));
    }

    #[test]
    fn encoder_position_produces_valid_model() {
        let model = encoder_position().unwrap();
        assert!(!model.is_stateful());
    }

    #[test]
    fn encoder_velocity_produces_valid_model() {
        let model = encoder_velocity().unwrap();
        assert!(!model.is_stateful());
    }

    #[test]
    fn force_sensor_produces_valid_model() {
        let model = force_sensor(0.5).unwrap();
        assert!(!model.is_stateful());
    }

    #[test]
    fn mems_gyro_is_deterministic() {
        let samples_a = {
            let mut rng = test_rng();
            let mut model = mems_gyro(200.0).unwrap();
            model.reset(&mut rng);
            (0..50).map(|_| model.sample(&mut rng)).collect::<Vec<_>>()
        };
        let samples_b = {
            let mut rng = test_rng();
            let mut model = mems_gyro(200.0).unwrap();
            model.reset(&mut rng);
            (0..50).map(|_| model.sample(&mut rng)).collect::<Vec<_>>()
        };
        assert_eq!(samples_a, samples_b);
    }
}
