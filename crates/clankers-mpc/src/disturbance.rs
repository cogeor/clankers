//! Disturbance estimator for compensating model mismatch in the MPC.
//!
//! Maintains a 6D bias estimate (3 angular velocity + 3 linear velocity)
//! that captures persistent prediction errors. The bias is updated each
//! control step using the innovation (measured - predicted state) and
//! fed back to the MPC as a correction to the initial state vector.
//!
//! This achieves the same model-mismatch rejection as full state augmentation
//! (19D) while keeping the QP formulation at 13D for efficiency.

use nalgebra::{DVector, Vector3};

/// Configuration for the disturbance estimator.
#[derive(Clone, Debug)]
pub struct DisturbanceEstimatorConfig {
    /// Innovation filter gain (0, 1]. Higher = faster adaptation, more noise.
    /// Typical: 0.05–0.2.
    pub alpha: f64,
    /// Maximum bias magnitude per component (rad/s or m/s). Clamps the estimate
    /// to prevent runaway from sensor noise or outliers.
    pub max_bias: f64,
}

impl Default for DisturbanceEstimatorConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            max_bias: 1.0,
        }
    }
}

/// Exponential-moving-average disturbance estimator.
///
/// Tracks the persistent component of the prediction error in angular and
/// linear velocity, which captures unmodeled effects like:
/// - Mass/inertia mismatch
/// - Unmodeled friction and contact dynamics
/// - Actuator delays and bandwidth limits
/// - Terrain slope and external pushes
#[derive(Clone, Debug)]
pub struct DisturbanceEstimator {
    config: DisturbanceEstimatorConfig,
    /// Estimated angular velocity bias (world frame, rad/s).
    pub omega_bias: Vector3<f64>,
    /// Estimated linear velocity bias (world frame, m/s).
    pub velocity_bias: Vector3<f64>,
    /// Previous state prediction for computing innovation.
    prev_predicted: Option<DVector<f64>>,
}

impl DisturbanceEstimator {
    /// Create a new estimator with zero initial bias.
    pub fn new(config: DisturbanceEstimatorConfig) -> Self {
        Self {
            config,
            omega_bias: Vector3::zeros(),
            velocity_bias: Vector3::zeros(),
            prev_predicted: None,
        }
    }

    /// Update the bias estimate from prediction error.
    ///
    /// Call this each control step BEFORE the MPC solve:
    /// 1. Compare the actual measured state with what the model predicted
    /// 2. Update the bias estimate via exponential smoothing
    /// 3. Use `compensate_state` to correct x0 before feeding to MPC
    ///
    /// # Arguments
    /// * `x_measured` - Current 13D state from sensors
    /// * `x_predicted` - State predicted by the model from the previous step
    ///   (i.e., A*x_prev + B*u_prev). If `None`, only stores prediction.
    pub fn update(&mut self, x_measured: &DVector<f64>, x_predicted: Option<&DVector<f64>>) {
        if let Some(pred) = x_predicted.or(self.prev_predicted.as_ref()) {
            let alpha = self.config.alpha;
            let max_b = self.config.max_bias;

            // Innovation: measured - predicted in velocity subspace
            for i in 0..3 {
                let e_omega = x_measured[6 + i] - pred[6 + i];
                self.omega_bias[i] = ((1.0 - alpha) * self.omega_bias[i] + alpha * e_omega)
                    .clamp(-max_b, max_b);

                let e_vel = x_measured[9 + i] - pred[9 + i];
                self.velocity_bias[i] = ((1.0 - alpha) * self.velocity_bias[i] + alpha * e_vel)
                    .clamp(-max_b, max_b);
            }
        }
    }

    /// Store the one-step-ahead prediction for the next update cycle.
    ///
    /// Call this AFTER the MPC solve with the predicted next state
    /// (first step of the state trajectory).
    pub fn set_prediction(&mut self, x_predicted: DVector<f64>) {
        self.prev_predicted = Some(x_predicted);
    }

    /// Apply bias compensation to the MPC initial state.
    ///
    /// Subtracts the estimated bias from velocity states so the MPC "sees"
    /// a state that's corrected for persistent model errors.
    pub fn compensate_state(&self, x0: &DVector<f64>) -> DVector<f64> {
        let mut x_comp = x0.clone();
        for i in 0..3 {
            x_comp[6 + i] -= self.omega_bias[i];
            x_comp[9 + i] -= self.velocity_bias[i];
        }
        x_comp
    }

    /// Total bias norm (useful for diagnostics).
    pub fn bias_norm(&self) -> f64 {
        (self.omega_bias.norm_squared() + self.velocity_bias.norm_squared()).sqrt()
    }

    /// Reset the estimator to zero bias.
    pub fn reset(&mut self) {
        self.omega_bias = Vector3::zeros();
        self.velocity_bias = Vector3::zeros();
        self.prev_predicted = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_state() -> DVector<f64> {
        DVector::zeros(crate::types::STATE_DIM)
    }

    #[test]
    fn initial_bias_is_zero() {
        let est = DisturbanceEstimator::new(DisturbanceEstimatorConfig::default());
        assert_eq!(est.omega_bias, Vector3::zeros());
        assert_eq!(est.velocity_bias, Vector3::zeros());
        assert_eq!(est.bias_norm(), 0.0);
    }

    #[test]
    fn bias_tracks_persistent_error() {
        let mut est = DisturbanceEstimator::new(DisturbanceEstimatorConfig {
            alpha: 0.5,
            max_bias: 10.0,
        });

        let pred = zero_state();
        let mut measured = zero_state();
        measured[9] = 1.0; // vx error of 1.0

        // After first update: bias = 0.5 * 1.0 = 0.5
        est.update(&measured, Some(&pred));
        assert!((est.velocity_bias.x - 0.5).abs() < 1e-10);

        // After second update with same error: bias = 0.5*0.5 + 0.5*1.0 = 0.75
        est.update(&measured, Some(&pred));
        assert!((est.velocity_bias.x - 0.75).abs() < 1e-10);
    }

    #[test]
    fn bias_clamped_at_max() {
        let mut est = DisturbanceEstimator::new(DisturbanceEstimatorConfig {
            alpha: 1.0, // instant adaptation
            max_bias: 0.5,
        });

        let pred = zero_state();
        let mut measured = zero_state();
        measured[9] = 10.0; // huge vx error

        est.update(&measured, Some(&pred));
        assert!((est.velocity_bias.x - 0.5).abs() < 1e-10);
    }

    #[test]
    fn compensate_subtracts_bias() {
        let mut est = DisturbanceEstimator::new(DisturbanceEstimatorConfig {
            alpha: 1.0,
            max_bias: 10.0,
        });

        let pred = zero_state();
        let mut measured = zero_state();
        measured[6] = 0.1; // omega_x error
        measured[9] = 0.5; // vx error
        est.update(&measured, Some(&pred));

        let x0 = measured.clone();
        let x_comp = est.compensate_state(&x0);

        // Compensated state should have reduced velocity
        assert!((x_comp[6] - 0.0).abs() < 1e-10); // 0.1 - 0.1 = 0
        assert!((x_comp[9] - 0.0).abs() < 1e-10); // 0.5 - 0.5 = 0
    }

    #[test]
    fn reset_clears_state() {
        let mut est = DisturbanceEstimator::new(DisturbanceEstimatorConfig {
            alpha: 1.0,
            max_bias: 10.0,
        });

        let pred = zero_state();
        let mut measured = zero_state();
        measured[9] = 1.0;
        est.update(&measured, Some(&pred));
        assert!(est.bias_norm() > 0.0);

        est.reset();
        assert_eq!(est.bias_norm(), 0.0);
        assert!(est.prev_predicted.is_none());
    }
}
