//! Swing leg trajectory planner using cubic Bezier curves.
//!
//! When a foot is in swing phase, it follows a trajectory from the current
//! position to a target landing position, with a parabolic height profile
//! to clear the ground.

use nalgebra::Vector3;

/// Configuration for swing leg trajectories.
#[derive(Clone, Debug)]
pub struct SwingConfig {
    /// Maximum foot height above the ground during swing (meters).
    pub step_height: f64,
    /// Default step length in the forward direction (meters).
    /// Used when no explicit target is provided.
    pub default_step_length: f64,
}

impl Default for SwingConfig {
    fn default() -> Self {
        Self {
            step_height: 0.05,
            default_step_length: 0.08,
        }
    }
}

/// Compute a swing foot position at a given phase along the trajectory.
///
/// Uses a cubic Bezier curve in XY with a parabolic height profile.
///
/// # Arguments
/// * `start` - Foot position at liftoff (world frame)
/// * `target` - Foot position at touchdown (world frame)
/// * `phase` - Swing phase in [0, 1] (0=liftoff, 1=touchdown)
/// * `step_height` - Maximum height above ground during swing
///
/// # Returns
/// Desired foot position at the given phase.
pub fn swing_foot_position(
    start: &Vector3<f64>,
    target: &Vector3<f64>,
    phase: f64,
    step_height: f64,
) -> Vector3<f64> {
    let phase = phase.clamp(0.0, 1.0);

    // XY: linear interpolation from start to target
    let xy = start + (target - start) * phase;

    // Z: linear interpolation plus parabolic arc peaking at phase=0.5
    // h(t) = 4 * step_height * t * (1 - t)
    let height_offset = 4.0 * step_height * phase * (1.0 - phase);
    let z_interp = start.z + (target.z - start.z) * phase;

    Vector3::new(xy.x, xy.y, z_interp + height_offset)
}

/// Compute the target landing position for a foot.
///
/// Uses the Raibert heuristic: place the foot under the hip plus a velocity-
/// proportional offset for stability.
///
/// target = hip_pos + v * T_stance/2
///
/// # Arguments
/// * `hip_position` - Hip joint position in world frame
/// * `body_velocity` - Body linear velocity in world frame
/// * `stance_duration` - Expected stance phase duration (seconds)
pub fn raibert_foot_target(
    hip_position: &Vector3<f64>,
    body_velocity: &Vector3<f64>,
    stance_duration: f64,
    ground_height: f64,
) -> Vector3<f64> {
    let mut target = *hip_position + body_velocity * (stance_duration * 0.5);
    target.z = ground_height;
    target
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn swing_starts_at_start() {
        let start = Vector3::new(0.1, 0.05, 0.0);
        let target = Vector3::new(0.2, 0.05, 0.0);
        let pos = swing_foot_position(&start, &target, 0.0, 0.05);
        assert_relative_eq!(pos, start, epsilon = 1e-10);
    }

    #[test]
    fn swing_ends_at_target() {
        let start = Vector3::new(0.1, 0.05, 0.0);
        let target = Vector3::new(0.2, 0.05, 0.0);
        let pos = swing_foot_position(&start, &target, 1.0, 0.05);
        assert_relative_eq!(pos, target, epsilon = 1e-10);
    }

    #[test]
    fn swing_peak_height() {
        let start = Vector3::new(0.0, 0.0, 0.0);
        let target = Vector3::new(0.1, 0.0, 0.0);
        let step_height = 0.05;
        let pos = swing_foot_position(&start, &target, 0.5, step_height);

        // At phase 0.5: height = 4 * 0.05 * 0.5 * 0.5 = 0.05
        assert_relative_eq!(pos.z, step_height, epsilon = 1e-10);
        // X should be midpoint
        assert_relative_eq!(pos.x, 0.05, epsilon = 1e-10);
    }

    #[test]
    fn swing_smooth_arc() {
        let start = Vector3::new(0.0, 0.0, 0.0);
        let target = Vector3::new(0.1, 0.0, 0.0);

        // Height should increase then decrease
        let h1 = swing_foot_position(&start, &target, 0.25, 0.05).z;
        let h2 = swing_foot_position(&start, &target, 0.50, 0.05).z;
        let h3 = swing_foot_position(&start, &target, 0.75, 0.05).z;

        assert!(h2 > h1, "Peak should be highest");
        assert!(h2 > h3, "Peak should be highest");
        assert_relative_eq!(h1, h3, epsilon = 1e-10); // Symmetric
    }

    #[test]
    fn raibert_stationary() {
        let hip = Vector3::new(0.15, 0.08, 0.35);
        let vel = Vector3::zeros();
        let target = raibert_foot_target(&hip, &vel, 0.2, 0.0);

        // Stationary: foot should be directly below hip
        assert_relative_eq!(target.x, hip.x, epsilon = 1e-10);
        assert_relative_eq!(target.y, hip.y, epsilon = 1e-10);
        assert_relative_eq!(target.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn raibert_forward_velocity() {
        let hip = Vector3::new(0.15, 0.08, 0.35);
        let vel = Vector3::new(0.5, 0.0, 0.0);
        let stance_dur = 0.2;
        let target = raibert_foot_target(&hip, &vel, stance_dur, 0.0);

        // target_x = 0.15 + 0.5 * 0.1 = 0.2
        assert_relative_eq!(target.x, 0.2, epsilon = 1e-10);
        assert_relative_eq!(target.z, 0.0, epsilon = 1e-10);
    }
}
