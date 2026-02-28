//! Swing leg trajectory planner.
//!
//! When a foot is in swing phase, it follows a trajectory from the current
//! position to a target landing position, with a smooth height profile.
//!
//! Supports two trajectory types:
//! - **Min-jerk quintic** (default): zero velocity/acceleration at endpoints
//! - **Parabolic arc**: simpler, nonzero endpoint velocities

use nalgebra::Vector3;

/// Configuration for swing leg trajectories.
#[derive(Clone, Debug)]
pub struct SwingConfig {
    /// Maximum foot height above the ground during swing (meters).
    pub step_height: f64,
    /// Default step length in the forward direction (meters).
    pub default_step_length: f64,
    /// Cartesian PD proportional gains [x, y, z] for swing foot tracking.
    ///
    /// MIT Cheetah uses Kp=[700, 700, 150]. Scaled for our lighter robot.
    pub kp_cartesian: Vector3<f64>,
    /// Cartesian PD derivative gains [x, y, z] for swing foot tracking.
    pub kd_cartesian: Vector3<f64>,
    /// Capture-point gain for velocity error correction in foot placement.
    ///
    /// The LIPM (linear inverted pendulum model) capture point gives an ideal
    /// gain of `cp_gain * sqrt(z_com / g)`, which adapts to the robot's actual
    /// height. A value of 0.5 matches the theoretical LIPM capture point.
    pub cp_gain: f64,
    /// Maximum foot reach radius from hip (meters).
    ///
    /// Foot targets are clamped to this distance from the hip projection to
    /// stay within the kinematic workspace. MIT Cheetah uses ~0.3m.
    pub max_reach: f64,
}

impl Default for SwingConfig {
    fn default() -> Self {
        Self {
            step_height: 0.10,
            default_step_length: 0.08,
            kp_cartesian: Vector3::new(500.0, 500.0, 500.0),
            kd_cartesian: Vector3::new(20.0, 20.0, 20.0),
            cp_gain: 0.5,
            max_reach: 0.3,
        }
    }
}

/// Compute a swing foot position using min-jerk quintic interpolation.
///
/// The min-jerk polynomial `s(t) = 10t^3 - 15t^4 + 6t^5` ensures:
/// - Zero velocity at start (t=0) and end (t=1)
/// - Zero acceleration at start and end
/// - Smooth, continuous motion
///
/// Height profile uses a symmetric bump: `64 * s^3 * (1-s)^3 * step_height`
/// which also has zero velocity at endpoints.
///
/// # Arguments
/// * `start` - Foot position at liftoff (world frame)
/// * `target` - Foot position at touchdown (world frame)
/// * `phase` - Swing phase in [0, 1] (0=liftoff, 1=touchdown)
/// * `step_height` - Maximum height above ground during swing
pub fn swing_foot_position(
    start: &Vector3<f64>,
    target: &Vector3<f64>,
    phase: f64,
    step_height: f64,
) -> Vector3<f64> {
    let t = phase.clamp(0.0, 1.0);

    // Min-jerk smooth interpolation: s = 10t^3 - 15t^4 + 6t^5
    let t2 = t * t;
    let t3 = t2 * t;
    let s = 10.0 * t3 - 15.0 * t2 * t2 + 6.0 * t2 * t3;

    // XY: smooth interpolation
    let xy = start + (target - start) * s;

    // Z: smooth interpolation + symmetric height bump
    // Height bump: 64 * t^3 * (1-t)^3 peaks at t=0.5 with value step_height
    // This function has zero first and second derivatives at t=0 and t=1
    let u = t * (1.0 - t);
    let height_offset = 64.0 * u * u * u * step_height;
    let z_interp = start.z + (target.z - start.z) * s;

    Vector3::new(xy.x, xy.y, z_interp + height_offset)
}

/// Compute the desired swing foot velocity at a given phase.
///
/// Analytical derivative of the min-jerk trajectory with respect to time.
pub fn swing_foot_velocity(
    start: &Vector3<f64>,
    target: &Vector3<f64>,
    phase: f64,
    step_height: f64,
    swing_duration: f64,
) -> Vector3<f64> {
    let t = phase.clamp(0.0, 1.0);
    if swing_duration < 1e-10 {
        return Vector3::zeros();
    }

    let inv_dur = 1.0 / swing_duration;

    // ds/dt = (30t^2 - 60t^3 + 30t^4) / T
    let t2 = t * t;
    let ds_dt = (30.0 * t2 - 60.0 * t * t2 + 30.0 * t2 * t2) * inv_dur;

    // d/dt of XY interpolation
    let diff = target - start;
    let dxy = diff * ds_dt;

    // d/dt of height bump: 64 * 3 * t^2 * (1-t)^3 * (-1) + 64 * t^3 * 3 * (1-t)^2 * (-1)
    // Simplify: d/dt[64 t^3 (1-t)^3] = 64 * 3 * t^2 * (1-t)^2 * [(1-t) - t]
    //         = 192 * t^2 * (1-t)^2 * (1-2t)
    let dh_dt = 192.0 * t * t * (1.0 - t) * (1.0 - t) * (1.0 - 2.0 * t) * inv_dur * step_height;

    let dz_interp = (target.z - start.z) * ds_dt;

    Vector3::new(dxy.x, dxy.y, dz_interp + dh_dt)
}

/// Compute the target landing position for a foot using the Raibert heuristic
/// with capture-point velocity correction.
///
/// The foot target accounts for body displacement during swing and uses
/// LIPM-derived capture-point gain for velocity error correction:
/// ```text
/// hip_at_td = hip_now + v_body * T_swing       // predicted hip at touchdown
/// kv = cp_gain * sqrt(z_com / g)               // capture-point feedback gain
/// target = hip_at_td + v_body * T_stance/2 + kv * (v_body - v_desired)
/// ```
///
/// The height-dependent `kv` automatically scales correction strength with
/// the robot's actual standing height, matching the inverted pendulum dynamics.
#[allow(clippy::too_many_arguments)]
pub fn raibert_foot_target(
    hip_position: &Vector3<f64>,
    body_velocity: &Vector3<f64>,
    desired_velocity: &Vector3<f64>,
    stance_duration: f64,
    swing_duration: f64,
    ground_height: f64,
    cp_gain: f64,
    body_height: f64,
    gravity: f64,
    max_reach: f64,
) -> Vector3<f64> {
    // Predict where the hip will be when the foot touches down
    let hip_at_touchdown = hip_position + body_velocity * swing_duration;
    // Raibert symmetry offset
    let vel_offset = body_velocity * (stance_duration * 0.5);
    // Capture-point velocity error correction: kv = cp_gain * sqrt(z/g)
    let kv = cp_gain * (body_height / gravity).max(0.0).sqrt();
    let vel_correction = (body_velocity - desired_velocity) * kv;

    let mut target = hip_at_touchdown + vel_offset + vel_correction;
    // Clamp foot placement to max_reach from hip (reachable-set constraint)
    let offset_xy = Vector3::new(target.x - hip_position.x, target.y - hip_position.y, 0.0);
    let dist = offset_xy.norm();
    if dist > max_reach {
        let scale = max_reach / dist;
        target.x = hip_position.x + offset_xy.x * scale;
        target.y = hip_position.y + offset_xy.y * scale;
    }
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
    fn swing_peak_height_at_midpoint() {
        let start = Vector3::new(0.0, 0.0, 0.0);
        let target = Vector3::new(0.1, 0.0, 0.0);
        let step_height = 0.06;
        let pos = swing_foot_position(&start, &target, 0.5, step_height);

        // 64 * 0.5^3 * 0.5^3 = 64 * 0.015625 = 1.0
        assert_relative_eq!(pos.z, step_height, epsilon = 1e-10);
    }

    #[test]
    fn swing_smooth_arc() {
        let start = Vector3::new(0.0, 0.0, 0.0);
        let target = Vector3::new(0.1, 0.0, 0.0);

        let h1 = swing_foot_position(&start, &target, 0.25, 0.06).z;
        let h2 = swing_foot_position(&start, &target, 0.50, 0.06).z;
        let h3 = swing_foot_position(&start, &target, 0.75, 0.06).z;

        assert!(h2 > h1, "Peak should be highest");
        assert!(h2 > h3, "Peak should be highest");
        assert_relative_eq!(h1, h3, epsilon = 1e-10); // Symmetric
    }

    #[test]
    fn swing_velocity_zero_at_endpoints() {
        let start = Vector3::new(0.0, 0.0, 0.0);
        let target = Vector3::new(0.1, 0.0, 0.0);
        let dur = 0.2;

        // Velocity should be zero at start and end (min-jerk property)
        let v_start = swing_foot_velocity(&start, &target, 0.0, 0.06, dur);
        let v_end = swing_foot_velocity(&start, &target, 1.0, 0.06, dur);

        assert_relative_eq!(v_start.norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(v_end.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn swing_velocity_nonzero_at_midpoint() {
        let start = Vector3::new(0.0, 0.0, 0.0);
        let target = Vector3::new(0.1, 0.0, 0.0);
        let dur = 0.2;

        let v_mid = swing_foot_velocity(&start, &target, 0.5, 0.06, dur);

        // At midpoint, XY velocity should be maximal (for min-jerk)
        // ds/dt at t=0.5 = 30*0.25 - 60*0.125 + 30*0.0625 = 7.5 - 7.5 + 1.875 = 1.875
        // Actually: 30*(0.25) - 60*(0.125) + 30*(0.0625) = 7.5 - 7.5 + 1.875 = 1.875
        // vx = (0.1 - 0) * 1.875 / 0.2 = 0.9375
        assert!(v_mid.x > 0.5, "vx at midpoint should be significant");

        // Z velocity should be zero at midpoint (height bump peak)
        assert_relative_eq!(v_mid.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn raibert_stationary() {
        let hip = Vector3::new(0.15, 0.08, 0.35);
        let vel = Vector3::zeros();
        let des_vel = Vector3::zeros();
        let target =
            raibert_foot_target(&hip, &vel, &des_vel, 0.2, 0.2, 0.0, 0.5, 0.32, 9.81, 0.3);

        assert_relative_eq!(target.x, hip.x, epsilon = 1e-10);
        assert_relative_eq!(target.y, hip.y, epsilon = 1e-10);
        assert_relative_eq!(target.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn raibert_capture_point_correction() {
        let hip = Vector3::new(0.15, 0.08, 0.35);
        let vel = Vector3::new(0.5, 0.0, 0.0);
        let des_vel = Vector3::new(0.3, 0.0, 0.0);
        let cp_gain = 0.5;
        let height = 0.32;
        let gravity = 9.81;
        let stance_dur = 0.2;
        let swing_dur = 0.2;

        let target = raibert_foot_target(
            &hip, &vel, &des_vel, stance_dur, swing_dur, 0.0, cp_gain, height, gravity, 0.3,
        );

        // kv = 0.5 * sqrt(0.32/9.81) ≈ 0.5 * 0.1806 ≈ 0.0903
        let kv = cp_gain * (height / gravity).sqrt();
        // hip_at_td_x = 0.15 + 0.5*0.2 = 0.25
        // target_x = 0.25 + 0.5*0.1 + kv*(0.5-0.3) = 0.25 + 0.05 + kv*0.2
        let expected_x = 0.25 + 0.05 + kv * 0.2;
        assert_relative_eq!(target.x, expected_x, epsilon = 1e-10);
    }

    #[test]
    fn raibert_height_scales_correction() {
        let hip = Vector3::new(0.0, 0.0, 0.35);
        let vel = Vector3::new(1.0, 0.0, 0.0);
        let des_vel = Vector3::zeros();

        let t_low = raibert_foot_target(
            &hip, &vel, &des_vel, 0.2, 0.2, 0.0, 0.5, 0.20, 9.81, 0.5,
        );
        let t_high = raibert_foot_target(
            &hip, &vel, &des_vel, 0.2, 0.2, 0.0, 0.5, 0.40, 9.81, 0.5,
        );

        // Higher body → larger CP correction → foot placed further forward
        assert!(t_high.x > t_low.x, "Higher body should produce larger correction");
    }

    #[test]
    fn raibert_reach_clamping() {
        let hip = Vector3::new(0.0, 0.0, 0.35);
        let vel = Vector3::new(5.0, 0.0, 0.0); // very fast → large offset
        let des_vel = Vector3::zeros();
        let max_reach = 0.2;

        let target = raibert_foot_target(
            &hip, &vel, &des_vel, 0.2, 0.2, 0.0, 0.5, 0.32, 9.81, max_reach,
        );

        let offset = ((target.x - hip.x).powi(2) + (target.y - hip.y).powi(2)).sqrt();
        assert!(
            offset <= max_reach + 1e-10,
            "Offset {offset} exceeds max_reach {max_reach}"
        );
    }
}
