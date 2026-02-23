//! Control modes and PID/PD controllers for actuator position and velocity
//! control.
//!
//! Controllers are implemented in-house (no external `pid` crate dependency)
//! for minimal footprint and full control over the implementation.

// ---------------------------------------------------------------------------
// ControlMode
// ---------------------------------------------------------------------------

/// Control mode configuration.
///
/// Determines how an actuator's command is interpreted.
#[derive(Clone, Debug, Default)]
pub enum ControlMode {
    /// Direct torque command (Nm) or force (N).
    #[default]
    Torque,
    /// Velocity control with PD gains.
    ///
    /// - `kp`: proportional gain on velocity error `(Nm/(rad/s))`.
    /// - `kd`: derivative gain (reserved for acceleration damping; currently
    ///   used for error derivative).
    Velocity { kp: f32, kd: f32 },
    /// Position control with PID gains.
    ///
    /// - `kp`: `Nm/rad`.
    /// - `ki`: `Nm/(rad·s)`.
    /// - `kd`: `Nm·s/rad`.
    Position { kp: f32, ki: f32, kd: f32 },
}

// ---------------------------------------------------------------------------
// PidController
// ---------------------------------------------------------------------------

/// PID controller for position control.
///
/// Output units are Nm (torque command).
///
/// Gains:
/// - `kp`: `Nm/rad` — proportional.
/// - `ki`: `Nm/(rad·s)` — integral.
/// - `kd`: `Nm·s/rad` — derivative.
///
/// Features:
/// - Anti-windup via integral clamping.
/// - Output clamping.
/// - Derivative-of-error (not derivative-of-measurement).
#[derive(Clone, Debug)]
pub struct PidController {
    kp: f32,
    ki: f32,
    kd: f32,
    output_limit: f32,
    integral_limit: f32,
    integral: f32,
    last_error: f32,
    initialized: bool,
}

impl PidController {
    /// Create a new PID controller with the given gains.
    ///
    /// Default output limit: `1000.0 Nm`.
    /// Default integral limit: `100.0`.
    pub const fn new(kp: f32, ki: f32, kd: f32) -> Self {
        Self {
            kp,
            ki,
            kd,
            output_limit: 1000.0,
            integral_limit: 100.0,
            integral: 0.0,
            last_error: 0.0,
            initialized: false,
        }
    }

    /// Set the output clamp limit (symmetric: `[-limit, limit]`).
    pub const fn with_output_limit(mut self, limit: f32) -> Self {
        self.output_limit = limit;
        self
    }

    /// Set the integral windup limit.
    pub const fn with_integral_limit(mut self, limit: f32) -> Self {
        self.integral_limit = limit;
        self
    }

    /// Compute control output (Nm).
    ///
    /// - `setpoint`: target position (rad).
    /// - `measured`: current position (rad).
    /// - `dt`: timestep (seconds), must be > 0.
    pub fn compute(&mut self, setpoint: f32, measured: f32, dt: f32) -> f32 {
        let error = setpoint - measured;

        // Integral term with anti-windup.
        self.integral += error * dt;
        self.integral = self
            .integral
            .clamp(-self.integral_limit, self.integral_limit);

        // Derivative term (of error).
        let derivative = if self.initialized {
            (error - self.last_error) / dt
        } else {
            self.initialized = true;
            0.0
        };
        self.last_error = error;

        let output = self
            .kd
            .mul_add(derivative, self.kp.mul_add(error, self.ki * self.integral));
        output.clamp(-self.output_limit, self.output_limit)
    }

    /// Reset integral and derivative state.
    pub const fn reset(&mut self) {
        self.integral = 0.0;
        self.last_error = 0.0;
        self.initialized = false;
    }

    /// Returns the accumulated integral term.
    pub const fn integral(&self) -> f32 {
        self.integral
    }

    /// Returns the proportional gain.
    pub const fn kp(&self) -> f32 {
        self.kp
    }

    /// Returns the integral gain.
    pub const fn ki(&self) -> f32 {
        self.ki
    }

    /// Returns the derivative gain.
    pub const fn kd(&self) -> f32 {
        self.kd
    }
}

// ---------------------------------------------------------------------------
// PdController
// ---------------------------------------------------------------------------

/// PD controller (no integral term).
///
/// Output: `kp × error + kd × d(error)/dt`.
///
/// Gains:
/// - `kp`: `Nm/rad` (position) or `Nm/(rad/s)` (velocity).
/// - `kd`: `Nm·s/rad`.
#[derive(Clone, Debug)]
pub struct PdController {
    /// Proportional gain.
    pub kp: f32,
    /// Derivative gain.
    pub kd: f32,
    last_error: f32,
    initialized: bool,
}

impl PdController {
    /// Create a new PD controller.
    pub const fn new(kp: f32, kd: f32) -> Self {
        Self {
            kp,
            kd,
            last_error: 0.0,
            initialized: false,
        }
    }

    /// Compute control output.
    ///
    /// - `setpoint`: target value (rad or rad/s).
    /// - `measured`: current value.
    /// - `dt`: timestep (seconds), must be > 0.
    pub fn compute(&mut self, setpoint: f32, measured: f32, dt: f32) -> f32 {
        let error = setpoint - measured;
        let derivative = if self.initialized {
            (error - self.last_error) / dt
        } else {
            self.initialized = true;
            0.0
        };
        self.last_error = error;
        self.kp.mul_add(error, self.kd * derivative)
    }

    /// Reset derivative state.
    pub const fn reset(&mut self) {
        self.last_error = 0.0;
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.001;

    // -- PidController --

    #[test]
    fn pid_proportional_only() {
        let mut pid = PidController::new(10.0, 0.0, 0.0);
        let out = pid.compute(1.0, 0.0, DT);
        assert!((out - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn pid_integral_accumulates() {
        let mut pid = PidController::new(0.0, 10.0, 0.0);
        // Error = 1.0, dt = 0.001 → integral += 0.001 → output = 10 * 0.001 = 0.01
        let out = pid.compute(1.0, 0.0, DT);
        assert!((out - 0.01).abs() < 1e-5);
        // Second step: integral = 0.002 → output = 0.02
        let out = pid.compute(1.0, 0.0, DT);
        assert!((out - 0.02).abs() < 1e-5);
    }

    #[test]
    fn pid_integral_windup_clamped() {
        let mut pid = PidController::new(0.0, 100.0, 0.0).with_integral_limit(1.0);
        for _ in 0..10_000 {
            pid.compute(1.0, 0.0, DT);
        }
        // Integral should be clamped at 1.0
        assert!((pid.integral() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn pid_derivative_first_step_is_zero() {
        let mut pid = PidController::new(0.0, 0.0, 10.0);
        let out = pid.compute(1.0, 0.0, DT);
        // First step: derivative = 0 (not initialized)
        assert!((out).abs() < f32::EPSILON);
    }

    #[test]
    fn pid_derivative_second_step() {
        let mut pid = PidController::new(0.0, 0.0, 1.0);
        pid.compute(1.0, 0.0, DT); // initialize
        // Error changes from 1.0 to 0.5 → derivative = -0.5/0.001 = -500
        let out = pid.compute(1.0, 0.5, DT);
        assert!((out - (-500.0)).abs() < 1e-3);
    }

    #[test]
    fn pid_output_clamped() {
        let mut pid = PidController::new(1000.0, 0.0, 0.0).with_output_limit(10.0);
        let out = pid.compute(1.0, 0.0, DT);
        assert!((out - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn pid_reset_clears_state() {
        let mut pid = PidController::new(10.0, 10.0, 10.0);
        pid.compute(1.0, 0.0, DT);
        pid.compute(1.0, 0.0, DT);
        pid.reset();
        assert!((pid.integral()).abs() < f32::EPSILON);
    }

    #[test]
    fn pid_getters() {
        let pid = PidController::new(1.0, 2.0, 3.0);
        assert!((pid.kp() - 1.0).abs() < f32::EPSILON);
        assert!((pid.ki() - 2.0).abs() < f32::EPSILON);
        assert!((pid.kd() - 3.0).abs() < f32::EPSILON);
    }

    // -- PdController --

    #[test]
    fn pd_proportional_only() {
        let mut pd = PdController::new(10.0, 0.0);
        let out = pd.compute(1.0, 0.0, DT);
        assert!((out - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn pd_derivative() {
        let mut pd = PdController::new(0.0, 1.0);
        pd.compute(1.0, 0.0, DT); // initialize
        let out = pd.compute(1.0, 0.5, DT);
        // Error: 1.0→0.5, derivative = -0.5/0.001 = -500
        assert!((out - (-500.0)).abs() < 1e-3);
    }

    #[test]
    fn pd_reset() {
        let mut pd = PdController::new(10.0, 5.0);
        pd.compute(1.0, 0.0, DT);
        pd.reset();
        assert!(!pd.initialized);
    }

    // -- ControlMode --

    #[test]
    fn control_mode_default_is_torque() {
        assert!(matches!(ControlMode::default(), ControlMode::Torque));
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn control_types_are_send_sync() {
        assert_send_sync::<ControlMode>();
        assert_send_sync::<PidController>();
        assert_send_sync::<PdController>();
    }
}
