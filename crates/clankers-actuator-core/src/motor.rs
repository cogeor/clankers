//! Motor models for robotics simulation.
//!
//! Three fidelity levels are provided:
//! - [`IdealMotor`]: stateless, instantaneous response, torque-speed limit only.
//! - [`DcMotor`]: first-order dynamics with torque-speed curve.
//! - [`FullDcMotor`]: complete electrical model (armature circuit).
//!
//! [`MotorType`] is a dispatch enum wrapping all three.

// ---------------------------------------------------------------------------
// IdealMotor
// ---------------------------------------------------------------------------

/// Ideal motor — instantaneous response with linear torque-speed saturation.
///
/// Stateless: the `dt` parameter on [`MotorModel::step`] is unused.
///
/// Torque-speed curve: `available = max_torque * (1 - |velocity| / max_velocity)`
#[derive(Clone, Debug, Default)]
pub struct IdealMotor {
    /// Maximum torque output (Nm).
    pub max_torque: f32,
    /// Maximum velocity (rad/s).
    pub max_velocity: f32,
}

impl IdealMotor {
    /// Create a new ideal motor with the given limits.
    pub const fn new(max_torque: f32, max_velocity: f32) -> Self {
        Self {
            max_torque,
            max_velocity,
        }
    }

    /// Compute output torque given a torque command and current velocity.
    ///
    /// Uses linear torque-speed saturation:
    /// `available = max_torque * (1 - |velocity| / max_velocity)`
    pub fn compute(&self, torque_cmd: f32, velocity: f32) -> f32 {
        let speed_ratio = (velocity.abs() / self.max_velocity).min(1.0);
        let available = self.max_torque * (1.0 - speed_ratio);
        torque_cmd.clamp(-available, available)
    }
}

// ---------------------------------------------------------------------------
// DcMotor
// ---------------------------------------------------------------------------

/// DC motor with first-order dynamics and torque-speed curve.
///
/// Physics:
/// - Torque-speed saturation: `T_max(ω) = stall_torque * (1 - |ω| / no_load_speed)`
/// - First-order response: `τ·dT/dt + T = T_target`
/// - Discretization: `α = dt / (τ + dt)`, `state += α * (target - state)`
///
/// For stability, `dt << time_constant` is recommended.
#[derive(Clone, Debug)]
pub struct DcMotor {
    /// Stall torque at zero velocity (Nm).
    pub stall_torque: f32,
    /// No-load speed (rad/s).
    pub no_load_speed: f32,
    /// Time constant for first-order filter (seconds). `0.0` means instant.
    pub time_constant: f32,
    /// Maximum torque limit (Nm), capped at `stall_torque`.
    pub max_torque: f32,
    /// Maximum velocity limit (rad/s).
    pub max_velocity: f32,
    // Internal state: filtered torque (Nm).
    torque_state: f32,
}

impl DcMotor {
    /// Create a new DC motor with stall torque and no-load speed.
    pub const fn new(stall_torque: f32, no_load_speed: f32) -> Self {
        Self {
            stall_torque,
            no_load_speed,
            time_constant: 0.0,
            max_torque: stall_torque,
            max_velocity: no_load_speed,
            torque_state: 0.0,
        }
    }

    /// Set the first-order time constant (seconds).
    pub const fn with_time_constant(mut self, tau: f32) -> Self {
        self.time_constant = tau;
        self
    }

    /// Limit maximum torque output.  Clamped to `stall_torque`.
    pub const fn with_torque_limit(mut self, limit: f32) -> Self {
        self.max_torque = limit.min(self.stall_torque);
        self
    }

    /// Torque available at given velocity (torque-speed curve).
    fn torque_at_speed(&self, velocity: f32) -> f32 {
        let speed_ratio = velocity.abs() / self.no_load_speed;
        let curve_limit = self.stall_torque * (1.0 - speed_ratio).max(0.0);
        curve_limit.min(self.max_torque)
    }

    /// Step motor dynamics by `dt` seconds.  Returns output torque (Nm).
    pub fn step(&mut self, torque_cmd: f32, velocity: f32, dt: f32) -> f32 {
        let available = self.torque_at_speed(velocity);
        let target = torque_cmd.clamp(-available, available);

        if self.time_constant > 0.0 {
            let alpha = dt / (self.time_constant + dt);
            self.torque_state += alpha * (target - self.torque_state);
        } else {
            self.torque_state = target;
        }

        self.torque_state
    }

    /// Reset internal state to zero.
    pub const fn reset(&mut self) {
        self.torque_state = 0.0;
    }

    /// Returns the current filtered torque state (Nm).
    pub const fn torque_state(&self) -> f32 {
        self.torque_state
    }
}

// ---------------------------------------------------------------------------
// FullDcMotor
// ---------------------------------------------------------------------------

/// Full DC motor with electrical dynamics (armature circuit).
///
/// Electrical: `L · di/dt = V - R·i - Ke·ω`
/// Torque: `T = Kt · i`
///
/// Derived steady-state quantities:
/// - Steady-state torque: `T_ss = Kt · (V - Ke·ω) / R`
/// - No-load speed: `ω_nl = V / Ke`
/// - Stall current: `I_stall = V / R`
#[derive(Clone, Debug)]
pub struct FullDcMotor {
    /// Armature resistance (Ohms).
    pub resistance: f32,
    /// Armature inductance (Henries).
    pub inductance: f32,
    /// Back-EMF constant (V/(rad/s)).
    pub back_emf_constant: f32,
    /// Torque constant (Nm/A).
    pub torque_constant: f32,
    /// Maximum voltage (V).
    pub max_voltage: f32,
    /// Maximum current (A).
    pub max_current: f32,
    // Current state (A).
    current: f32,
}

impl FullDcMotor {
    /// Create from electrical specifications.
    ///
    /// For an ideal DC motor, `Ke = Kt` (back-EMF constant equals torque constant).
    pub const fn from_specs(
        resistance: f32,
        inductance: f32,
        kt: f32,
        max_voltage: f32,
        max_current: f32,
    ) -> Self {
        Self {
            resistance,
            inductance,
            back_emf_constant: kt,
            torque_constant: kt,
            max_voltage,
            max_current,
            current: 0.0,
        }
    }

    /// Step electrical dynamics by `dt` seconds.  Returns output torque (Nm).
    pub fn step(&mut self, voltage_cmd: f32, omega: f32, dt: f32) -> f32 {
        let voltage = voltage_cmd.clamp(-self.max_voltage, self.max_voltage);
        let back_emf = self.back_emf_constant * omega;

        if self.inductance > 1e-6 {
            let di_dt =
                (self.resistance.mul_add(-self.current, voltage) - back_emf) / self.inductance;
            self.current += di_dt * dt;
        } else {
            // Resistive approximation (L ≈ 0).
            self.current = (voltage - back_emf) / self.resistance;
        }

        self.current = self.current.clamp(-self.max_current, self.max_current);
        self.torque_constant * self.current
    }

    /// Returns the current motor current (A).
    pub const fn current(&self) -> f32 {
        self.current
    }

    /// Reset internal state to zero.
    pub const fn reset(&mut self) {
        self.current = 0.0;
    }
}

// ---------------------------------------------------------------------------
// MotorType enum (static dispatch)
// ---------------------------------------------------------------------------

/// Dispatch enum for motor types.
///
/// Prefer this over `dyn MotorModel` when all variants are known at compile
/// time.  Stack-allocated, no vtable overhead.
#[derive(Clone, Debug)]
pub enum MotorType {
    /// Ideal (stateless, instant response).
    Ideal(IdealMotor),
    /// DC motor with first-order dynamics.
    Dc(DcMotor),
    /// Full DC motor with electrical model.
    FullDc(FullDcMotor),
}

impl MotorType {
    /// Step the motor model.  Returns output torque (Nm).
    pub fn step(&mut self, command: f32, velocity: f32, dt: f32) -> f32 {
        match self {
            Self::Ideal(m) => m.compute(command, velocity),
            Self::Dc(m) => m.step(command, velocity, dt),
            Self::FullDc(m) => m.step(command, velocity, dt),
        }
    }

    /// Reset internal state.
    pub const fn reset(&mut self) {
        match self {
            Self::Ideal(_) => {}
            Self::Dc(m) => m.reset(),
            Self::FullDc(m) => m.reset(),
        }
    }
}

impl Default for MotorType {
    fn default() -> Self {
        Self::Ideal(IdealMotor::new(100.0, 10.0))
    }
}

// ---------------------------------------------------------------------------
// MotorModel trait (dynamic dispatch)
// ---------------------------------------------------------------------------

/// Trait for motor models (dynamic dispatch).
///
/// Use when the set of motor variants is open-ended or when trait objects are
/// needed.  For the common case, prefer [`MotorType`].
pub trait MotorModel: Send + Sync {
    /// Step motor dynamics by `dt` seconds.  Returns output torque (Nm).
    fn step(&mut self, command: f32, velocity: f32, dt: f32) -> f32;

    /// Reset internal state to zero.
    fn reset(&mut self);
}

impl MotorModel for IdealMotor {
    fn step(&mut self, command: f32, velocity: f32, _dt: f32) -> f32 {
        self.compute(command, velocity)
    }

    fn reset(&mut self) {}
}

impl MotorModel for DcMotor {
    fn step(&mut self, command: f32, velocity: f32, dt: f32) -> f32 {
        Self::step(self, command, velocity, dt)
    }

    fn reset(&mut self) {
        Self::reset(self);
    }
}

impl MotorModel for FullDcMotor {
    fn step(&mut self, command: f32, velocity: f32, dt: f32) -> f32 {
        Self::step(self, command, velocity, dt)
    }

    fn reset(&mut self) {
        Self::reset(self);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.001;

    // -- IdealMotor --

    #[test]
    fn ideal_motor_passes_within_limits() {
        let m = IdealMotor::new(10.0, 5.0);
        let out = m.compute(5.0, 0.0);
        assert!((out - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ideal_motor_clamps_at_max_torque() {
        let m = IdealMotor::new(10.0, 5.0);
        let out = m.compute(20.0, 0.0);
        assert!((out - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ideal_motor_reduces_at_speed() {
        let m = IdealMotor::new(10.0, 10.0);
        // At half speed: available = 10 * (1 - 0.5) = 5.0
        let out = m.compute(10.0, 5.0);
        assert!((out - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn ideal_motor_zero_at_max_speed() {
        let m = IdealMotor::new(10.0, 5.0);
        let out = m.compute(10.0, 5.0);
        assert!((out).abs() < f32::EPSILON);
    }

    #[test]
    fn ideal_motor_negative_command() {
        let m = IdealMotor::new(10.0, 5.0);
        let out = m.compute(-5.0, 0.0);
        assert!((out - (-5.0)).abs() < f32::EPSILON);
    }

    // -- DcMotor --

    #[test]
    fn dc_motor_instant_without_time_constant() {
        let mut m = DcMotor::new(10.0, 5.0);
        let out = m.step(5.0, 0.0, DT);
        assert!((out - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn dc_motor_first_order_filter() {
        let mut m = DcMotor::new(10.0, 5.0).with_time_constant(0.1);
        // First step: alpha = 0.001 / (0.1 + 0.001) ≈ 0.0099
        // state = 0 + 0.0099 * (5.0 - 0) ≈ 0.0495
        let out = m.step(5.0, 0.0, DT);
        assert!(out > 0.0);
        assert!(out < 5.0);
    }

    #[test]
    fn dc_motor_converges_with_time_constant() {
        let mut m = DcMotor::new(10.0, 5.0).with_time_constant(0.01);
        // Run many steps to converge
        let mut out = 0.0;
        for _ in 0..10_000 {
            out = m.step(5.0, 0.0, DT);
        }
        assert!((out - 5.0).abs() < 0.01);
    }

    #[test]
    fn dc_motor_torque_speed_saturation() {
        let mut m = DcMotor::new(10.0, 10.0);
        // At full speed, available torque = 0
        let out = m.step(10.0, 10.0, DT);
        assert!((out).abs() < f32::EPSILON);
    }

    #[test]
    fn dc_motor_torque_limit() {
        let mut m = DcMotor::new(100.0, 5.0).with_torque_limit(10.0);
        let out = m.step(50.0, 0.0, DT);
        assert!((out - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn dc_motor_reset() {
        let mut m = DcMotor::new(10.0, 5.0).with_time_constant(0.1);
        m.step(10.0, 0.0, DT);
        assert!(m.torque_state().abs() > f32::EPSILON);
        m.reset();
        assert!(m.torque_state().abs() < f32::EPSILON);
    }

    // -- FullDcMotor --

    #[test]
    fn full_dc_motor_produces_torque() {
        let mut m = FullDcMotor::from_specs(1.0, 0.001, 0.1, 12.0, 10.0);
        let out = m.step(12.0, 0.0, DT);
        assert!(out > 0.0);
    }

    #[test]
    fn full_dc_motor_back_emf_reduces_current() {
        let mut m = FullDcMotor::from_specs(1.0, 0.0, 0.1, 12.0, 10.0);
        // At no-load speed ω = V/Ke = 12/0.1 = 120 rad/s → current = 0
        let out = m.step(12.0, 120.0, DT);
        assert!(out.abs() < f32::EPSILON);
    }

    #[test]
    fn full_dc_motor_current_clamp() {
        let mut m = FullDcMotor::from_specs(0.01, 0.0, 0.1, 100.0, 5.0);
        // V/R = 100/0.01 = 10000 A, but max_current = 5
        m.step(100.0, 0.0, DT);
        assert!((m.current() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn full_dc_motor_reset() {
        let mut m = FullDcMotor::from_specs(1.0, 0.001, 0.1, 12.0, 10.0);
        m.step(12.0, 0.0, DT);
        assert!(m.current().abs() > f32::EPSILON);
        m.reset();
        assert!(m.current().abs() < f32::EPSILON);
    }

    #[test]
    fn full_dc_motor_resistive_approximation() {
        // inductance ~= 0 → steady-state: i = (V - Ke*ω) / R
        let mut m = FullDcMotor::from_specs(2.0, 0.0, 0.1, 12.0, 100.0);
        m.step(12.0, 0.0, DT);
        // Expected: i = (12 - 0) / 2 = 6 A, T = 0.1 * 6 = 0.6 Nm
        assert!((m.current() - 6.0).abs() < f32::EPSILON);
    }

    // -- MotorType enum --

    #[test]
    fn motor_type_default_is_ideal() {
        let m = MotorType::default();
        assert!(matches!(m, MotorType::Ideal(_)));
    }

    #[test]
    fn motor_type_step_dispatches() {
        let mut m = MotorType::Dc(DcMotor::new(10.0, 5.0));
        let out = m.step(5.0, 0.0, DT);
        assert!((out - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn motor_type_reset_dispatches() {
        let mut m = MotorType::Dc(DcMotor::new(10.0, 5.0).with_time_constant(0.1));
        m.step(10.0, 0.0, DT);
        m.reset();
        if let MotorType::Dc(dc) = &m {
            assert!(dc.torque_state().abs() < f32::EPSILON);
        }
    }

    // -- MotorModel trait --

    #[test]
    fn motor_model_trait_for_ideal() {
        let mut m = IdealMotor::new(10.0, 5.0);
        let out = MotorModel::step(&mut m, 5.0, 0.0, DT);
        assert!((out - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn motor_model_trait_for_dc() {
        let mut m = DcMotor::new(10.0, 5.0);
        let out = MotorModel::step(&mut m, 5.0, 0.0, DT);
        assert!((out - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn motor_model_trait_for_full_dc() {
        let mut m = FullDcMotor::from_specs(1.0, 0.001, 0.1, 12.0, 10.0);
        let out = MotorModel::step(&mut m, 12.0, 0.0, DT);
        assert!(out > 0.0);
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn motor_types_are_send_sync() {
        assert_send_sync::<IdealMotor>();
        assert_send_sync::<DcMotor>();
        assert_send_sync::<FullDcMotor>();
        assert_send_sync::<MotorType>();
    }
}
