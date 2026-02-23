use std::fmt;
use std::ops::{Add, AddAssign, Sub};
use std::time::Duration;

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SimTime
// ---------------------------------------------------------------------------

/// Integer-nanosecond simulation clock.
///
/// Avoids floating-point accumulation errors by tracking elapsed time as a
/// monotonically increasing `u64` nanosecond count.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    Serialize,
    Deserialize,
    Resource,
)]
pub struct SimTime {
    nanos: u64,
}

impl SimTime {
    /// Create a new `SimTime` at zero.
    #[must_use]
    pub const fn new() -> Self {
        Self { nanos: 0 }
    }

    /// Create a `SimTime` from a raw nanosecond count.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }

    /// Create a `SimTime` from seconds (as `f64`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn from_secs(secs: f64) -> Self {
        Self {
            nanos: (secs * 1_000_000_000.0) as u64,
        }
    }

    /// Create a `SimTime` from a [`Duration`].
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn from_duration(duration: Duration) -> Self {
        Self {
            nanos: duration.as_nanos() as u64,
        }
    }

    /// Raw nanosecond count.
    #[must_use]
    pub const fn nanos(&self) -> u64 {
        self.nanos
    }

    /// Elapsed microseconds (truncated).
    #[must_use]
    pub const fn micros(&self) -> u64 {
        self.nanos / 1_000
    }

    /// Elapsed milliseconds (truncated).
    #[must_use]
    pub const fn millis(&self) -> u64 {
        self.nanos / 1_000_000
    }

    /// Elapsed whole seconds (truncated).
    #[must_use]
    pub const fn secs(&self) -> u64 {
        self.nanos / 1_000_000_000
    }

    /// Elapsed seconds as `f64`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn secs_f64(&self) -> f64 {
        self.nanos as f64 / 1_000_000_000.0
    }

    /// Elapsed seconds as `f32`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn secs_f32(&self) -> f32 {
        self.nanos as f32 / 1_000_000_000.0
    }

    /// Convert to a standard [`Duration`].
    #[must_use]
    pub const fn to_duration(&self) -> Duration {
        Duration::from_nanos(self.nanos)
    }

    /// Advance the clock by `delta_nanos` nanoseconds.
    pub const fn advance(&mut self, delta_nanos: u64) {
        self.nanos = self.nanos.saturating_add(delta_nanos);
    }

    /// Advance the clock by `delta_secs` seconds.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn advance_secs(&mut self, delta_secs: f64) {
        let delta_nanos = (delta_secs * 1_000_000_000.0) as u64;
        self.advance(delta_nanos);
    }

    /// Advance the clock by a [`Duration`].
    #[allow(clippy::cast_possible_truncation)]
    pub const fn advance_duration(&mut self, duration: Duration) {
        self.advance(duration.as_nanos() as u64);
    }

    /// Reset the clock to zero.
    pub const fn reset(&mut self) {
        self.nanos = 0;
    }

    /// Number of complete steps of `dt_secs` that fit in the current time.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn step_count(&self, dt_secs: f64) -> u64 {
        let dt_nanos = (dt_secs * 1_000_000_000.0) as u64;
        if dt_nanos == 0 {
            return 0;
        }
        self.nanos / dt_nanos
    }

    /// Nanoseconds elapsed since `earlier`. Returns zero if `earlier` is ahead.
    #[must_use]
    pub const fn elapsed_since(&self, earlier: Self) -> Duration {
        Duration::from_nanos(self.nanos.saturating_sub(earlier.nanos))
    }
}

// -- Operator impls --

impl Add<Duration> for SimTime {
    type Output = Self;

    #[allow(clippy::cast_possible_truncation)]
    fn add(self, rhs: Duration) -> Self {
        Self {
            nanos: self.nanos.saturating_add(rhs.as_nanos() as u64),
        }
    }
}

impl AddAssign<Duration> for SimTime {
    #[allow(clippy::cast_possible_truncation)]
    fn add_assign(&mut self, rhs: Duration) {
        self.nanos = self.nanos.saturating_add(rhs.as_nanos() as u64);
    }
}

impl Sub for SimTime {
    type Output = Duration;

    /// Subtract two `SimTime` values, yielding a [`Duration`].
    /// Uses saturating subtraction to prevent underflow.
    fn sub(self, rhs: Self) -> Duration {
        Duration::from_nanos(self.nanos.saturating_sub(rhs.nanos))
    }
}

impl fmt::Display for SimTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_secs = self.nanos / 1_000_000_000;
        let remaining_nanos = self.nanos % 1_000_000_000;
        let millis = remaining_nanos / 1_000_000;
        let micros = (remaining_nanos % 1_000_000) / 1_000;
        write!(f, "{total_secs}.{millis:03}{micros:03}s")
    }
}

// ---------------------------------------------------------------------------
// Accumulator
// ---------------------------------------------------------------------------

/// Fixed-timestep accumulator implementing the "fix your timestep" pattern.
///
/// Accumulates real-world delta time and dispenses fixed-size simulation steps.
/// Caps the number of steps per frame to prevent the "spiral of death".
#[derive(Debug, Clone)]
pub struct Accumulator {
    accumulated: u64,
    timestep_nanos: u64,
    timestep_secs: f64,
    max_steps: u32,
    steps_this_frame: u32,
}

impl Accumulator {
    /// Create a new accumulator with the given fixed timestep in seconds.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn new(timestep_secs: f64) -> Self {
        let timestep_nanos = (timestep_secs * 1_000_000_000.0) as u64;
        Self {
            accumulated: 0,
            timestep_nanos,
            timestep_secs,
            max_steps: 10,
            steps_this_frame: 0,
        }
    }

    /// Set the maximum number of steps allowed per frame.
    #[must_use]
    pub const fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Feed a real-world delta into the accumulator and reset the per-frame
    /// step counter.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn accumulate(&mut self, delta: Duration) {
        self.accumulated = self.accumulated.saturating_add(delta.as_nanos() as u64);
        self.steps_this_frame = 0;
    }

    /// Returns `true` if at least one timestep worth of time is accumulated
    /// and the per-frame step cap has not been reached.
    ///
    /// Each call that returns `true` consumes one timestep from the
    /// accumulator and increments the step counter.
    pub const fn should_step(&mut self) -> bool {
        if self.steps_this_frame >= self.max_steps {
            return false;
        }
        if self.accumulated >= self.timestep_nanos {
            self.accumulated -= self.timestep_nanos;
            self.steps_this_frame += 1;
            return true;
        }
        false
    }

    /// Interpolation alpha in `[0, 1)` representing how far into the next
    /// timestep the accumulator has progressed. Useful for visual
    /// interpolation between physics states.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn alpha(&self) -> f32 {
        if self.timestep_nanos == 0 {
            return 0.0;
        }
        self.accumulated as f32 / self.timestep_nanos as f32
    }

    /// The fixed timestep in seconds.
    #[must_use]
    pub const fn timestep(&self) -> f64 {
        self.timestep_secs
    }

    /// Currently accumulated (unconsumed) time in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn accumulated_secs(&self) -> f64 {
        self.accumulated as f64 / 1_000_000_000.0
    }

    /// Reset accumulated time and step counter to zero.
    pub const fn reset(&mut self) {
        self.accumulated = 0;
        self.steps_this_frame = 0;
    }
}

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------

/// High-level simulation clock combining [`SimTime`] with an [`Accumulator`].
///
/// Typical usage:
/// ```ignore
/// clock.tick(frame_delta);
/// while clock.should_step() {
///     clock.advance();
///     // run fixed-step simulation logic using clock.time()
/// }
/// let alpha = clock.alpha(); // for visual interpolation
/// ```
#[derive(Debug, Clone)]
pub struct Clock {
    time: SimTime,
    accumulator: Accumulator,
}

impl Clock {
    /// Create a new clock with the given fixed timestep in seconds.
    pub fn new(timestep_secs: f64) -> Self {
        Self {
            time: SimTime::new(),
            accumulator: Accumulator::new(timestep_secs),
        }
    }

    /// Set the maximum number of simulation steps per tick.
    #[must_use]
    pub const fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.accumulator = self.accumulator.with_max_steps(max_steps);
        self
    }

    /// Feed a real-world frame delta into the accumulator.
    pub const fn tick(&mut self, delta: Duration) {
        self.accumulator.accumulate(delta);
    }

    /// Returns `true` if a simulation step should be taken.
    ///
    /// Call in a loop after [`tick`](Self::tick). Each `true` result means
    /// the caller should run one fixed-step update, then call
    /// [`advance`](Self::advance).
    pub const fn should_step(&mut self) -> bool {
        self.accumulator.should_step()
    }

    /// Advance the simulation time by one timestep.
    pub const fn advance(&mut self) {
        self.time.advance(self.accumulator.timestep_nanos);
    }

    /// Current simulation time.
    #[must_use]
    pub const fn time(&self) -> SimTime {
        self.time
    }

    /// The fixed timestep in seconds.
    #[must_use]
    pub const fn timestep(&self) -> f64 {
        self.accumulator.timestep()
    }

    /// Interpolation alpha for visual smoothing.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.accumulator.alpha()
    }

    /// Reset both the simulation time and the accumulator.
    pub const fn reset(&mut self) {
        self.time.reset();
        self.accumulator.reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SimTime: construction ----

    #[test]
    fn simtime_new() {
        let t = SimTime::new();
        assert_eq!(t.nanos(), 0);
    }

    #[test]
    fn simtime_from_nanos() {
        let t = SimTime::from_nanos(1_500_000_000);
        assert_eq!(t.nanos(), 1_500_000_000);
    }

    #[test]
    fn simtime_from_secs() {
        let t = SimTime::from_secs(2.5);
        assert_eq!(t.nanos(), 2_500_000_000);
    }

    #[test]
    fn simtime_from_duration() {
        let t = SimTime::from_duration(Duration::from_millis(1500));
        assert_eq!(t.nanos(), 1_500_000_000);
    }

    // ---- SimTime: accessors ----

    #[test]
    fn simtime_nanos() {
        let t = SimTime::from_nanos(123_456_789);
        assert_eq!(t.nanos(), 123_456_789);
    }

    #[test]
    fn simtime_micros() {
        let t = SimTime::from_nanos(123_456_789);
        assert_eq!(t.micros(), 123_456);
    }

    #[test]
    fn simtime_millis() {
        let t = SimTime::from_nanos(123_456_789);
        assert_eq!(t.millis(), 123);
    }

    #[test]
    fn simtime_secs() {
        let t = SimTime::from_nanos(2_500_000_000);
        assert_eq!(t.secs(), 2);
    }

    #[test]
    fn simtime_secs_f64() {
        let t = SimTime::from_nanos(1_500_000_000);
        assert!((t.secs_f64() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn simtime_secs_f32() {
        let t = SimTime::from_nanos(1_500_000_000);
        assert!((t.secs_f32() - 1.5).abs() < 1e-4);
    }

    // ---- SimTime: advance and reset ----

    #[test]
    fn simtime_advance_nanos() {
        let mut t = SimTime::new();
        t.advance(1_000_000);
        assert_eq!(t.nanos(), 1_000_000);
        t.advance(2_000_000);
        assert_eq!(t.nanos(), 3_000_000);
    }

    #[test]
    fn simtime_advance_secs() {
        let mut t = SimTime::new();
        t.advance_secs(0.5);
        assert_eq!(t.nanos(), 500_000_000);
    }

    #[test]
    fn simtime_advance_duration() {
        let mut t = SimTime::new();
        t.advance_duration(Duration::from_millis(100));
        assert_eq!(t.nanos(), 100_000_000);
    }

    #[test]
    fn simtime_reset() {
        let mut t = SimTime::from_secs(5.0);
        assert!(t.nanos() > 0);
        t.reset();
        assert_eq!(t.nanos(), 0);
    }

    // ---- SimTime: arithmetic ----

    #[test]
    fn simtime_add_duration() {
        let t = SimTime::from_secs(1.0);
        let result = t + Duration::from_secs(2);
        assert_eq!(result.nanos(), 3_000_000_000);
    }

    #[test]
    fn simtime_add_assign_duration() {
        let mut t = SimTime::from_secs(1.0);
        t += Duration::from_millis(500);
        assert_eq!(t.nanos(), 1_500_000_000);
    }

    #[test]
    fn simtime_sub_yields_duration() {
        let a = SimTime::from_secs(3.0);
        let b = SimTime::from_secs(1.0);
        let d = a - b;
        assert_eq!(d, Duration::from_secs(2));
    }

    #[test]
    fn simtime_sub_saturates() {
        let a = SimTime::from_secs(1.0);
        let b = SimTime::from_secs(5.0);
        let d = a - b;
        assert_eq!(d, Duration::ZERO);
    }

    // ---- SimTime: step_count, elapsed_since ----

    #[test]
    fn simtime_step_count() {
        let t = SimTime::from_secs(1.0);
        // 1 second / 0.01s per step = 100 steps
        assert_eq!(t.step_count(0.01), 100);
    }

    #[test]
    fn simtime_step_count_zero_dt() {
        let t = SimTime::from_secs(1.0);
        assert_eq!(t.step_count(0.0), 0);
    }

    #[test]
    fn simtime_elapsed_since() {
        let a = SimTime::from_secs(5.0);
        let b = SimTime::from_secs(2.0);
        assert_eq!(a.elapsed_since(b), Duration::from_secs(3));
    }

    #[test]
    fn simtime_elapsed_since_saturates() {
        let a = SimTime::from_secs(1.0);
        let b = SimTime::from_secs(5.0);
        assert_eq!(a.elapsed_since(b), Duration::ZERO);
    }

    // ---- SimTime: Display ----

    #[test]
    fn simtime_display() {
        let t = SimTime::from_nanos(1_234_567_890);
        let s = format!("{t}");
        assert_eq!(s, "1.234567s");
    }

    #[test]
    fn simtime_display_zero() {
        let t = SimTime::new();
        let s = format!("{t}");
        assert_eq!(s, "0.000000s");
    }

    // ---- SimTime: ordering ----

    #[test]
    fn simtime_ordering() {
        let a = SimTime::from_secs(1.0);
        let b = SimTime::from_secs(2.0);
        let c = SimTime::from_secs(1.0);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, c);
        assert!(a <= c);
        assert!(a >= c);
    }

    // ---- Accumulator: construction and basic stepping ----

    #[test]
    fn accumulator_new() {
        let acc = Accumulator::new(1.0 / 60.0);
        assert!((acc.timestep() - 1.0 / 60.0).abs() < 1e-12);
        assert!((acc.accumulated_secs() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn accumulator_basic_stepping() {
        let mut acc = Accumulator::new(1.0 / 60.0);
        // Feed exactly one timestep
        let dt = Duration::from_secs_f64(1.0 / 60.0);
        acc.accumulate(dt);
        assert!(acc.should_step());
        assert!(!acc.should_step()); // only one step available
    }

    #[test]
    fn accumulator_multiple_steps() {
        let mut acc = Accumulator::new(0.01); // 10ms timestep
        acc.accumulate(Duration::from_millis(35)); // 3.5 steps worth
        let mut count = 0;
        while acc.should_step() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    // ---- Accumulator: max_steps spiral of death prevention ----

    #[test]
    fn accumulator_max_steps() {
        let mut acc = Accumulator::new(0.01).with_max_steps(3);
        acc.accumulate(Duration::from_millis(100)); // 10 steps worth
        let mut count = 0;
        while acc.should_step() {
            count += 1;
        }
        assert_eq!(count, 3); // capped at max_steps
    }

    #[test]
    fn accumulator_default_max_steps() {
        let mut acc = Accumulator::new(0.001); // 1ms timestep
        acc.accumulate(Duration::from_millis(50)); // 50 steps worth
        let mut count = 0;
        while acc.should_step() {
            count += 1;
        }
        assert_eq!(count, 10); // default max_steps = 10
    }

    // ---- Accumulator: alpha interpolation ----

    #[test]
    fn accumulator_alpha_zero() {
        let mut acc = Accumulator::new(0.01);
        // Feed exactly one step, consume it
        acc.accumulate(Duration::from_millis(10));
        assert!(acc.should_step());
        assert!(!acc.should_step());
        // Nothing left over
        assert!(acc.alpha().abs() < 1e-4);
    }

    #[test]
    fn accumulator_alpha_nonzero() {
        let mut acc = Accumulator::new(0.01); // 10ms
        acc.accumulate(Duration::from_millis(15)); // 1.5 steps
        assert!(acc.should_step());
        assert!(!acc.should_step());
        // 5ms left over out of 10ms step -> alpha ~ 0.5
        assert!((acc.alpha() - 0.5).abs() < 0.01);
    }

    // ---- Accumulator: reset ----

    #[test]
    fn accumulator_reset() {
        let mut acc = Accumulator::new(0.01);
        acc.accumulate(Duration::from_millis(50));
        assert!(acc.should_step());
        acc.reset();
        assert!(!acc.should_step());
        assert!((acc.accumulated_secs() - 0.0).abs() < 1e-12);
    }

    // ---- Clock: tick/should_step/advance cycle ----

    #[test]
    fn clock_tick_step_advance() {
        let mut clock = Clock::new(0.01); // 10ms
        clock.tick(Duration::from_millis(25)); // 2.5 steps worth
        let mut steps = 0;
        while clock.should_step() {
            clock.advance();
            steps += 1;
        }
        assert_eq!(steps, 2);
    }

    // ---- Clock: time progression ----

    #[test]
    fn clock_time_progression() {
        let mut clock = Clock::new(0.01); // 10ms
        assert_eq!(clock.time().nanos(), 0);
        clock.tick(Duration::from_millis(25));
        while clock.should_step() {
            clock.advance();
        }
        // 2 steps of 10ms = 20ms
        assert_eq!(clock.time().millis(), 20);
    }

    #[test]
    fn clock_timestep() {
        let clock = Clock::new(1.0 / 120.0);
        assert!((clock.timestep() - 1.0 / 120.0).abs() < 1e-12);
    }

    #[test]
    fn clock_alpha() {
        let mut clock = Clock::new(0.01); // 10ms
        clock.tick(Duration::from_millis(15));
        while clock.should_step() {
            clock.advance();
        }
        // 5ms left over / 10ms step -> alpha ~ 0.5
        assert!((clock.alpha() - 0.5).abs() < 0.01);
    }

    // ---- Clock: reset ----

    #[test]
    fn clock_reset() {
        let mut clock = Clock::new(0.01);
        clock.tick(Duration::from_millis(50));
        while clock.should_step() {
            clock.advance();
        }
        assert!(clock.time().nanos() > 0);
        clock.reset();
        assert_eq!(clock.time().nanos(), 0);
        assert!(!clock.should_step());
    }

    // ---- Clock: with_max_steps ----

    #[test]
    fn clock_with_max_steps() {
        let mut clock = Clock::new(0.01).with_max_steps(2);
        clock.tick(Duration::from_millis(100)); // 10 steps worth
        let mut steps = 0;
        while clock.should_step() {
            clock.advance();
            steps += 1;
        }
        assert_eq!(steps, 2);
    }
}
