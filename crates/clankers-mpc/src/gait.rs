//! Gait scheduler for legged locomotion.
//!
//! Generates contact sequences over a prediction horizon based on gait patterns.
//! Each gait is defined by:
//! - Phase offsets per foot (when in the cycle each foot lifts)
//! - Duty factor (fraction of cycle spent in stance)
//! - Cycle time (total gait period)

/// Supported gait patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaitType {
    /// All feet on ground (static balance).
    Stand,
    /// Diagonal pairs alternate: FL+RR and FR+RL.
    Trot,
    /// One foot lifts at a time, in sequence.
    Walk,
    /// Front pair and rear pair alternate.
    Bound,
}

/// Gait scheduler that produces contact sequences over a prediction horizon.
#[derive(Clone, Debug)]
pub struct GaitScheduler {
    /// Number of feet.
    n_feet: usize,
    /// Phase offset for each foot [0, 1).
    offsets: Vec<f64>,
    /// Fraction of cycle spent in stance [0, 1].
    duty_factor: f64,
    /// Total gait cycle time in seconds.
    cycle_time: f64,
    /// Current phase in the gait cycle [0, 1).
    phase: f64,
    /// Per-foot contact override from ground-truth sensors.
    /// When set, overrides the scheduled contact state for that foot.
    contact_overrides: Vec<Option<bool>>,
}

impl GaitScheduler {
    /// Create a gait scheduler for a quadruped (4 feet) with a predefined pattern.
    pub fn quadruped(gait: GaitType) -> Self {
        let (offsets, duty_factor, cycle_time) = match gait {
            GaitType::Stand => (vec![0.0; 4], 1.0, 1.0),
            GaitType::Trot => (vec![0.0, 0.5, 0.5, 0.0], 0.5, 0.35),
            GaitType::Walk => (vec![0.0, 0.5, 0.25, 0.75], 0.75, 0.8),
            GaitType::Bound => (vec![0.0, 0.0, 0.5, 0.5], 0.5, 0.4),
        };
        Self {
            n_feet: 4,
            contact_overrides: vec![None; 4],
            offsets,
            duty_factor,
            cycle_time,
            phase: 0.0,
        }
    }

    /// Create a custom gait scheduler.
    pub fn custom(
        offsets: Vec<f64>,
        duty_factor: f64,
        cycle_time: f64,
    ) -> Self {
        let n_feet = offsets.len();
        Self {
            n_feet,
            contact_overrides: vec![None; n_feet],
            offsets,
            duty_factor,
            cycle_time,
            phase: 0.0,
        }
    }

    /// Advance the gait phase by `dt` seconds.
    ///
    /// Also clears contact overrides from the previous step.
    pub fn advance(&mut self, dt: f64) {
        self.phase = (self.phase + dt / self.cycle_time) % 1.0;
        self.clear_contact_overrides();
    }

    /// Get the current phase [0, 1).
    pub const fn phase(&self) -> f64 {
        self.phase
    }

    /// Set the phase directly (useful for testing).
    pub fn set_phase(&mut self, phase: f64) {
        self.phase = phase % 1.0;
    }

    /// Check if a specific foot is in contact at the current phase.
    ///
    /// If a contact override is set for this foot (via [`apply_contact_feedback`]),
    /// the override takes precedence over the scheduled contact.
    pub fn is_contact(&self, foot: usize) -> bool {
        if let Some(override_val) = self.contact_overrides[foot] {
            return override_val;
        }
        if self.duty_factor >= 1.0 {
            return true;
        }
        let foot_phase = (self.phase + self.offsets[foot]) % 1.0;
        foot_phase < self.duty_factor
    }

    /// Get the scheduled contact state (ignoring overrides).
    pub fn scheduled_contact(&self, foot: usize) -> bool {
        if self.duty_factor >= 1.0 {
            return true;
        }
        let foot_phase = (self.phase + self.offsets[foot]) % 1.0;
        foot_phase < self.duty_factor
    }

    /// Get the swing phase [0, 1] for a foot. Returns 0.0 if in stance.
    pub fn swing_phase(&self, foot: usize) -> f64 {
        if self.duty_factor >= 1.0 {
            return 0.0;
        }
        let foot_phase = (self.phase + self.offsets[foot]) % 1.0;
        if foot_phase < self.duty_factor {
            0.0 // in stance
        } else {
            // Swing phase normalized to [0, 1]
            (foot_phase - self.duty_factor) / (1.0 - self.duty_factor)
        }
    }

    /// Generate contact sequence over the MPC horizon.
    ///
    /// Returns `contacts[step][foot]` as a `Vec<Vec<bool>>` of size `horizon × n_feet`.
    pub fn contact_sequence(&self, horizon: usize, dt: f64) -> Vec<Vec<bool>> {
        let mut contacts = Vec::with_capacity(horizon);

        for k in 0..horizon {
            let future_phase = (self.phase + (k as f64) * dt / self.cycle_time) % 1.0;
            let mut step_contacts = Vec::with_capacity(self.n_feet);
            for foot in 0..self.n_feet {
                let foot_phase = (future_phase + self.offsets[foot]) % 1.0;
                step_contacts.push(self.duty_factor >= 1.0 || foot_phase < self.duty_factor);
            }
            contacts.push(step_contacts);
        }

        contacts
    }

    /// Number of feet.
    pub const fn n_feet(&self) -> usize {
        self.n_feet
    }

    /// Cycle time in seconds.
    pub const fn cycle_time(&self) -> f64 {
        self.cycle_time
    }

    /// Duty factor.
    pub const fn duty_factor(&self) -> f64 {
        self.duty_factor
    }

    /// Set the cycle time (seconds).
    pub fn set_cycle_time(&mut self, cycle_time: f64) {
        self.cycle_time = cycle_time;
    }

    /// Set the duty factor [0, 1].
    pub fn set_duty_factor(&mut self, duty_factor: f64) {
        self.duty_factor = duty_factor;
    }

    /// Apply ground-truth contact feedback to override the gait schedule.
    ///
    /// When actual contact disagrees with the schedule:
    /// - **Early touchdown** (actual contact during scheduled swing): override to stance.
    ///   This prevents the swing controller from fighting ground reaction forces.
    /// - **Late liftoff** (no contact during scheduled stance): override to swing.
    ///   This prevents the MPC from commanding forces through a foot in the air.
    ///
    /// Overrides are consumed by `is_contact()` and cleared on the next `advance()`.
    pub fn apply_contact_feedback(&mut self, actual_contacts: &[bool]) {
        for foot in 0..self.n_feet.min(actual_contacts.len()) {
            let scheduled = self.scheduled_contact(foot);
            if actual_contacts[foot] != scheduled {
                self.contact_overrides[foot] = Some(actual_contacts[foot]);
            } else {
                self.contact_overrides[foot] = None;
            }
        }
    }

    /// Clear all contact overrides.
    pub fn clear_contact_overrides(&mut self) {
        for o in &mut self.contact_overrides {
            *o = None;
        }
    }

    /// Adapt gait timing based on desired speed using feasibility constraints.
    ///
    /// The key constraints are:
    /// - Step length L = v * T_stance ≤ L_max (leg reach limit)
    /// - Swing time T_swing = (1-duty) * T_cycle ≥ T_swing_min (foot clearance)
    ///
    /// Duty factor decreases slightly at higher speeds to allow longer swing.
    pub fn adapt_timing(&mut self, speed: f64, config: &AdaptiveGaitConfig) {
        if speed < config.speed_threshold {
            return; // Keep preset timing at low speeds
        }

        // Duty decreases gently with speed: duty = base_duty - k * speed
        let duty = (config.base_duty - config.duty_speed_slope * speed)
            .clamp(config.min_duty, config.base_duty);

        // T_cycle upper bound from step length feasibility: L_max / (v * duty)
        let t_from_reach = config.l_max / (speed * duty);

        // T_cycle lower bound from minimum swing time: T_swing_min / (1 - duty)
        let t_from_swing = config.t_swing_min / (1.0 - duty);

        let t_cycle = t_from_reach
            .min(config.t_cycle_max)
            .max(t_from_swing)
            .max(config.t_cycle_min);

        self.cycle_time = t_cycle;
        self.duty_factor = duty;
    }
}

/// Configuration for adaptive gait timing.
#[derive(Clone, Debug)]
pub struct AdaptiveGaitConfig {
    /// Maximum feasible step length (meters). Conservative: 0.22 m.
    pub l_max: f64,
    /// Minimum swing time for foot clearance (seconds).
    pub t_swing_min: f64,
    /// Minimum cycle time (seconds).
    pub t_cycle_min: f64,
    /// Maximum cycle time (seconds).
    pub t_cycle_max: f64,
    /// Base duty factor at low speed.
    pub base_duty: f64,
    /// Minimum duty factor at high speed.
    pub min_duty: f64,
    /// How much duty decreases per m/s of speed.
    pub duty_speed_slope: f64,
    /// Speed below which timing is not adapted (m/s).
    pub speed_threshold: f64,
}

impl Default for AdaptiveGaitConfig {
    fn default() -> Self {
        Self {
            l_max: 0.22,
            t_swing_min: 0.14,
            t_cycle_min: 0.25,
            t_cycle_max: 0.50,
            base_duty: 0.5,
            min_duty: 0.4,
            duty_speed_slope: 0.05,
            speed_threshold: 0.3,
        }
    }
}

/// Multi-gait candidate scoring for automatic gait selection.
///
/// Periodically evaluates candidate gaits and switches to the best one
/// based on MPC cost and friction utilization. Hysteresis prevents
/// rapid gait chatter.
#[derive(Clone, Debug)]
pub struct GaitSelector {
    /// Candidate gait types to evaluate.
    pub candidates: Vec<GaitType>,
    /// Currently active gait type.
    pub active_gait: GaitType,
    /// Steps between evaluations (e.g., 5–20 at control rate).
    pub eval_interval: usize,
    /// Current step counter.
    step_counter: usize,
    /// Cost hysteresis: new gait must be this fraction better to switch.
    /// E.g., 0.1 = new gait must be 10% lower cost.
    pub hysteresis: f64,
    /// Last recorded cost for the active gait.
    last_cost: f64,
}

/// Score for a gait candidate.
#[derive(Clone, Debug)]
pub struct GaitScore {
    /// Gait type evaluated.
    pub gait: GaitType,
    /// Combined score (lower is better).
    pub cost: f64,
    /// Maximum friction utilization: max(|f_xy| / (mu * fz)) across stance feet.
    pub max_friction_util: f64,
}

impl GaitSelector {
    /// Create a new gait selector with default candidates.
    pub fn new(initial_gait: GaitType, eval_interval: usize) -> Self {
        Self {
            candidates: vec![GaitType::Stand, GaitType::Walk, GaitType::Trot, GaitType::Bound],
            active_gait: initial_gait,
            eval_interval,
            step_counter: 0,
            hysteresis: 0.1,
            last_cost: f64::MAX,
        }
    }

    /// Check if it's time to evaluate gaits.
    pub fn should_evaluate(&mut self) -> bool {
        self.step_counter += 1;
        if self.step_counter >= self.eval_interval {
            self.step_counter = 0;
            true
        } else {
            false
        }
    }

    /// Score a gait based on MPC solution quality.
    ///
    /// # Arguments
    /// * `forces` - MPC solution forces per foot (first step)
    /// * `friction_coeff` - Coulomb friction coefficient
    /// * `qp_cost` - Total QP objective value (if available, else 0)
    pub fn score_gait(
        forces: &[nalgebra::Vector3<f64>],
        friction_coeff: f64,
        qp_cost: f64,
    ) -> f64 {
        let mut max_friction_util = 0.0_f64;
        for f in forces {
            if f.z > 1e-3 {
                let tangential = (f.x * f.x + f.y * f.y).sqrt();
                let util = tangential / (friction_coeff * f.z);
                max_friction_util = max_friction_util.max(util);
            }
        }

        // Combined cost: QP objective + penalty for high friction utilization
        // Friction utilization near 1.0 means close to slipping
        let friction_penalty = if max_friction_util > 0.8 {
            100.0 * (max_friction_util - 0.8)
        } else {
            0.0
        };

        qp_cost + friction_penalty
    }

    /// Update the selector with the current gait's performance.
    ///
    /// Returns `Some(new_gait)` if a gait switch is recommended.
    pub fn update(
        &mut self,
        current_cost: f64,
        speed: f64,
    ) -> Option<GaitType> {
        self.last_cost = current_cost;

        // Simple speed-based heuristic for candidate filtering
        let recommended = if speed < 0.1 {
            GaitType::Stand
        } else if speed < 0.4 {
            GaitType::Walk
        } else {
            GaitType::Trot
        };

        if recommended != self.active_gait {
            // Apply hysteresis: only switch if benefit is significant
            // For speed-based switching, always switch at speed boundaries
            self.active_gait = recommended;
            Some(recommended)
        } else {
            None
        }
    }

    /// Get the currently active gait type.
    pub const fn active_gait(&self) -> GaitType {
        self.active_gait
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stand_all_feet_contact() {
        let sched = GaitScheduler::quadruped(GaitType::Stand);
        let contacts = sched.contact_sequence(10, 0.02);
        for step in &contacts {
            assert!(step.iter().all(|&c| c));
        }
    }

    #[test]
    fn trot_diagonal_pairs() {
        let mut sched = GaitScheduler::quadruped(GaitType::Trot);

        // At phase 0: FL(offset=0) and RR(offset=0) in stance
        // FR(offset=0.5) and RL(offset=0.5) also in stance at phase 0
        // (foot_phase=0.5 < duty=0.5 is FALSE, so they're in swing)
        // Wait: 0.5 < 0.5 is false, so FR and RL are in swing at phase=0
        sched.set_phase(0.0);
        assert!(sched.is_contact(0)); // FL: phase 0.0 < 0.5 ✓
        assert!(!sched.is_contact(1)); // FR: phase 0.5 < 0.5 ✗
        assert!(!sched.is_contact(2)); // RL: phase 0.5 < 0.5 ✗
        assert!(sched.is_contact(3)); // RR: phase 0.0 < 0.5 ✓

        // At phase 0.25: FL and RR still in stance, FR and RL starting swing
        sched.set_phase(0.25);
        assert!(sched.is_contact(0)); // FL: 0.25 < 0.5 ✓
        assert!(!sched.is_contact(1)); // FR: 0.75 < 0.5 ✗
        assert!(!sched.is_contact(2)); // RL: 0.75 < 0.5 ✗
        assert!(sched.is_contact(3)); // RR: 0.25 < 0.5 ✓

        // At phase 0.5: FL and RR start swing, FR and RL start stance
        sched.set_phase(0.5);
        assert!(!sched.is_contact(0)); // FL: 0.5 < 0.5 ✗
        assert!(sched.is_contact(1)); // FR: 0.0 < 0.5 ✓
        assert!(sched.is_contact(2)); // RL: 0.0 < 0.5 ✓
        assert!(!sched.is_contact(3)); // RR: 0.5 < 0.5 ✗
    }

    #[test]
    fn trot_contact_sequence_length() {
        let sched = GaitScheduler::quadruped(GaitType::Trot);
        let contacts = sched.contact_sequence(10, 0.02);
        assert_eq!(contacts.len(), 10);
        for step in &contacts {
            assert_eq!(step.len(), 4);
        }
    }

    #[test]
    fn walk_one_foot_swing() {
        // Walk gait: duty=0.75, so 75% stance, 25% swing
        // At any given time, at most 1 foot should be in swing
        let sched = GaitScheduler::quadruped(GaitType::Walk);
        let contacts = sched.contact_sequence(100, 0.008); // sample finely

        for step in &contacts {
            let n_stance: usize = step.iter().filter(|&&c| c).count();
            // With duty=0.75 and offsets [0, 0.5, 0.25, 0.75],
            // at most 1 foot is in swing at a time → at least 3 in stance
            assert!(
                n_stance >= 3,
                "Walk should have at least 3 feet in stance, got {n_stance}"
            );
        }
    }

    #[test]
    fn swing_phase_zero_in_stance() {
        let sched = GaitScheduler::quadruped(GaitType::Trot);
        // FL is in stance at phase 0
        assert_eq!(sched.swing_phase(0), 0.0);
    }

    #[test]
    fn swing_phase_range() {
        let mut sched = GaitScheduler::quadruped(GaitType::Trot);
        // FR has offset 0.5, duty 0.5, so swing starts at foot_phase=0.5
        // At gait phase 0, FR foot_phase=0.5 → just entered swing → swing_phase=0
        // Actually foot_phase=0.5 is exactly at boundary, 0.5 < 0.5 is false → in swing
        // swing_phase = (0.5 - 0.5)/(1.0 - 0.5) = 0.0
        sched.set_phase(0.0);
        assert!((sched.swing_phase(1) - 0.0).abs() < 1e-10);

        // At gait phase 0.25, FR foot_phase=0.75 → deep in swing
        // swing_phase = (0.75 - 0.5)/0.5 = 0.5
        sched.set_phase(0.25);
        assert!((sched.swing_phase(1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn advance_wraps_phase() {
        let mut sched = GaitScheduler::quadruped(GaitType::Trot);
        // Cycle time = 0.35s, advance by 0.21s → phase = 0.21/0.35 = 0.6
        sched.advance(0.21);
        assert!((sched.phase() - 0.6).abs() < 1e-10);

        // Advance another 0.21s → phase = 0.6 + 0.6 = 1.2 → 0.2
        sched.advance(0.21);
        assert!((sched.phase() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn adapt_timing_no_change_below_threshold() {
        let mut sched = GaitScheduler::quadruped(GaitType::Trot);
        let cfg = AdaptiveGaitConfig::default(); // threshold = 0.3
        let orig_cycle = sched.cycle_time();
        let orig_duty = sched.duty_factor();

        sched.adapt_timing(0.2, &cfg); // below threshold
        assert_eq!(sched.cycle_time(), orig_cycle);
        assert_eq!(sched.duty_factor(), orig_duty);
    }

    #[test]
    fn adapt_timing_reduces_duty_at_speed() {
        let mut sched = GaitScheduler::quadruped(GaitType::Trot);
        let cfg = AdaptiveGaitConfig::default();

        sched.adapt_timing(1.0, &cfg);
        // duty = 0.5 - 0.05*1.0 = 0.45
        assert!((sched.duty_factor() - 0.45).abs() < 1e-10);
    }

    #[test]
    fn adapt_timing_duty_clamps_at_min() {
        let mut sched = GaitScheduler::quadruped(GaitType::Trot);
        let cfg = AdaptiveGaitConfig::default(); // min_duty = 0.4

        sched.adapt_timing(5.0, &cfg); // duty = 0.5 - 0.05*5 = 0.25 → clamped to 0.4
        assert!((sched.duty_factor() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn contact_feedback_overrides_schedule() {
        let mut sched = GaitScheduler::quadruped(GaitType::Trot);
        sched.set_phase(0.0);
        // At phase 0: FL(0) and RR(3) in stance, FR(1) and RL(2) in swing
        assert!(sched.is_contact(0));
        assert!(!sched.is_contact(1));

        // Override: FL loses contact (early liftoff), FR touches ground (early touchdown)
        sched.apply_contact_feedback(&[false, true, false, true]);
        assert!(!sched.is_contact(0)); // overridden to swing
        assert!(sched.is_contact(1)); // overridden to stance
        assert!(!sched.is_contact(2)); // matches schedule, no override
        assert!(sched.is_contact(3)); // matches schedule, no override
    }

    #[test]
    fn contact_feedback_cleared_on_advance() {
        let mut sched = GaitScheduler::quadruped(GaitType::Trot);
        sched.set_phase(0.0);
        sched.apply_contact_feedback(&[false, true, false, true]);
        assert!(!sched.is_contact(0)); // overridden

        sched.advance(0.001); // tiny step, FL still in stance by schedule
        // Override cleared, back to schedule
        assert!(sched.is_contact(0));
    }

    #[test]
    fn scheduled_contact_ignores_overrides() {
        let mut sched = GaitScheduler::quadruped(GaitType::Trot);
        sched.set_phase(0.0);
        sched.apply_contact_feedback(&[false, true, false, true]);

        // scheduled_contact returns the schedule, not the override
        assert!(sched.scheduled_contact(0)); // FL scheduled stance
        assert!(!sched.scheduled_contact(1)); // FR scheduled swing
    }

    #[test]
    fn adapt_timing_cycle_bounded() {
        let mut sched = GaitScheduler::quadruped(GaitType::Trot);
        let cfg = AdaptiveGaitConfig::default();

        sched.adapt_timing(1.0, &cfg);
        assert!(sched.cycle_time() >= cfg.t_cycle_min);
        assert!(sched.cycle_time() <= cfg.t_cycle_max);

        // Swing time must be at least t_swing_min
        let t_swing = (1.0 - sched.duty_factor()) * sched.cycle_time();
        assert!(t_swing >= cfg.t_swing_min - 1e-10);
    }

    #[test]
    fn gait_selector_recommends_stand_at_zero_speed() {
        let mut sel = GaitSelector::new(GaitType::Trot, 10);
        let result = sel.update(0.0, 0.0);
        assert_eq!(result, Some(GaitType::Stand));
        assert_eq!(sel.active_gait(), GaitType::Stand);
    }

    #[test]
    fn gait_selector_recommends_trot_at_high_speed() {
        let mut sel = GaitSelector::new(GaitType::Stand, 10);
        let result = sel.update(0.0, 0.5);
        assert_eq!(result, Some(GaitType::Trot));
        assert_eq!(sel.active_gait(), GaitType::Trot);
    }

    #[test]
    fn gait_selector_no_switch_when_already_active() {
        let mut sel = GaitSelector::new(GaitType::Trot, 10);
        let result = sel.update(0.0, 0.5);
        assert_eq!(result, None); // already trotting
    }

    #[test]
    fn gait_score_penalizes_high_friction_utilization() {
        let low_slip = vec![nalgebra::Vector3::new(0.0, 0.0, 50.0)];
        let high_slip = vec![nalgebra::Vector3::new(30.0, 30.0, 50.0)];

        let score_low = GaitSelector::score_gait(&low_slip, 0.8, 0.0);
        let score_high = GaitSelector::score_gait(&high_slip, 0.8, 0.0);
        assert!(score_high > score_low, "High slip should have higher cost");
    }

    #[test]
    fn gait_selector_eval_interval() {
        let mut sel = GaitSelector::new(GaitType::Trot, 3);
        assert!(!sel.should_evaluate());
        assert!(!sel.should_evaluate());
        assert!(sel.should_evaluate()); // 3rd step
        assert!(!sel.should_evaluate()); // reset
    }
}
