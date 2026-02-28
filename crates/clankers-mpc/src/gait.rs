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
            offsets,
            duty_factor,
            cycle_time,
            phase: 0.0,
        }
    }

    /// Create a custom gait scheduler.
    pub const fn custom(
        offsets: Vec<f64>,
        duty_factor: f64,
        cycle_time: f64,
    ) -> Self {
        let n_feet = offsets.len();
        Self {
            n_feet,
            offsets,
            duty_factor,
            cycle_time,
            phase: 0.0,
        }
    }

    /// Advance the gait phase by `dt` seconds.
    pub fn advance(&mut self, dt: f64) {
        self.phase = (self.phase + dt / self.cycle_time) % 1.0;
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
    pub fn is_contact(&self, foot: usize) -> bool {
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
}
