//! Per-step simulation phase timings (G10).
//!
//! `CODE_QUALITY_REVIEW` Gap Analysis 10 / "Observability and
//! Debuggability Are Underdeveloped". Performance regressions today
//! require custom instrumentation; users can't tell whether a slow
//! step is physics, sensors, protocol, recording, or policy without
//! adding their own timers.
//!
//! [`SimDiagnostics`] is a Bevy `Resource` holding the most recent
//! per-step microsecond breakdown plus a rolling window for
//! short-term smoothing. Producers update the slot they own with
//! [`SimDiagnostics::record_phase`]; consumers (CLI summaries,
//! viz panels, recorder MCAP metadata) read the snapshot via
//! [`SimDiagnostics::last`] or [`SimDiagnostics::rolling_mean`].
//!
//! This commit defines the data shape and the recording API only.
//! Phase 3 / future work wires individual physics / sensor / recorder
//! systems to call [`SimDiagnostics::record_phase`] at the right
//! points; the shape is stable enough that those wires don't have
//! to come along atomically.

use std::collections::VecDeque;

use bevy::prelude::Resource;

// ---------------------------------------------------------------------------
// SimPhase
// ---------------------------------------------------------------------------

/// Discrete phase of the simulation step. Each is a slot in
/// [`TelemetryFrame`]; producers attribute their work to exactly one
/// phase so the sum is meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimPhase {
    /// Reading the action buffer + applying commands to joints.
    ActionApply,
    /// The physics backend step itself.
    Physics,
    /// Sensor reads + observation buffer fill.
    Sensor,
    /// Policy inference (when running an in-process policy).
    Policy,
    /// Recorder write / queue push.
    Record,
    /// Render / present (when running viz).
    Render,
}

impl SimPhase {
    /// Position of this phase in [`TelemetryFrame::phase_us`]. Kept
    /// in sync with [`Self::PHASES`] so iteration order matches.
    pub const fn idx(self) -> usize {
        match self {
            Self::ActionApply => 0,
            Self::Physics => 1,
            Self::Sensor => 2,
            Self::Policy => 3,
            Self::Record => 4,
            Self::Render => 5,
        }
    }

    /// All phases, in canonical step order.
    pub const PHASES: [Self; 6] = [
        Self::ActionApply,
        Self::Physics,
        Self::Sensor,
        Self::Policy,
        Self::Record,
        Self::Render,
    ];
}

// ---------------------------------------------------------------------------
// TelemetryFrame
// ---------------------------------------------------------------------------

/// One simulation step's worth of phase timings in microseconds.
///
/// `phase_us` is indexed by [`SimPhase::idx`]; zero means "no time
/// attributed to this phase this step" (e.g. policy idle, render off).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TelemetryFrame {
    /// Per-phase elapsed microseconds. Indices come from
    /// [`SimPhase::idx`]; do not read by position without going
    /// through the helper.
    pub phase_us: [u64; 6],
    /// Total step microseconds (sum of `phase_us` plus any work the
    /// producer attributes to "other" — typically Bevy schedule
    /// overhead). Computed by the consumer; this struct holds the
    /// raw signal only.
    pub total_us: u64,
}

impl TelemetryFrame {
    /// Read the microsecond count attributed to `phase` this step.
    #[must_use]
    pub const fn phase(&self, phase: SimPhase) -> u64 {
        self.phase_us[phase.idx()]
    }

    /// Sum of every phase counter. Useful as a denominator for
    /// percentage breakdowns.
    #[must_use]
    pub fn sum_phases_us(&self) -> u64 {
        self.phase_us.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// SimDiagnostics
// ---------------------------------------------------------------------------

/// Default rolling-window length (≈ 1 s at 60 Hz, 0.5 s at 120 Hz).
const DEFAULT_WINDOW: usize = 64;

/// Bevy resource exposing the most recent [`TelemetryFrame`] plus a
/// short rolling window for smoothing CLI / viz read-outs.
///
/// Producers call [`Self::record_phase`] from inside whichever system
/// owns a phase. The simulation orchestrator calls
/// [`Self::finalize_step`] at the end of each step to atomically
/// publish the current frame into the rolling window and clear the
/// per-phase accumulators for the next step.
#[derive(Resource, Debug, Clone)]
pub struct SimDiagnostics {
    /// Frame currently under construction this step.
    current: TelemetryFrame,
    /// Most recently finalised frame; what consumers read.
    last: TelemetryFrame,
    /// Rolling window of finalised frames for short-term smoothing.
    window: VecDeque<TelemetryFrame>,
    /// Cap on `window` length.
    window_cap: usize,
}

impl Default for SimDiagnostics {
    fn default() -> Self {
        Self {
            current: TelemetryFrame::default(),
            last: TelemetryFrame::default(),
            window: VecDeque::with_capacity(DEFAULT_WINDOW),
            window_cap: DEFAULT_WINDOW,
        }
    }
}

impl SimDiagnostics {
    /// Override the rolling-window length. Capped at construction so
    /// `record_phase` can stay branch-free.
    #[must_use]
    pub fn with_window(mut self, n: usize) -> Self {
        self.window_cap = n.max(1);
        self.window = VecDeque::with_capacity(self.window_cap);
        self
    }

    /// Attribute `us` microseconds of work to `phase` for the current
    /// step. Cheap (one array slot add); safe to call from any
    /// number of systems in any order within one step.
    pub const fn record_phase(&mut self, phase: SimPhase, us: u64) {
        self.current.phase_us[phase.idx()] = self.current.phase_us[phase.idx()].saturating_add(us);
    }

    /// Attribute total-step microseconds (typically wall-clock from a
    /// schedule timer). Overwrites; the orchestrator owns this slot.
    pub const fn record_total(&mut self, us: u64) {
        self.current.total_us = us;
    }

    /// Publish the current step's frame into the rolling window and
    /// reset the accumulators for the next step. Idempotent if called
    /// on an empty step (publishes a zeroed frame).
    pub fn finalize_step(&mut self) {
        self.last = self.current;
        if self.window.len() >= self.window_cap {
            self.window.pop_front();
        }
        self.window.push_back(self.current);
        self.current = TelemetryFrame::default();
    }

    /// Most recent finalised frame.
    #[must_use]
    pub const fn last(&self) -> TelemetryFrame {
        self.last
    }

    /// Number of frames currently buffered in the rolling window.
    #[must_use]
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Per-phase mean over the rolling window. Returns a zeroed
    /// frame when the window is empty.
    #[must_use]
    pub fn rolling_mean(&self) -> TelemetryFrame {
        if self.window.is_empty() {
            return TelemetryFrame::default();
        }
        let n = self.window.len() as u64;
        let mut sum = [0u64; 6];
        let mut sum_total = 0u64;
        for f in &self.window {
            for (i, v) in f.phase_us.iter().enumerate() {
                sum[i] = sum[i].saturating_add(*v);
            }
            sum_total = sum_total.saturating_add(f.total_us);
        }
        let mut mean = [0u64; 6];
        for (i, s) in sum.iter().enumerate() {
            mean[i] = s / n;
        }
        TelemetryFrame {
            phase_us: mean,
            total_us: sum_total / n,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_phase_idx_is_unique_and_in_range() {
        let mut seen = [false; 6];
        for phase in SimPhase::PHASES {
            let i = phase.idx();
            assert!(i < 6, "{phase:?} idx out of range");
            assert!(!seen[i], "duplicate idx for {phase:?}");
            seen[i] = true;
        }
        assert!(seen.iter().all(|v| *v), "every slot must be assigned");
    }

    #[test]
    fn record_phase_accumulates_within_step() {
        let mut d = SimDiagnostics::default();
        d.record_phase(SimPhase::Physics, 100);
        d.record_phase(SimPhase::Physics, 50);
        d.record_phase(SimPhase::Sensor, 25);
        d.finalize_step();
        let f = d.last();
        assert_eq!(f.phase(SimPhase::Physics), 150);
        assert_eq!(f.phase(SimPhase::Sensor), 25);
        assert_eq!(f.phase(SimPhase::Render), 0);
    }

    #[test]
    fn finalize_clears_current() {
        let mut d = SimDiagnostics::default();
        d.record_phase(SimPhase::Physics, 100);
        d.finalize_step();
        d.finalize_step(); // empty step -> zeroed last
        assert_eq!(d.last(), TelemetryFrame::default());
    }

    #[test]
    fn rolling_mean_averages_window() {
        let mut d = SimDiagnostics::default().with_window(4);
        for us in [100, 200, 300, 400] {
            d.record_phase(SimPhase::Physics, us);
            d.finalize_step();
        }
        let mean = d.rolling_mean();
        assert_eq!(mean.phase(SimPhase::Physics), 250); // (100+200+300+400) / 4
    }

    #[test]
    fn rolling_window_caps_length() {
        let mut d = SimDiagnostics::default().with_window(2);
        for us in [10, 20, 30, 40] {
            d.record_phase(SimPhase::Physics, us);
            d.finalize_step();
        }
        // Only the last 2 frames stay; mean over (30, 40) = 35.
        assert_eq!(d.window_len(), 2);
        assert_eq!(d.rolling_mean().phase(SimPhase::Physics), 35);
    }

    #[test]
    fn empty_diagnostics_returns_zeroed_mean() {
        let d = SimDiagnostics::default();
        let mean = d.rolling_mean();
        for phase in SimPhase::PHASES {
            assert_eq!(mean.phase(phase), 0);
        }
    }

    #[test]
    fn record_total_overwrites_current() {
        let mut d = SimDiagnostics::default();
        d.record_total(500);
        d.record_total(1000);
        d.finalize_step();
        assert_eq!(d.last().total_us, 1000);
    }

    #[test]
    fn sum_phases_us_matches_total_when_no_other_work() {
        let mut d = SimDiagnostics::default();
        d.record_phase(SimPhase::Physics, 10);
        d.record_phase(SimPhase::Sensor, 20);
        d.record_phase(SimPhase::Render, 30);
        d.finalize_step();
        assert_eq!(d.last().sum_phases_us(), 60);
    }
}
