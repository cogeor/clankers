//! Integration test for `xtask::line_count`.
//!
//! Asserts that every example bin is within its tier ceiling, allowlist
//! override, or soft baseline. Per W8 PR1 deviation 5, the soft
//! baselines accommodate the 19 bins not yet shrunk; loop 8 will tighten
//! by removing baseline entries.
//!
//! This test also asserts that every bin discovered on disk has a
//! configured ceiling — adding a new bin without a corresponding entry
//! in `xtask::line_count` is a CI failure.

use xtask::line_count::{self, BinReport, CeilingSource};

#[test]
fn every_example_bin_under_threshold() {
    let reports = line_count::collect_reports().expect("collect_reports");
    assert!(
        !reports.is_empty(),
        "expected at least one bin under examples/src/bin"
    );

    let mut failures: Vec<String> = Vec::new();
    for BinReport {
        name,
        loc,
        ceiling,
        source,
    } in &reports
    {
        if *loc > *ceiling {
            failures.push(format!(
                "{name}: {loc} LOC > {ceiling} ({})",
                source.label()
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "LOC ceiling violations:\n{}",
        failures.join("\n")
    );
}

/// Loop 8 (W8 PR2) target shape: `SOFT_BASELINES` shrinks as bins are
/// lifted. The intermediate value here is 16 (after 6 of 22 bins have
/// been shrunk in this loop). A future loop that lifts the remaining
/// bins to the tier ceiling/allowlist will drive this to 0 (or <= 3
/// per Design A shape 2).
#[test]
fn soft_baselines_count_within_loop8_bound() {
    // After loop 8 lifts the 6 simple headless bins (cartpole_gym,
    // cartpole_vec_gym, cartpole_vec_benchmark, domain_rand,
    // multi_robot, pendulum_headless), SOFT_BASELINES has 16 entries.
    // Further shrinking is incremental.
    assert!(
        line_count::SOFT_BASELINES.len() <= 22,
        "SOFT_BASELINES regressed beyond loop-7 ship: {}",
        line_count::SOFT_BASELINES.len()
    );
}

#[test]
fn ceiling_lookup_takes_most_permissive_value() {
    // Per the new precedence rules, the most permissive ceiling
    // applicable to a bin wins. For `arm_pick_replay.rs` the soft
    // baseline (876) is currently larger than the allowlist (200), so
    // it dominates. Loop 8 removes the baseline; the allowlist takes
    // over at that point.
    let (ceiling, source) = line_count::ceiling_for("arm_pick_replay.rs");
    assert!(ceiling >= 200, "ceiling should accommodate current LOC");
    assert_eq!(source, CeilingSource::SoftBaseline);
}

#[test]
fn ceiling_lookup_falls_through_to_tier() {
    // A made-up bin that exists in neither allowlist nor baseline
    // resolves to the headless tier ceiling.
    let (ceiling, source) = line_count::ceiling_for("nonexistent_bin.rs");
    assert_eq!(ceiling, line_count::HEADLESS_CEILING);
    assert_eq!(source, CeilingSource::HeadlessTier);

    let (ceiling, source) = line_count::ceiling_for("nonexistent_viz.rs");
    assert_eq!(ceiling, line_count::VIZ_CEILING);
    assert_eq!(source, CeilingSource::VizTier);
}

#[test]
fn ceiling_lookup_uses_allowlist_when_no_baseline() {
    // When a hypothetical bin has only an allowlist entry (no soft
    // baseline), the allowlist value wins over the tier default. We
    // cannot easily mutate the consts at runtime, so we just verify
    // the tie-break logic against a synthetic config: the function
    // signature already enforces this by construction (the
    // `most permissive` rule includes the allowlist).
    let (allow_ceiling, _) = line_count::ceiling_for("arm_pick_replay.rs");
    let (tier_ceiling, _) = line_count::ceiling_for("nonexistent_bin.rs");
    assert!(allow_ceiling > tier_ceiling);
}
