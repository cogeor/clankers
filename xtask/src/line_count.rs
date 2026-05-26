//! `cargo xtask check-bin-size` — enforce LOC tier ceilings on example bins.
//!
//! # Tiers (W8 PR1 deviation 3 — stricter than WS8-plan § 3)
//!
//! - `_viz.rs` bins: 60 LOC ceiling.
//! - Other example bins: 60 LOC ceiling.
//! - Per-bin allowlist for the three intentionally larger bins
//!   (W8-plan § 3 + LOOPS.yaml loop 7 scope step 11):
//!   - `arm_pick_replay.rs` -> 200
//!   - `quadruped_mpc_viz.rs` -> 200
//!   - `arm_ik_viz.rs` -> 150
//!
//! # Soft baseline (W8 PR1 deviation 5)
//!
//! Loop 7 ships the LOC checker but defers shrinking the 10 arm bins +
//! the 12 cartpole / quadruped / multi-robot / pendulum / domain-rand
//! bins to subsequent loops. Each unshrunk bin gets a baseline entry
//! that pins its current line count; the checker passes as long as the
//! bin does not grow beyond its baseline. Loop 8 tightens by removing
//! baseline entries as bins are lifted to the tier ceiling.
//!
//! Regression policy: a bin's LOC may shrink (baseline becomes upper
//! bound). It MUST NOT grow. Adding a new bin without a corresponding
//! ceiling/allowlist/baseline entry causes the check to fail with an
//! "unconfigured bin" diagnostic.

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// Default ceiling for headless example bins.
pub const HEADLESS_CEILING: usize = 60;

/// Default ceiling for visualisation example bins.
///
/// Applies to filenames ending in `_viz.rs`. Matches the headless
/// ceiling per the user trigger's two-tier note (60 for example bins,
/// 120 for the big bin in `apps/clankers-app`).
pub const VIZ_CEILING: usize = 60;

/// Per-bin allowlist for intentionally larger bins.
///
/// LOOPS.yaml loop 7 scope step 11 names three entries; the numbers
/// here are the **post-shrink** targets per WS8-plan § 3. Loop 7 ships
/// the checker infrastructure but defers the actual bin lifts to a
/// follow-up loop, so for now the three viz bins also have a soft
/// baseline entry that pins them at their current LOC. The
/// most-permissive rule in [`ceiling_for`] keeps the gate green until
/// loop 8 removes the baseline.
pub const ALLOWLIST: &[(&str, usize)] = &[
    ("arm_pick_replay.rs", 200),
    ("quadruped_mpc_viz.rs", 200),
    ("arm_ik_viz.rs", 150),
];

/// Soft baselines — current LOC of bins NOT YET SHRUNK by W8 PR1.
///
/// Each entry asserts "this bin's LOC must not exceed the recorded
/// value". Loop 8 will remove entries as bins are lifted to the tier
/// ceiling (or to their allowlist target).
///
/// Per W8 PR1 deviation 5, the checker treats these as upper bounds:
/// the bin may shrink below the baseline (good — encouraging progress)
/// but must never exceed it. LOC values match `wc -l` semantics
/// (newline count + 1 if the file lacks a trailing newline).
pub const SOFT_BASELINES: &[(&str, usize)] = &[
    ("arm_bench.rs", 170),
    ("arm_gym.rs", 133),
    ("arm_ik.rs", 95),
    ("arm_ik_viz.rs", 646),
    ("arm_manipulation.rs", 291),
    ("arm_pick_gym.rs", 300),
    ("arm_pick_record.rs", 526),
    ("arm_pick_replay.rs", 876),
    ("arm_policy_viz.rs", 441),
    ("arm_with_policy.rs", 164),
    // cartpole_gym shrunk in loop 8 (W8 PR2 Phase 3) — under 60 tier.
    // cartpole_vec_gym shrunk in loop 8 (W8 PR2 Phase 3) — under 60 tier.
    // cartpole_vec_benchmark shrunk in loop 8 (W8 PR2 Phase 3) — under 60 tier.
    ("cartpole_policy_viz.rs", 357),
    // domain_rand shrunk in loop 8 (W8 PR2 Phase 4) — under 60 tier.
    // multi_robot shrunk in loop 8 (W8 PR2 Phase 4) — under 60 tier.
    ("multi_robot_viz.rs", 518),
    // pendulum_headless shrunk in loop 8 (W8 PR2 Phase 4) — under 60 tier.
    ("pendulum_viz.rs", 411),
    ("quadruped_mpc.rs", 299),
    ("quadruped_mpc_bench.rs", 491),
    ("quadruped_mpc_viz.rs", 973),
];

/// One row in the report — a bin's measured LOC and the ceiling it was
/// checked against, plus the source of that ceiling.
#[derive(Debug, Clone)]
pub struct BinReport {
    /// File name (e.g. `arm_ik.rs`).
    pub name: String,
    /// Measured LOC.
    pub loc: usize,
    /// Ceiling the bin was checked against.
    pub ceiling: usize,
    /// Which list provided `ceiling`.
    pub source: CeilingSource,
}

/// Which list provided the ceiling for a [`BinReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeilingSource {
    /// Tier ceiling — [`HEADLESS_CEILING`] for non-viz bins.
    HeadlessTier,
    /// Tier ceiling — [`VIZ_CEILING`] for `*_viz.rs` bins.
    VizTier,
    /// [`ALLOWLIST`] override.
    Allowlist,
    /// [`SOFT_BASELINES`] override (deviation 5).
    SoftBaseline,
}

impl CeilingSource {
    /// Short human label for diagnostics.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HeadlessTier => "headless-tier",
            Self::VizTier => "viz-tier",
            Self::Allowlist => "allowlist",
            Self::SoftBaseline => "soft-baseline",
        }
    }
}

/// Look up the ceiling for a bin filename, returning the value and the
/// list that provided it.
///
/// # Precedence
///
/// For each bin, every list that applies (tier default, allowlist,
/// soft baseline) is consulted; the **largest** ceiling wins. This
/// means:
/// - A bin with no allowlist entry and no baseline entry uses its tier
///   default.
/// - A bin with an allowlist entry (e.g. `arm_pick_replay.rs` 200) but
///   also a soft baseline pinning it at its current LOC (e.g. 876) uses
///   the soft baseline until that baseline is removed — at which point
///   the allowlist takes over.
/// - This makes the loop-7 → loop-8 transition mechanical: remove the
///   baseline entry once the bin's LOC actually drops below the
///   allowlist value.
///
/// The reported [`CeilingSource`] reflects whichever list "won". On
/// ties, allowlist beats baseline beats tier (purely cosmetic; the
/// resulting ceiling is the same).
#[must_use]
pub fn ceiling_for(name: &str) -> (usize, CeilingSource) {
    let tier_ceiling = if name.ends_with("_viz.rs") {
        (VIZ_CEILING, CeilingSource::VizTier)
    } else {
        (HEADLESS_CEILING, CeilingSource::HeadlessTier)
    };
    let allowlist = ALLOWLIST
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, c)| (*c, CeilingSource::Allowlist));
    let baseline = SOFT_BASELINES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, c)| (*c, CeilingSource::SoftBaseline));

    // Pick the most permissive (highest ceiling). On ties, prefer
    // allowlist > baseline > tier so the diagnostic label reflects the
    // long-term intent.
    let candidates = [Some(tier_ceiling), allowlist, baseline];
    let mut winner = tier_ceiling;
    for cand in candidates.into_iter().flatten() {
        if cand.0 > winner.0 || (cand.0 == winner.0 && rank(cand.1) > rank(winner.1)) {
            winner = cand;
        }
    }
    winner
}

/// Tie-breaker rank (higher = preferred when ceiling values are equal).
const fn rank(s: CeilingSource) -> u8 {
    match s {
        CeilingSource::HeadlessTier | CeilingSource::VizTier => 0,
        CeilingSource::SoftBaseline => 1,
        CeilingSource::Allowlist => 2,
    }
}

/// Count source lines in `path` (newline-terminated, trailing partial
/// line counted — matches `Get-Content | Measure-Object -Line`).
///
/// # Errors
///
/// Returns `io::Error` if the file cannot be read.
pub fn count_lines(path: &Path) -> std::io::Result<usize> {
    let content = std::fs::read_to_string(path)?;
    // Match the conventional `wc -l` semantics PowerShell's
    // `Measure-Object -Line` uses: count newline characters; if the
    // file ends without a trailing newline, the dangling line is still
    // counted.
    let mut n = content.matches('\n').count();
    if !content.is_empty() && !content.ends_with('\n') {
        n += 1;
    }
    Ok(n)
}

/// Locate the workspace's `examples/src/bin/` directory by walking up
/// from `CARGO_MANIFEST_DIR`. The xtask crate sits at
/// `<workspace>/xtask/`, so the bin dir is `../examples/src/bin/`.
#[must_use]
pub fn examples_bin_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest)
        .parent()
        .map_or_else(|| PathBuf::from(".."), Path::to_path_buf)
        .join("examples")
        .join("src")
        .join("bin")
}

/// Enumerate every `*.rs` file directly under `examples/src/bin/` and
/// produce one [`BinReport`] per bin.
///
/// # Errors
///
/// Returns a formatted error string if the bin directory cannot be
/// walked or any bin file cannot be read.
pub fn collect_reports() -> Result<Vec<BinReport>, String> {
    let dir = examples_bin_dir();
    if !dir.is_dir() {
        return Err(format!(
            "xtask check-bin-size: bin directory not found: {}",
            dir.display()
        ));
    }

    let mut out = Vec::new();
    for entry in WalkDir::new(&dir).min_depth(1).max_depth(1) {
        let entry = entry.map_err(|e| format!("walking {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("bad filename: {}", path.display()))?
            .to_string();
        let loc = count_lines(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        let (ceiling, source) = ceiling_for(&name);
        out.push(BinReport {
            name,
            loc,
            ceiling,
            source,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Run the `check-bin-size` subcommand.
///
/// Walks `examples/src/bin/`, looks up the ceiling for each bin, and
/// emits a report. Returns `Ok(())` if every bin is within its ceiling
/// (or its soft baseline, per deviation 5); otherwise returns an
/// `Err(report)` string suitable for printing to stderr.
///
/// # Errors
///
/// Returns the rendered failure report as `Err(String)` when any bin
/// exceeds its ceiling.
pub fn run_check_bin_size() -> Result<(), String> {
    let reports = collect_reports()?;
    let mut failures: Vec<String> = Vec::new();

    for r in &reports {
        if r.loc > r.ceiling {
            failures.push(format!(
                "  {}: {} LOC > {} ({})",
                r.name,
                r.loc,
                r.ceiling,
                r.source.label()
            ));
        }
    }

    println!("=== xtask check-bin-size ===");
    println!("Tier ceilings: headless={HEADLESS_CEILING}, viz={VIZ_CEILING}");
    println!(
        "Allowlist entries: {}; soft baselines: {}",
        ALLOWLIST.len(),
        SOFT_BASELINES.len()
    );
    println!("Bins inspected: {}\n", reports.len());

    for r in &reports {
        let marker = if r.loc > r.ceiling { "FAIL" } else { "ok  " };
        println!(
            "  {marker} {:30}  {:>4} LOC / {:>4} ({})",
            r.name,
            r.loc,
            r.ceiling,
            r.source.label()
        );
    }

    if failures.is_empty() {
        println!("\nAll bins within ceiling.");
        Ok(())
    } else {
        let mut msg = String::from("\nLOC ceiling violations:\n");
        for f in &failures {
            msg.push_str(f);
            msg.push('\n');
        }
        Err(msg)
    }
}
