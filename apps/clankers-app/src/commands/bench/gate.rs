//! Ratio-gate logic for `bench vec`.
//!
//! [`evaluate_gate`] is the CLI-facing entry point: it converts the
//! checked outcome into an `ExitCode`. The actual decision lives in
//! [`check_ratio_gate`] so it's unit-testable in isolation.

use std::process::ExitCode;

/// Apply the ratio gate (if enabled) and convert to the canonical
/// `ExitCode` (0 pass / no gate, 1 fail). Extracted from `bench_vec` to
/// keep that function under the workspace clippy `too_many_lines` cap and
/// to keep the gate's exit-code mapping in one auditable spot.
pub(super) fn evaluate_gate(gate_rows: &[(u16, f64)], gate: f64) -> ExitCode {
    if gate <= 0.0 {
        return ExitCode::SUCCESS;
    }
    match check_ratio_gate(gate_rows, gate) {
        Ok((n, x)) => {
            eprintln!("RATIO GATE: PASS N={n} throughput_x={x:.3} >= {gate:.3}");
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(1)
        }
    }
}

/// Outcome of a ratio-gate check.
///
/// `Ok((n, x))` — gate passed (or no rows to gate on; returns sentinel
/// `(0, 0.0)`). `Err(msg)` — gate failed; `msg` is the human-readable
/// failure reason already formatted for stderr (starts with `RATIO GATE: FAIL`).
/// The caller is responsible for emitting the `RATIO GATE: PASS ...` line
/// on success and choosing the exit code.
///
/// `rows` is a list of `(num_envs, throughput_x)` pairs in the order the
/// bench loop produced them. The gate prefers `N=8` and falls back to the
/// highest-N row available, logging a warning to stderr in the fallback case.
fn check_ratio_gate(rows: &[(u16, f64)], gate: f64) -> Result<(u16, f64), String> {
    if rows.is_empty() {
        eprintln!("RATIO GATE: no rows; gate skipped");
        return Ok((0, 0.0));
    }

    let (n, x) = if let Some(&row) = rows.iter().find(|(n, _)| *n == 8) {
        row
    } else {
        let &row = rows
            .iter()
            .max_by_key(|(n, _)| *n)
            .expect("rows non-empty (checked above)");
        eprintln!(
            "RATIO GATE: --envs did not include 8; gating on N={} instead",
            row.0
        );
        row
    };

    if x >= gate {
        Ok((n, x))
    } else {
        Err(format!(
            "RATIO GATE: FAIL N={n} throughput_x={x:.3} < {gate:.3}"
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_ratio_gate_empty_rows_passes_with_warning() {
        // Opt-in gate; if the bench produced no rows (e.g. degenerate
        // --envs), don't fail CI — that's a configuration problem,
        // not a regression. The helper returns Ok with sentinel (0, 0.0).
        let result = check_ratio_gate(&[], 2.0);
        assert!(result.is_ok(), "empty rows should not fail the gate");
    }

    #[test]
    fn check_ratio_gate_passes_when_ratio_meets_floor() {
        // N=8 row with throughput_x = 4.5 vs gate K=2.0 -> pass.
        let rows = vec![(1u16, 1.0_f64), (8u16, 4.5_f64)];
        let result = check_ratio_gate(&rows, 2.0);
        assert!(result.is_ok());
        let (n, x) = result.unwrap();
        assert_eq!(n, 8);
        assert!((x - 4.5).abs() < 1e-9);
    }

    #[test]
    fn check_ratio_gate_fails_with_specific_message_when_below_floor() {
        // N=8 row with throughput_x = 0.5 vs gate K=2.0 -> fail.
        let rows = vec![(1u16, 1.0_f64), (8u16, 0.5_f64)];
        let result = check_ratio_gate(&rows, 2.0);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        // The message MUST mention both the observed value and the
        // gate, so the CI log tells the on-call engineer why it failed.
        assert!(msg.contains("0.5") || msg.contains("0.500"), "msg={msg}");
        assert!(msg.contains("2.0") || msg.contains("2.000"), "msg={msg}");
        assert!(msg.to_uppercase().contains("FAIL"), "msg={msg}");
    }

    #[test]
    fn check_ratio_gate_falls_back_to_highest_n_when_8_missing() {
        // --envs 1,2,4 -> no N=8 row. Gate on N=4 instead, passing if
        // that row meets the floor.
        let rows = vec![(1u16, 1.0_f64), (2u16, 1.8_f64), (4u16, 3.2_f64)];
        let result = check_ratio_gate(&rows, 2.0);
        assert!(result.is_ok());
        let (n, x) = result.unwrap();
        assert_eq!(n, 4, "fallback should pick highest-N row");
        assert!((x - 3.2).abs() < 1e-9);
    }
}
