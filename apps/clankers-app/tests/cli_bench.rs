//! Integration tests for `clankers-app bench` (W5 PR4, loop 8).
//!
//! All three tests use `cartpole` (lighter scenario than `arm_pick`) and
//! deliberately small `--runs` / `--warmup-runs` / `--max-steps` values
//! so the suite stays under ~250 ms each on a developer machine.
//!
//! Tests never assert throughput values — those vary by host. Only:
//!
//! 1. The locked 11-column CSV header is byte-equal to the documented
//!    constant.
//! 2. JSON mode emits exactly one JSON object (not JSONL).
//! 3. Unknown scenarios exit nonzero with `unknown scenario` on stderr.

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use std::fs;
use tempfile::tempdir;

const LOCKED_HEADER: &str = "scenario,steps,runs,max_steps,wall_ms_total,wall_ms_per_run_mean,wall_ms_per_run_std,steps_per_sec_mean,step_us_p50,step_us_p99,notes";

#[test]
fn bench_writes_csv_with_locked_header() {
    let dir = tempdir().expect("tmpdir");
    let csv_path = dir.path().join("bench.csv");

    Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args(["bench", "--scenario", "cartpole", "--csv"])
        .arg(&csv_path)
        .args([
            "--runs",
            "2",
            "--warmup-runs",
            "1",
            "--max-steps",
            "50",
            "--seed",
            "0",
        ])
        .assert()
        .success();

    let text = fs::read_to_string(&csv_path).expect("CSV exists");
    let mut lines = text.lines();
    let header = lines.next().expect("CSV has a header line");
    assert_eq!(
        header, LOCKED_HEADER,
        "CSV header must be byte-equal to the locked schema",
    );
    let row = lines.next().expect("CSV has a data row");
    assert!(lines.next().is_none(), "CSV has exactly two lines");

    // Spot-check column 1 (scenario) and column 2 (steps = runs * max_steps).
    let cells: Vec<&str> = row.split(',').collect();
    assert_eq!(
        cells.len(),
        11,
        "data row must have 11 columns, got {}: {row}",
        cells.len()
    );
    assert_eq!(cells[0], "cartpole");
    assert_eq!(cells[1], "100", "steps = runs(2) * max_steps(50) = 100");
    assert_eq!(cells[2], "2", "runs column");
    assert_eq!(cells[3], "50", "max_steps column");
}

#[test]
fn bench_json_mode_emits_single_json_object() {
    let output = Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args([
            "bench",
            "--scenario",
            "cartpole",
            "--json",
            "--runs",
            "2",
            "--warmup-runs",
            "1",
            "--max-steps",
            "50",
            "--seed",
            "0",
        ])
        .assert()
        .success()
        .get_output()
        .clone();

    let stdout = String::from_utf8(output.stdout).expect("utf-8 stdout");
    // A single JSON object (NOT JSONL): the entire stdout must parse as
    // one Value::Object after a trim.
    let value: Value = serde_json::from_str(stdout.trim()).expect("stdout parses as JSON");
    let obj = value.as_object().expect("top-level value is an object");

    for key in [
        "scenario",
        "steps",
        "runs",
        "max_steps",
        "wall_ms_total",
        "wall_ms_per_run_mean",
        "wall_ms_per_run_std",
        "steps_per_sec_mean",
        "step_us_p50",
        "step_us_p99",
        "notes",
    ] {
        assert!(obj.contains_key(key), "missing locked key `{key}`");
    }
    assert_eq!(obj["scenario"].as_str(), Some("cartpole"));
    assert_eq!(obj["runs"].as_u64(), Some(2));
    assert_eq!(obj["max_steps"].as_u64(), Some(50));
    assert_eq!(obj["steps"].as_u64(), Some(100));
    assert!(
        obj["steps_per_sec_mean"].as_f64().unwrap_or(0.0) > 0.0,
        "steps_per_sec_mean must be positive on a successful bench"
    );
}

#[test]
fn bench_unknown_scenario_returns_nonzero_exit() {
    Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args([
            "bench",
            "--scenario",
            "does-not-exist",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--max-steps",
            "5",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown scenario"));
}
