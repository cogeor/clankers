//! Integration tests for `clankers-app record` (W5 PR3, loop 7 gate
//! item 6 / WS5-plan § 5 PR3-3).
//!
//! Uses `cartpole` (lighter scenario than `arm_pick`) per PLAN Design
//! choice I — keeps test wall-time minimal under the orchestrator's
//! resource constraints.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::tempdir;

#[test]
fn record_writes_mcap_file() {
    let dir = tempdir().expect("tmpdir");
    let mcap_path = dir.path().join("run.mcap");

    Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args(["record", "--scenario", "cartpole", "--output"])
        .arg(&mcap_path)
        .args(["--max-steps", "5", "--seed", "0"])
        .assert()
        .success()
        .stderr(predicate::str::contains("Dropped frames: 0"));

    let bytes = fs::read(&mcap_path).expect("MCAP file exists");
    // MCAP magic bytes: 0x89 'M' 'C' 'A' 'P' '0' '\r' '\n'
    assert!(
        bytes.starts_with(b"\x89MCAP0\r\n"),
        "MCAP magic missing; first 16 bytes = {:02x?}",
        &bytes[..bytes.len().min(16)]
    );
    assert!(
        bytes.len() > 100,
        "MCAP file too small ({} bytes); expected at least the header + schema + footer",
        bytes.len()
    );
}

#[test]
fn record_unknown_scenario_returns_nonzero_exit() {
    let dir = tempdir().expect("tmpdir");
    Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args(["record", "--scenario", "does-not-exist", "--output"])
        .arg(dir.path().join("noop.mcap"))
        .args(["--max-steps", "1"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown scenario"));
}

#[test]
fn record_camera_flag_emits_warning_and_succeeds() {
    let dir = tempdir().expect("tmpdir");
    let mcap_path = dir.path().join("cam.mcap");

    Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args(["record", "--scenario", "cartpole", "--output"])
        .arg(&mcap_path)
        .args(["--max-steps", "2", "--camera", "front"])
        .assert()
        .success()
        .stderr(predicate::str::contains("camera recording requires GPU"));
}
