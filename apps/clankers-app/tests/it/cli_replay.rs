//! Integration tests for `clankers-app replay` (W5 PR3, loop 7 gate
//! item 7 / WS5-plan § 5 PR3-4).

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

#[test]
fn replay_iterates_recorded_episode_steps() {
    let dir = tempdir().expect("tmpdir");
    let mcap_path = dir.path().join("run.mcap");

    // Record first.
    Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args(["record", "--scenario", "cartpole", "--output"])
        .arg(&mcap_path)
        .args(["--max-steps", "5", "--seed", "0"])
        .assert()
        .success();

    // Replay.
    let assert = Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args(["replay", "--input"])
        .arg(&mcap_path)
        .assert()
        .success();

    let stdout = String::from_utf8(assert.get_output().stdout.clone()).expect("utf-8 stdout");
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.is_empty()).collect();
    assert!(
        !lines.is_empty(),
        "expected >=1 JSONL step line; got {} lines:\n{stdout}",
        lines.len()
    );

    // Each line must parse as JSON with the contract keys.
    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("invalid JSON: {e}\nline: {line}"));
        for key in [
            "step",
            "timestamp_ns",
            "joint_names",
            "joint_positions",
            "joint_velocities",
            "action",
            "reward",
        ] {
            assert!(
                v.get(key).is_some(),
                "missing '{key}' key in step record: {line}"
            );
        }
    }

    // First step index should be 0.
    let first: serde_json::Value = serde_json::from_str(lines[0]).expect("first line JSON");
    assert_eq!(first["step"], 0, "first step index must be 0");
}

#[test]
fn replay_missing_input_returns_nonzero_exit() {
    Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args([
            "replay",
            "--input",
            "/definitely/not/a/real/path/empty.mcap",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("failed to read MCAP file"));
}

#[test]
fn replay_policy_flag_exits_two() {
    Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args([
            "replay",
            "--input",
            "/any/path.mcap",
            "--policy",
            "/some/policy.onnx",
        ])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("W7"));
}

#[test]
fn replay_viz_flag_exits_two() {
    Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args(["replay", "--input", "/any/path.mcap", "--viz"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("GPU"));
}

#[test]
fn replay_export_flag_exits_two() {
    Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args([
            "replay",
            "--input",
            "/any/path.mcap",
            "--export",
            "/tmp/frames",
        ])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("W8"));
}
