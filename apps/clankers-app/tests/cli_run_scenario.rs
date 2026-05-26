//! Integration test for `clankers-app run --scenario <name>` (W5 PR2,
//! loop 6 gate item 4 / WS5-plan § 6).

#[test]
fn run_arm_pick_scenario_completes_one_episode() {
    let output = assert_cmd::Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args([
            "run",
            "--scenario",
            "arm_pick",
            "--episodes",
            "1",
            "--max-steps",
            "50",
            "--seed",
            "0",
            "--json",
        ])
        .output()
        .expect("failed to spawn clankers-app");

    assert!(
        output.status.success(),
        "exit code {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let last_line = stdout
        .lines()
        .last()
        .expect("at least one NDJSON line on stdout")
        .to_string();

    let v: serde_json::Value =
        serde_json::from_str(&last_line).expect("last stdout line should be NDJSON");
    assert_eq!(v["episode"], 1, "first episode index is 1");
    assert!(
        v["steps"].as_u64().expect("steps is a number") <= 50,
        "steps must be <= max-steps (50); got {:?}",
        v["steps"]
    );
}

#[test]
fn run_cartpole_scenario_emits_multiple_ndjson_lines() {
    let output = assert_cmd::Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args([
            "run",
            "--scenario",
            "cartpole",
            "--episodes",
            "2",
            "--max-steps",
            "10",
            "--json",
        ])
        .output()
        .expect("failed to spawn clankers-app");

    assert!(
        output.status.success(),
        "exit code {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        lines.len(),
        2,
        "expected 2 NDJSON lines (one per episode), got {}: {stdout:?}",
        lines.len()
    );
    for (i, line) in lines.iter().enumerate() {
        let v: serde_json::Value =
            serde_json::from_str(line).expect("each line is parseable NDJSON");
        assert_eq!(v["episode"], i as u64 + 1);
    }
}

#[test]
fn run_unknown_scenario_returns_nonzero_exit() {
    let output = assert_cmd::Command::cargo_bin("clankers-app")
        .expect("clankers-app binary built")
        .args(["run", "--scenario", "nope_not_a_real_one"])
        .output()
        .expect("failed to spawn clankers-app");

    assert!(!output.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown scenario"),
        "stderr should explain the failure: {stderr:?}"
    );
}
