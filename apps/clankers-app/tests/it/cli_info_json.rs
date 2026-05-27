//! Integration test for `clankers-app info --json`.

use assert_cmd::Command;

#[test]
fn info_json_includes_version_and_scenarios() {
    let output = Command::cargo_bin("clankers-app")
        .expect("binary built")
        .args(["info", "--json"])
        .output()
        .expect("info --json runs");
    assert!(output.status.success(), "exit status: {:?}", output.status);

    let v: serde_json::Value = serde_json::from_slice(&output.stdout).expect("parse JSON");

    // Schema (design choice D): the keys below are stable for the W5
    // workstream.
    assert!(v["version"].is_string(), "version key");
    assert!(v["protocol_version"].is_string(), "protocol_version key");
    assert!(v["build_profile"].is_string(), "build_profile key");
    assert!(v["edition"].is_string(), "edition key");
    assert!(v["crates"].is_array(), "crates array");
    assert!(v["scenarios"].is_array(), "scenarios array (empty in PR1)");
}
