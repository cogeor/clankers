//! Integration tests for `clankers-app validate`.
//!
//! The binary is built by cargo's test harness; we invoke it via
//! `assert_cmd::Command::cargo_bin("clankers-app")` so we exercise the
//! real `clap` dispatch path. GPU is not used — `validate` never opens
//! a window.

use assert_cmd::Command;
use predicates::str::contains;

const MINIMAL: &str = "tests/fixtures/minimal.urdf";
const CORRUPTED: &str = "tests/fixtures/corrupted.urdf";

#[test]
fn validate_corrupted_urdf_returns_nonzero_exit() {
    let mut cmd = Command::cargo_bin("clankers-app").expect("binary built");
    cmd.args(["validate", "--urdf", CORRUPTED]);
    cmd.assert().failure();
}

#[test]
fn validate_good_urdf_returns_zero_exit() {
    let mut cmd = Command::cargo_bin("clankers-app").expect("binary built");
    cmd.args(["validate", "--urdf", MINIMAL, "--json"]);
    cmd.assert()
        .success()
        .stdout(contains("\"status\": \"ok\""));
}

/// Design choice C: `validate --strict --urdf <path>` (no scenario) is
/// a no-op in PR1. It exits 0 with a one-line stderr warning.
#[test]
fn validate_strict_without_scenario_is_noop_warning() {
    let mut cmd = Command::cargo_bin("clankers-app").expect("binary built");
    cmd.args(["validate", "--urdf", MINIMAL, "--strict"]);
    cmd.assert()
        .success()
        .stderr(contains("--strict has no effect without --scenario"));
}
