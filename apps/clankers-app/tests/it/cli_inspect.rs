//! Integration test for `clankers-app inspect urdf`.

use assert_cmd::Command;

const MINIMAL: &str = "tests/fixtures/minimal.urdf";

#[test]
fn inspect_urdf_prints_joint_layout() {
    let run = || {
        Command::cargo_bin("clankers-app")
            .expect("binary built")
            .args(["inspect", "urdf", MINIMAL, "--json"])
            .output()
            .expect("inspect urdf runs")
    };
    let a = run();
    let b = run();
    assert!(a.status.success(), "first run: {:?}", a.status);
    assert!(b.status.success(), "second run: {:?}", b.status);

    let va: serde_json::Value = serde_json::from_slice(&a.stdout).expect("parse a");
    let vb: serde_json::Value = serde_json::from_slice(&b.stdout).expect("parse b");

    let hash_a = va["joint_layout"]["hash"].as_str().expect("hash key");
    let hash_b = vb["joint_layout"]["hash"].as_str().expect("hash key");

    assert!(!hash_a.is_empty(), "hash is non-empty");
    assert_eq!(
        hash_a, hash_b,
        "JointLayout hash must be deterministic across invocations"
    );

    // Sanity: structural shape.
    assert!(va["joints"].is_array(), "joints array");
    assert_eq!(va["joint_layout"]["count"].as_u64(), Some(1));
}
