//! Integration tests for the built-in scenarios (W5 PR2, loop 6 gate
//! item 3).
//!
//! Verifies that:
//!
//! - The `arm_pick` and `cartpole` scenarios build into a headless
//!   Bevy `App` without panicking.
//! - The two scenarios are registered by [`register_builtin`].
//!
//! All tests run under the default bevy minimal-plugin headless
//! harness — no windows, no rendering, no GPU.

use bevy::prelude::App;
use clankers_sim::{
    ClankersSimPlugin, ScenarioBuilder, ScenarioConfig, ScenarioRegistry,
    scenarios::{arm_pick::ArmPickScenario, cartpole::CartpoleScenario, register_builtin},
};

#[test]
fn arm_pick_scenario_builds_without_panic() {
    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);
    let handle = ArmPickScenario.build(&mut app, &ScenarioConfig::default());
    assert!(handle.layout.is_some(), "arm_pick must expose a layout");
    let layout = handle.layout.as_ref().unwrap();
    // 6 arm + 2 gripper = 8 actuated joints.
    assert_eq!(layout.len(), 8, "arm_pick layout should have 8 joints");
    app.finish();
    app.cleanup();
    app.update();
}

#[test]
fn cartpole_scenario_builds_without_panic() {
    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);
    let handle = CartpoleScenario.build(&mut app, &ScenarioConfig::default());
    assert!(handle.layout.is_some(), "cartpole must expose a layout");
    let layout = handle.layout.as_ref().unwrap();
    // cart_slide (prismatic) + pole_hinge (continuous) = 2 joints.
    assert_eq!(layout.len(), 2, "cartpole layout should have 2 joints");
    app.finish();
    app.cleanup();
    app.update();
}

#[test]
fn registry_lists_builtin_scenarios() {
    let mut registry = ScenarioRegistry::new();
    register_builtin(&mut registry);
    let names = registry.list_builtin();
    // Loop 7 (W8 PR1) added the arm-family scenarios; loop 8 (W8 PR2)
    // adds quadruped/multi/pendulum/domain-rand. Order is alphabetic
    // per `ScenarioRegistry::list_builtin`.
    assert_eq!(
        names,
        vec![
            "arm_bench",
            "arm_ik",
            "arm_pick",
            "arm_two_link",
            "cartpole",
            "domain_rand_pendulum",
            "multi_robot",
            "pendulum",
        ]
    );
}
