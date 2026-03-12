//! Diagnostic test: why does the arm collapse?

use clankers_env::prelude::*;
use clankers_examples::arm_setup::{ArmSetupConfig, initial_motor_overrides, setup_arm};
use clankers_physics::rapier::{MotorOverrides, RapierContext};

fn run_arm(steps: usize, use_overrides: bool, stiffness: f32, remove_colliders: bool) -> f32 {
    let setup = setup_arm(ArmSetupConfig {
        max_episode_steps: steps as u32 + 10,
        use_fixed_update: false,
        sensor_dof: 8,
        ..ArmSetupConfig::default()
    });

    let spawned = &setup.scene.robots["six_dof_arm"];
    let fl = spawned.joint_entity("j_finger_left").unwrap();
    let fr = spawned.joint_entity("j_finger_right").unwrap();

    let overrides = if use_overrides {
        let mut ov = initial_motor_overrides(&setup, &[fl, fr]);
        for params in ov.joints.values_mut() {
            params.stiffness = stiffness;
        }
        ov
    } else {
        MotorOverrides::default()
    };

    let mut scene = setup.scene;
    scene.app.insert_resource(overrides);

    // Optionally remove ALL colliders to isolate constraint issues
    if remove_colliders {
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        // Remove all colliders
        let handles: Vec<_> = ctx.collider_set.iter().map(|(h, _)| h).collect();
        #[allow(clippy::default_trait_access)]
        for h in handles {
            ctx.collider_set
                .remove(h, &mut Default::default(), &mut ctx.rigid_body_set, true);
        }
        world.insert_resource(ctx);
    }

    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    // Print initial EE position
    {
        let ctx = scene.app.world().resource::<RapierContext>();
        let ee_h = ctx.body_handles.get("end_effector").unwrap();
        let ee = ctx.rigid_body_set.get(*ee_h).unwrap();
        let t = ee.translation();
        println!("  Initial EE: [{:.4}, {:.4}, {:.4}]", t.x, t.y, t.z);
    }

    for step in 0..steps {
        scene.app.update();
        if step < 5 || step % 50 == 0 {
            let ctx = scene.app.world().resource::<RapierContext>();
            let ee_h = ctx.body_handles.get("end_effector").unwrap();
            let ee = ctx.rigid_body_set.get(*ee_h).unwrap();
            let t = ee.translation();
            println!(
                "  step {:>3}: EE=[{:.4}, {:.4}, {:.4}]",
                step, t.x, t.y, t.z
            );
        }
    }

    let ctx = scene.app.world().resource::<RapierContext>();
    let ee_h = ctx.body_handles.get("end_effector").unwrap();
    ctx.rigid_body_set.get(*ee_h).unwrap().translation().z
}

#[test]
fn diagnose_arm_collapse() {
    // Use default stiffness from ARM_STIFFNESS/ARM_DAMPING (50000/500)
    println!("\n=== Default gains (stiffness=50000) ===");
    let z = run_arm(500, true, 50_000.0, false);
    println!("  Final EE Z = {z:.4}\n");

    // Arm should hold near REST_POSE (initial EE Z ≈ 0.19)
    // With very high stiffness, droop should be minimal
    assert!(
        z > 0.10,
        "arm should hold near rest pose (Z≈0.19), got EE Z = {z:.4}"
    );
}
