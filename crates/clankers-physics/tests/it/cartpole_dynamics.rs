//! Integration test: verify raw rapier cart-pole dynamics.
//!
//! Creates a minimal cart-pole matching `OpenAI` Gym CartPole-v1 parameters
//! directly in rapier and checks that:
//! 1. Applying force to the cart causes it to accelerate
//! 2. The pole tilts when the cart accelerates (inertial effect)
//! 3. Gravity destabilizes the pole once tilted
//!
//! CartPole-v1 reference:
//!   Cart mass: 1.0 kg, Pole mass: 0.1 kg, Pole half-length: 0.5 m
//!   Force: 10 N, Gravity: 9.8 m/s², dt: 0.02 s, No damping

use bevy::prelude::Vec3;
use rapier3d::prelude::*;

/// Gravity matching CartPole-v1 (9.8, not 9.81)
const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -9.8);

/// All rapier physics state for a cart-pole simulation.
struct CartpoleWorld {
    bodies: RigidBodySet,
    joints: ImpulseJointSet,
    colliders: ColliderSet,
    multibody_joints: MultibodyJointSet,
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    ccd: CCDSolver,
    cart: RigidBodyHandle,
    pole: RigidBodyHandle,
    cart_joint: ImpulseJointHandle,
    pole_joint: ImpulseJointHandle,
}

impl CartpoleWorld {
    fn step(&mut self, gravity: Vec3, params: &IntegrationParameters) {
        self.pipeline.step(
            gravity,
            params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            &(),
            &(),
        );
    }
}

fn build_cartpole() -> CartpoleWorld {
    let mut rigid_body_set = RigidBodySet::new();
    let collider_set = ColliderSet::new();
    let mut impulse_joint_set = ImpulseJointSet::new();
    let multibody_joint_set = MultibodyJointSet::new();

    // Rail: fixed at origin
    let rail = rigid_body_set.insert(RigidBodyBuilder::fixed().build());

    // Cart: dynamic at (0, 0, 0.035), mass 1.0 kg (CartPole-v1 standard)
    let cart = rigid_body_set.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vec3::new(0.0, 0.0, 0.035))
            .can_sleep(false)
            .additional_mass_properties(MassProperties::new(
                Vec3::ZERO,
                1.0,
                Vec3::new(0.01, 0.01, 0.01),
            ))
            .build(),
    );

    // Prismatic joint: cart slides along X, limits ±2.4m (CartPole-v1 standard)
    let mut cart_joint: GenericJoint = PrismaticJointBuilder::new(Vec3::X)
        .local_anchor1(Vec3::new(0.0, 0.0, 0.035))
        .limits([-2.4, 2.4])
        .build()
        .into();
    cart_joint.set_motor_model(JointAxis::LinX, MotorModel::ForceBased);
    cart_joint.set_motor(JointAxis::LinX, 0.0, 0.0, 0.0, 0.0);
    let cart_jh = impulse_joint_set.insert(rail, cart, cart_joint, true);

    // Pole: dynamic, mass 0.1 kg, COM at 0.5m from pivot (CartPole-v1: half-length = 0.5m)
    // Inertia of uniform rod about COM: I = mL²/12 = 0.1 * 1.0² / 12 = 0.00833
    let pole = rigid_body_set.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vec3::new(0.0, 0.0, 0.06))
            .can_sleep(false)
            .additional_mass_properties(MassProperties::new(
                Vec3::new(0.0, 0.0, 0.5),
                0.1,
                Vec3::new(0.00833, 0.00833, 0.0001),
            ))
            .build(),
    );

    // Revolute joint: pole rotates around Y, no limits (continuous), no damping
    let mut pole_joint: GenericJoint = RevoluteJointBuilder::new(Vec3::Y)
        .local_anchor1(Vec3::new(0.0, 0.0, 0.025))
        .build()
        .into();
    pole_joint.set_motor_model(JointAxis::AngX, MotorModel::ForceBased);
    pole_joint.set_motor(JointAxis::AngX, 0.0, 0.0, 0.0, 0.0);
    let pole_jh = impulse_joint_set.insert(cart, pole, pole_joint, true);

    CartpoleWorld {
        bodies: rigid_body_set,
        joints: impulse_joint_set,
        colliders: collider_set,
        multibody_joints: multibody_joint_set,
        pipeline: PhysicsPipeline::new(),
        islands: IslandManager::new(),
        broad_phase: DefaultBroadPhase::new(),
        narrow_phase: NarrowPhase::new(),
        ccd: CCDSolver::new(),
        cart,
        pole,
        cart_joint: cart_jh,
        pole_joint: pole_jh,
    }
}

fn pole_angle(bodies: &RigidBodySet, cart: RigidBodyHandle, pole: RigidBodyHandle) -> f32 {
    let parent_rot = bodies[cart].position().rotation;
    let child_rot = bodies[pole].position().rotation;
    let rel = parent_rot.inverse() * child_rot;
    let axis = Vec3::Y;
    let sin_half = Vec3::new(rel.x, rel.y, rel.z);
    let sin_half_proj = sin_half.dot(axis);
    2.0 * f32::atan2(sin_half_proj, rel.w)
}

#[test]
fn cart_force_moves_cart() {
    let mut world = build_cartpole();

    let params = IntegrationParameters {
        dt: 0.001,
        ..Default::default()
    };

    // Apply 10N force to cart via motor trick (CartPole-v1 force magnitude)
    if let Some(joint) = world.joints.get_mut(world.cart_joint, true) {
        joint.data.set_motor(JointAxis::LinX, 0.0, 1e10, 0.0, 1.0);
        joint.data.set_motor_max_force(JointAxis::LinX, 10.0);
    }

    // Step 20 substeps (= 0.02s, one CartPole-v1 control step)
    for _ in 0..20 {
        world.step(GRAVITY, &params);
    }

    let cart_pos = world.bodies[world.cart].position().translation.x;
    assert!(cart_pos > 0.001, "cart should have moved: pos={cart_pos}");
}

#[test]
fn pole_tilts_from_cart_acceleration() {
    let mut world = build_cartpole();

    let params = IntegrationParameters {
        dt: 0.001,
        ..Default::default()
    };

    // Ensure pole motor is fully disabled (free DOF)
    if let Some(joint) = world.joints.get_mut(world.pole_joint, true) {
        joint.data.set_motor(JointAxis::AngX, 0.0, 0.0, 0.0, 0.0);
        joint.data.set_motor_max_force(JointAxis::AngX, 0.0);
    }

    let angle_before = pole_angle(&world.bodies, world.cart, world.pole);

    // Apply 10N force to cart (CartPole-v1 force magnitude)
    if let Some(joint) = world.joints.get_mut(world.cart_joint, true) {
        joint.data.set_motor(JointAxis::LinX, 0.0, 1e10, 0.0, 1.0);
        joint.data.set_motor_max_force(JointAxis::LinX, 10.0);
    }

    // Step 100 substeps (= 0.1s = 5 CartPole-v1 control steps)
    for _ in 0..100 {
        world.step(GRAVITY, &params);
    }

    let angle_after = pole_angle(&world.bodies, world.cart, world.pole);
    let delta = (angle_after - angle_before).abs();
    eprintln!("pole angle: before={angle_before}, after={angle_after}, delta={delta}");
    assert!(
        delta > 0.001,
        "pole should tilt from cart acceleration: delta={delta}"
    );
}

#[test]
fn pole_falls_under_gravity_when_perturbed() {
    let mut world = build_cartpole();

    let params = IntegrationParameters {
        dt: 0.001,
        ..Default::default()
    };

    // Disable pole motor
    if let Some(joint) = world.joints.get_mut(world.pole_joint, true) {
        joint.data.set_motor(JointAxis::AngX, 0.0, 0.0, 0.0, 0.0);
        joint.data.set_motor_max_force(JointAxis::AngX, 0.0);
    }

    // Give pole a larger initial angular velocity (perturbation)
    world.bodies[world.pole].set_angvel(Vec3::new(0.0, 0.5, 0.0), true);

    // Step 1000 substeps (= 1.0s)
    for _ in 0..1000 {
        world.step(GRAVITY, &params);
    }

    let angle = pole_angle(&world.bodies, world.cart, world.pole).abs();
    eprintln!("pole angle after 1.0s with perturbation: {angle}");
    assert!(
        angle > 0.05,
        "pole should have fallen under gravity: angle={angle}"
    );
}

#[test]
fn pole_mass_is_nonzero() {
    let mut world = build_cartpole();

    // Step once so rapier recomputes mass properties from additional_mass_properties
    let params = IntegrationParameters {
        dt: 0.001,
        ..Default::default()
    };
    world.step(GRAVITY, &params);

    let pole_body = &world.bodies[world.pole];
    let eff_inv_mass = pole_body.mass_properties().effective_inv_mass;
    eprintln!("pole effective_inv_mass: {eff_inv_mass:?}");
    // effective_inv_mass is a Vector3; any non-zero component means the body has mass
    assert!(
        eff_inv_mass.x > 0.0 || eff_inv_mass.y > 0.0 || eff_inv_mass.z > 0.0,
        "pole should have nonzero effective inverse mass: {eff_inv_mass:?}"
    );
}
