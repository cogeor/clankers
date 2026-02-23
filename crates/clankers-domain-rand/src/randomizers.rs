//! Concrete randomizer types for actuator and physics parameters.
//!
//! Each randomizer holds optional [`RandomizationRange`] fields for parameters
//! it can randomize. Call the `randomize` method with a mutable reference to the target
//! component and an RNG.

use crate::ranges::RandomizationRange;
use clankers_actuator::components::Actuator;
use clankers_actuator_core::friction::FrictionModel;
use clankers_actuator_core::motor::{DcMotor, FullDcMotor, IdealMotor, MotorType};
use clankers_actuator_core::transmission::Transmission;
use rand::Rng;

// ---------------------------------------------------------------------------
// MotorRandomizer
// ---------------------------------------------------------------------------

/// Randomizes motor parameters. Applies to the active motor variant.
#[derive(Clone, Debug, Default)]
pub struct MotorRandomizer {
    pub max_torque: Option<RandomizationRange>,
    pub max_velocity: Option<RandomizationRange>,
    pub stall_torque: Option<RandomizationRange>,
    pub no_load_speed: Option<RandomizationRange>,
    pub time_constant: Option<RandomizationRange>,
    pub resistance: Option<RandomizationRange>,
    pub inductance: Option<RandomizationRange>,
    pub back_emf_constant: Option<RandomizationRange>,
    pub torque_constant: Option<RandomizationRange>,
    pub max_voltage: Option<RandomizationRange>,
    pub max_current: Option<RandomizationRange>,
}

impl MotorRandomizer {
    /// Apply randomization to a motor type in-place.
    pub fn randomize<R: Rng + ?Sized>(&self, motor: &mut MotorType, rng: &mut R) {
        match motor {
            MotorType::Ideal(m) => self.randomize_ideal(m, rng),
            MotorType::Dc(m) => self.randomize_dc(m, rng),
            MotorType::FullDc(m) => self.randomize_full_dc(m, rng),
        }
    }

    fn randomize_ideal<R: Rng + ?Sized>(&self, motor: &mut IdealMotor, rng: &mut R) {
        if let Some(r) = &self.max_torque {
            motor.max_torque = r.sample(rng);
        }
        if let Some(r) = &self.max_velocity {
            motor.max_velocity = r.sample(rng);
        }
    }

    fn randomize_dc<R: Rng + ?Sized>(&self, motor: &mut DcMotor, rng: &mut R) {
        if let Some(r) = &self.stall_torque {
            motor.stall_torque = r.sample(rng);
        }
        if let Some(r) = &self.no_load_speed {
            motor.no_load_speed = r.sample(rng);
        }
        if let Some(r) = &self.time_constant {
            motor.time_constant = r.sample(rng);
        }
        if let Some(r) = &self.max_torque {
            motor.max_torque = r.sample(rng);
        }
        if let Some(r) = &self.max_velocity {
            motor.max_velocity = r.sample(rng);
        }
    }

    fn randomize_full_dc<R: Rng + ?Sized>(&self, motor: &mut FullDcMotor, rng: &mut R) {
        if let Some(r) = &self.resistance {
            motor.resistance = r.sample(rng);
        }
        if let Some(r) = &self.inductance {
            motor.inductance = r.sample(rng);
        }
        if let Some(r) = &self.back_emf_constant {
            motor.back_emf_constant = r.sample(rng);
        }
        if let Some(r) = &self.torque_constant {
            motor.torque_constant = r.sample(rng);
        }
        if let Some(r) = &self.max_voltage {
            motor.max_voltage = r.sample(rng);
        }
        if let Some(r) = &self.max_current {
            motor.max_current = r.sample(rng);
        }
    }
}

// ---------------------------------------------------------------------------
// TransmissionRandomizer
// ---------------------------------------------------------------------------

/// Randomizes transmission parameters.
#[derive(Clone, Debug, Default)]
pub struct TransmissionRandomizer {
    pub gear_ratio: Option<RandomizationRange>,
    pub efficiency: Option<RandomizationRange>,
    pub backlash: Option<RandomizationRange>,
}

impl TransmissionRandomizer {
    /// Apply randomization to a transmission in-place.
    pub fn randomize<R: Rng + ?Sized>(&self, trans: &mut Transmission, rng: &mut R) {
        if let Some(r) = &self.gear_ratio {
            trans.gear_ratio = r.sample(rng);
        }
        if let Some(r) = &self.efficiency {
            trans.efficiency = r.sample(rng).clamp(0.0, 1.0);
        }
        if let Some(r) = &self.backlash {
            trans.backlash = r.sample(rng).max(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// FrictionRandomizer
// ---------------------------------------------------------------------------

/// Randomizes friction model parameters.
#[derive(Clone, Debug, Default)]
pub struct FrictionRandomizer {
    pub coulomb: Option<RandomizationRange>,
    pub viscous: Option<RandomizationRange>,
    pub stiction: Option<RandomizationRange>,
    pub stiction_velocity: Option<RandomizationRange>,
}

impl FrictionRandomizer {
    /// Apply randomization to a friction model in-place.
    pub fn randomize<R: Rng + ?Sized>(&self, friction: &mut FrictionModel, rng: &mut R) {
        if let Some(r) = &self.coulomb {
            friction.coulomb = r.sample(rng).max(0.0);
        }
        if let Some(r) = &self.viscous {
            friction.viscous = r.sample(rng).max(0.0);
        }
        if let Some(r) = &self.stiction {
            friction.stiction = r.sample(rng).max(0.0);
        }
        if let Some(r) = &self.stiction_velocity {
            friction.stiction_velocity = r.sample(rng).max(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// ActuatorRandomizer
// ---------------------------------------------------------------------------

/// Composite randomizer for an [`Actuator`] component.
///
/// Groups motor, transmission, and friction randomization.
#[derive(Clone, Debug, Default)]
pub struct ActuatorRandomizer {
    pub motor: MotorRandomizer,
    pub transmission: TransmissionRandomizer,
    pub friction: FrictionRandomizer,
}

impl ActuatorRandomizer {
    /// Apply randomization to an actuator component in-place.
    pub fn randomize_actuator<R: Rng + ?Sized>(&self, actuator: &mut Actuator, rng: &mut R) {
        self.motor.randomize(&mut actuator.motor, rng);
        self.transmission.randomize(&mut actuator.transmission, rng);
        self.friction.randomize(&mut actuator.friction, rng);
    }
}

// ---------------------------------------------------------------------------
// MassRandomizer
// ---------------------------------------------------------------------------

/// Randomizes the [`Mass`](clankers_core::physics::Mass) component on entities.
#[derive(Clone, Debug, Default)]
pub struct MassRandomizer {
    /// Range for the mass value in kg.
    pub mass: Option<RandomizationRange>,
}

impl MassRandomizer {
    /// Apply randomization to a mass component in-place.
    pub fn randomize<R: Rng + ?Sized>(&self, mass: &mut clankers_core::physics::Mass, rng: &mut R) {
        if let Some(r) = &self.mass {
            mass.0 = r.sample(rng).max(0.001); // mass must be positive
        }
    }
}

// ---------------------------------------------------------------------------
// SurfaceFrictionRandomizer
// ---------------------------------------------------------------------------

/// Randomizes [`SurfaceFriction`](clankers_core::physics::SurfaceFriction) components.
#[derive(Clone, Debug, Default)]
pub struct SurfaceFrictionRandomizer {
    /// Range for static friction coefficient.
    pub static_friction: Option<RandomizationRange>,
    /// Range for dynamic friction coefficient.
    pub dynamic_friction: Option<RandomizationRange>,
}

impl SurfaceFrictionRandomizer {
    /// Apply randomization to a surface friction component in-place.
    pub fn randomize<R: Rng + ?Sized>(
        &self,
        friction: &mut clankers_core::physics::SurfaceFriction,
        rng: &mut R,
    ) {
        if let Some(r) = &self.static_friction {
            friction.static_friction = r.sample(rng).max(0.0);
        }
        if let Some(r) = &self.dynamic_friction {
            friction.dynamic_friction = r.sample(rng).max(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// GeometryRandomizer
// ---------------------------------------------------------------------------

/// Randomizes entity geometry by perturbing `Transform::scale`.
///
/// Applies a uniform scale factor to the entity's transform, simulating
/// manufacturing tolerances or calibration errors.
#[derive(Clone, Debug, Default)]
pub struct GeometryRandomizer {
    /// Range for the uniform scale multiplier (e.g., `Scaling { nominal: 1.0, fraction: 0.05 }`).
    pub scale: Option<RandomizationRange>,
}

impl GeometryRandomizer {
    /// Apply randomization to a transform's scale in-place.
    pub fn randomize<R: Rng + ?Sized>(
        &self,
        transform: &mut bevy::prelude::Transform,
        rng: &mut R,
    ) {
        if let Some(r) = &self.scale {
            let s = r.sample(rng).max(0.01); // scale must be positive
            transform.scale = bevy::prelude::Vec3::splat(s);
        }
    }
}

// ---------------------------------------------------------------------------
// ExternalForceRandomizer
// ---------------------------------------------------------------------------

/// Randomizes [`ExternalForce`](clankers_core::physics::ExternalForce) components.
///
/// Adds random perturbation forces/torques to simulate wind, vibration,
/// or unmodeled dynamics.
#[derive(Clone, Debug, Default)]
pub struct ExternalForceRandomizer {
    /// Range for force magnitude (Newtons). Applied in random direction.
    pub force_magnitude: Option<RandomizationRange>,
    /// Range for torque magnitude (Newton-meters). Applied in random direction.
    pub torque_magnitude: Option<RandomizationRange>,
}

impl ExternalForceRandomizer {
    /// Apply randomization to an external force component in-place.
    ///
    /// Generates a random 3D direction and scales it by the sampled magnitude.
    pub fn randomize<R: Rng + ?Sized>(
        &self,
        ext_force: &mut clankers_core::physics::ExternalForce,
        rng: &mut R,
    ) {
        if let Some(r) = &self.force_magnitude {
            let mag = r.sample(rng);
            let dir = random_unit_vec3(rng);
            ext_force.force = dir * mag;
        }
        if let Some(r) = &self.torque_magnitude {
            let mag = r.sample(rng);
            let dir = random_unit_vec3(rng);
            ext_force.torque = dir * mag;
        }
    }
}

/// Generate a random unit vector on the sphere using rejection sampling.
fn random_unit_vec3<R: Rng + ?Sized>(rng: &mut R) -> bevy::prelude::Vec3 {
    loop {
        let x = rng.gen_range(-1.0_f32..1.0);
        let y = rng.gen_range(-1.0_f32..1.0);
        let z = rng.gen_range(-1.0_f32..1.0);
        let len_sq = z.mul_add(z, y.mul_add(y, x * x));
        if len_sq > f32::EPSILON && len_sq <= 1.0 {
            let len = len_sq.sqrt();
            return bevy::prelude::Vec3::new(x / len, y / len, z / len);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ranges::RandomizationRange;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    // -- MotorRandomizer --

    #[test]
    fn motor_randomizer_ideal() {
        let randomizer = MotorRandomizer {
            max_torque: Some(RandomizationRange::uniform(5.0, 15.0).unwrap()),
            max_velocity: Some(RandomizationRange::uniform(50.0, 150.0).unwrap()),
            ..Default::default()
        };

        let mut motor = MotorType::Ideal(IdealMotor {
            max_torque: 10.0,
            max_velocity: 100.0,
        });
        let mut rng = rng();
        randomizer.randomize(&mut motor, &mut rng);

        if let MotorType::Ideal(m) = &motor {
            assert!(m.max_torque >= 5.0 && m.max_torque < 15.0);
            assert!(m.max_velocity >= 50.0 && m.max_velocity < 150.0);
        } else {
            panic!("expected Ideal motor");
        }
    }

    #[test]
    fn motor_randomizer_dc() {
        let randomizer = MotorRandomizer {
            stall_torque: Some(RandomizationRange::scaling(2.0, 0.1).unwrap()),
            time_constant: Some(RandomizationRange::uniform(0.01, 0.05).unwrap()),
            ..Default::default()
        };

        let mut motor = MotorType::Dc(
            DcMotor::new(2.0, 100.0)
                .with_time_constant(0.02)
                .with_torque_limit(3.0),
        );
        let mut rng = rng();
        randomizer.randomize(&mut motor, &mut rng);

        if let MotorType::Dc(m) = &motor {
            assert!(m.stall_torque >= 1.8 && m.stall_torque <= 2.2);
            assert!(m.time_constant >= 0.01 && m.time_constant < 0.05);
            // Unrandomized fields unchanged
            assert!((m.no_load_speed - 100.0).abs() < f32::EPSILON);
        } else {
            panic!("expected Dc motor");
        }
    }

    #[test]
    fn motor_randomizer_full_dc() {
        let randomizer = MotorRandomizer {
            resistance: Some(RandomizationRange::uniform(0.5, 2.0).unwrap()),
            ..Default::default()
        };

        let mut motor = MotorType::FullDc(FullDcMotor::from_specs(1.0, 0.001, 0.01, 24.0, 10.0));
        let mut rng = rng();
        randomizer.randomize(&mut motor, &mut rng);

        if let MotorType::FullDc(m) = &motor {
            assert!(m.resistance >= 0.5 && m.resistance < 2.0);
            assert!((m.inductance - 0.001).abs() < f32::EPSILON);
        } else {
            panic!("expected FullDc motor");
        }
    }

    #[test]
    fn motor_randomizer_none_leaves_unchanged() {
        let randomizer = MotorRandomizer::default();
        let mut motor = MotorType::Ideal(IdealMotor {
            max_torque: 10.0,
            max_velocity: 100.0,
        });
        let mut rng = rng();
        randomizer.randomize(&mut motor, &mut rng);

        if let MotorType::Ideal(m) = &motor {
            assert!((m.max_torque - 10.0).abs() < f32::EPSILON);
            assert!((m.max_velocity - 100.0).abs() < f32::EPSILON);
        } else {
            panic!("expected Ideal motor");
        }
    }

    // -- TransmissionRandomizer --

    #[test]
    fn transmission_randomizer_applies() {
        let randomizer = TransmissionRandomizer {
            gear_ratio: Some(RandomizationRange::uniform(90.0, 110.0).unwrap()),
            efficiency: Some(RandomizationRange::uniform(0.8, 1.0).unwrap()),
            backlash: Some(RandomizationRange::uniform(0.0, 0.01).unwrap()),
        };

        let mut trans = Transmission::new(100.0);
        let mut rng = rng();
        randomizer.randomize(&mut trans, &mut rng);

        assert!(trans.gear_ratio >= 90.0 && trans.gear_ratio < 110.0);
        assert!(trans.efficiency >= 0.8 && trans.efficiency <= 1.0);
        assert!(trans.backlash >= 0.0 && trans.backlash < 0.01);
    }

    #[test]
    fn transmission_efficiency_clamped() {
        let randomizer = TransmissionRandomizer {
            efficiency: Some(RandomizationRange::uniform(1.5, 2.0).unwrap()),
            ..Default::default()
        };

        let mut trans = Transmission::new(100.0);
        let mut rng = rng();
        randomizer.randomize(&mut trans, &mut rng);
        assert!(trans.efficiency <= 1.0);
    }

    // -- FrictionRandomizer --

    #[test]
    fn friction_randomizer_applies() {
        let randomizer = FrictionRandomizer {
            coulomb: Some(RandomizationRange::uniform(0.1, 0.5).unwrap()),
            viscous: Some(RandomizationRange::uniform(0.01, 0.1).unwrap()),
            ..Default::default()
        };

        let mut friction = FrictionModel::default();
        let mut rng = rng();
        randomizer.randomize(&mut friction, &mut rng);

        assert!(friction.coulomb >= 0.1 && friction.coulomb < 0.5);
        assert!(friction.viscous >= 0.01 && friction.viscous < 0.1);
    }

    #[test]
    fn friction_values_non_negative() {
        let randomizer = FrictionRandomizer {
            coulomb: Some(RandomizationRange::uniform(-1.0, 0.1).unwrap()),
            ..Default::default()
        };

        let mut friction = FrictionModel::default();
        let mut rng = rng();
        randomizer.randomize(&mut friction, &mut rng);
        assert!(friction.coulomb >= 0.0);
    }

    // -- ActuatorRandomizer --

    #[test]
    fn actuator_randomizer_applies_all() {
        let randomizer = ActuatorRandomizer {
            motor: MotorRandomizer {
                max_torque: Some(RandomizationRange::scaling(10.0, 0.2).unwrap()),
                ..Default::default()
            },
            transmission: TransmissionRandomizer {
                gear_ratio: Some(RandomizationRange::scaling(100.0, 0.05).unwrap()),
                ..Default::default()
            },
            friction: FrictionRandomizer {
                coulomb: Some(RandomizationRange::uniform(0.1, 0.5).unwrap()),
                ..Default::default()
            },
        };

        let mut actuator = Actuator::default();
        let mut rng = rng();

        randomizer.randomize_actuator(&mut actuator, &mut rng);

        if let MotorType::Ideal(m) = &actuator.motor {
            assert!(m.max_torque >= 8.0 && m.max_torque <= 12.0);
        }
        assert!(
            actuator.transmission.gear_ratio >= 95.0 && actuator.transmission.gear_ratio <= 105.0
        );
        assert!(actuator.friction.coulomb >= 0.1 && actuator.friction.coulomb < 0.5);
    }

    // -- Determinism --

    #[test]
    fn randomizer_deterministic_with_same_seed() {
        let randomizer = MotorRandomizer {
            max_torque: Some(RandomizationRange::uniform(5.0, 15.0).unwrap()),
            ..Default::default()
        };

        let mut m1 = MotorType::Ideal(IdealMotor {
            max_torque: 10.0,
            max_velocity: 100.0,
        });
        let mut m2 = m1.clone();

        let mut rng1 = ChaCha8Rng::seed_from_u64(99);
        let mut rng2 = ChaCha8Rng::seed_from_u64(99);

        randomizer.randomize(&mut m1, &mut rng1);
        randomizer.randomize(&mut m2, &mut rng2);

        if let (MotorType::Ideal(a), MotorType::Ideal(b)) = (&m1, &m2) {
            assert!((a.max_torque - b.max_torque).abs() < f32::EPSILON);
        }
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn randomizer_types_are_send_sync() {
        assert_send_sync::<MotorRandomizer>();
        assert_send_sync::<TransmissionRandomizer>();
        assert_send_sync::<FrictionRandomizer>();
        assert_send_sync::<ActuatorRandomizer>();
        assert_send_sync::<MassRandomizer>();
        assert_send_sync::<SurfaceFrictionRandomizer>();
        assert_send_sync::<GeometryRandomizer>();
        assert_send_sync::<ExternalForceRandomizer>();
    }

    // -- MassRandomizer --

    #[test]
    fn mass_randomizer_applies() {
        let randomizer = MassRandomizer {
            mass: Some(RandomizationRange::uniform(2.0, 10.0).unwrap()),
        };
        let mut mass = clankers_core::physics::Mass::new(5.0);
        let mut rng = rng();
        randomizer.randomize(&mut mass, &mut rng);
        assert!(mass.kg() >= 2.0 && mass.kg() < 10.0);
    }

    #[test]
    fn mass_randomizer_clamps_positive() {
        let randomizer = MassRandomizer {
            mass: Some(RandomizationRange::uniform(-1.0, 0.0001).unwrap()),
        };
        let mut mass = clankers_core::physics::Mass::new(1.0);
        let mut rng = rng();
        randomizer.randomize(&mut mass, &mut rng);
        assert!(mass.kg() >= 0.001);
    }

    #[test]
    fn mass_randomizer_none_leaves_unchanged() {
        let randomizer = MassRandomizer::default();
        let mut mass = clankers_core::physics::Mass::new(5.0);
        let mut rng = rng();
        randomizer.randomize(&mut mass, &mut rng);
        assert!((mass.kg() - 5.0).abs() < f32::EPSILON);
    }

    // -- SurfaceFrictionRandomizer --

    #[test]
    fn surface_friction_randomizer_applies() {
        let randomizer = SurfaceFrictionRandomizer {
            static_friction: Some(RandomizationRange::uniform(0.2, 0.8).unwrap()),
            dynamic_friction: Some(RandomizationRange::uniform(0.1, 0.5).unwrap()),
        };
        let mut fric = clankers_core::physics::SurfaceFriction::default();
        let mut rng = rng();
        randomizer.randomize(&mut fric, &mut rng);
        assert!(fric.static_friction >= 0.2 && fric.static_friction < 0.8);
        assert!(fric.dynamic_friction >= 0.1 && fric.dynamic_friction < 0.5);
    }

    #[test]
    fn surface_friction_non_negative() {
        let randomizer = SurfaceFrictionRandomizer {
            static_friction: Some(RandomizationRange::uniform(-1.0, 0.1).unwrap()),
            ..Default::default()
        };
        let mut fric = clankers_core::physics::SurfaceFriction::default();
        let mut rng = rng();
        randomizer.randomize(&mut fric, &mut rng);
        assert!(fric.static_friction >= 0.0);
    }

    // -- GeometryRandomizer --

    #[test]
    fn geometry_randomizer_applies_scale() {
        let randomizer = GeometryRandomizer {
            scale: Some(RandomizationRange::scaling(1.0, 0.1).unwrap()),
        };
        let mut transform = bevy::prelude::Transform::default();
        let mut rng = rng();
        randomizer.randomize(&mut transform, &mut rng);
        // Scale should be near 1.0 ± 10%
        let s = transform.scale.x;
        assert!((0.9..=1.1).contains(&s), "scale = {s}");
        assert!((transform.scale.x - transform.scale.y).abs() < f32::EPSILON);
        assert!((transform.scale.x - transform.scale.z).abs() < f32::EPSILON);
    }

    #[test]
    fn geometry_randomizer_positive_scale() {
        let randomizer = GeometryRandomizer {
            scale: Some(RandomizationRange::uniform(-1.0, 0.001).unwrap()),
        };
        let mut transform = bevy::prelude::Transform::default();
        let mut rng = rng();
        randomizer.randomize(&mut transform, &mut rng);
        assert!(transform.scale.x >= 0.01);
    }

    // -- ExternalForceRandomizer --

    #[test]
    fn external_force_randomizer_applies() {
        let randomizer = ExternalForceRandomizer {
            force_magnitude: Some(RandomizationRange::uniform(0.5, 2.0).unwrap()),
            torque_magnitude: Some(RandomizationRange::uniform(0.1, 0.5).unwrap()),
        };
        let mut ext = clankers_core::physics::ExternalForce::default();
        let mut rng = rng();
        randomizer.randomize(&mut ext, &mut rng);
        let force_mag = ext.force.length();
        let torque_mag = ext.torque.length();
        assert!(
            (0.5..=2.0).contains(&force_mag),
            "force_mag = {force_mag}"
        );
        assert!(
            (0.1..=0.5).contains(&torque_mag),
            "torque_mag = {torque_mag}"
        );
    }

    #[test]
    fn external_force_randomizer_none_leaves_zero() {
        let randomizer = ExternalForceRandomizer::default();
        let mut ext = clankers_core::physics::ExternalForce::default();
        let mut rng = rng();
        randomizer.randomize(&mut ext, &mut rng);
        assert_eq!(ext.force, bevy::prelude::Vec3::ZERO);
        assert_eq!(ext.torque, bevy::prelude::Vec3::ZERO);
    }

    #[test]
    fn random_unit_vec3_is_normalized() {
        let mut rng = rng();
        for _ in 0..100 {
            let v = super::random_unit_vec3(&mut rng);
            assert!((v.length() - 1.0).abs() < 0.01, "length = {}", v.length());
        }
    }
}
