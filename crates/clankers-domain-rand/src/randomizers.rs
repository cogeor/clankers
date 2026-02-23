//! Concrete randomizer types for actuator and physics parameters.
//!
//! Each randomizer holds optional [`RandomizationRange`] fields for parameters
//! it can randomize. Call the `randomize` method with a mutable reference to the target
//! component and an RNG.

use crate::ranges::RandomizationRange;
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
    /// Apply randomization to an actuator's motor, transmission, and friction.
    pub fn randomize<R: Rng + ?Sized>(
        &self,
        motor: &mut MotorType,
        transmission: &mut Transmission,
        friction: &mut FrictionModel,
        rng: &mut R,
    ) {
        self.motor.randomize(motor, rng);
        self.transmission.randomize(transmission, rng);
        self.friction.randomize(friction, rng);
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

        let mut motor = MotorType::Ideal(IdealMotor {
            max_torque: 10.0,
            max_velocity: 100.0,
        });
        let mut trans = Transmission::new(100.0);
        let mut friction = FrictionModel::default();
        let mut rng = rng();

        randomizer.randomize(&mut motor, &mut trans, &mut friction, &mut rng);

        if let MotorType::Ideal(m) = &motor {
            assert!(m.max_torque >= 8.0 && m.max_torque <= 12.0);
        }
        assert!(trans.gear_ratio >= 95.0 && trans.gear_ratio <= 105.0);
        assert!(friction.coulomb >= 0.1 && friction.coulomb < 0.5);
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
    }
}
