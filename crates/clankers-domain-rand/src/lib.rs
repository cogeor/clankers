//! Physics parameter randomization for sim-to-real transfer.
//!
//! Domain randomization varies physical parameters (motor dynamics, friction,
//! transmission) on episode reset to improve policy robustness.
//!
//! # Example
//!
//! ```no_run
//! use bevy::prelude::*;
//! use clankers_domain_rand::prelude::*;
//!
//! App::new()
//!     .add_plugins(clankers_core::ClankersCorePlugin)
//!     .add_plugins(ClankersDomainRandPlugin)
//!     .run();
//! ```

pub mod randomizers;
pub mod ranges;

use bevy::prelude::*;
use clankers_actuator::components::Actuator;
use clankers_core::ClankersSet;
use clankers_core::seed::SeedHierarchy;
use clankers_env::episode::{Episode, EpisodeState};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::randomizers::{
    ActuatorRandomizer, ExternalForceRandomizer, GeometryRandomizer, MassRandomizer,
    SurfaceFrictionRandomizer,
};

// ---------------------------------------------------------------------------
// DomainRandConfig
// ---------------------------------------------------------------------------

/// Resource configuring domain randomization behavior.
#[derive(Resource, Clone, Debug)]
pub struct DomainRandConfig {
    /// Whether randomization is enabled.
    pub enabled: bool,
    /// Seed for the randomization RNG. Changed each episode.
    pub seed: u64,
    /// Actuator parameter randomizer.
    pub actuator: ActuatorRandomizer,
    /// Mass randomizer (for entities with [`Mass`](clankers_core::physics::Mass)).
    pub mass: MassRandomizer,
    /// Surface friction randomizer (for entities with [`SurfaceFriction`](clankers_core::physics::SurfaceFriction)).
    pub surface_friction: SurfaceFrictionRandomizer,
    /// Geometry scale randomizer (perturbs `Transform::scale`).
    pub geometry: GeometryRandomizer,
    /// External force randomizer (for entities with [`ExternalForce`](clankers_core::physics::ExternalForce)).
    pub external_force: ExternalForceRandomizer,
}

impl Default for DomainRandConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            seed: 0,
            actuator: ActuatorRandomizer::default(),
            mass: MassRandomizer::default(),
            surface_friction: SurfaceFrictionRandomizer::default(),
            geometry: GeometryRandomizer::default(),
            external_force: ExternalForceRandomizer::default(),
        }
    }
}

impl DomainRandConfig {
    /// Builder: set the actuator randomizer.
    #[must_use]
    pub const fn with_actuator(mut self, actuator: ActuatorRandomizer) -> Self {
        self.actuator = actuator;
        self
    }

    /// Builder: set the mass randomizer.
    #[must_use]
    pub const fn with_mass(mut self, mass: MassRandomizer) -> Self {
        self.mass = mass;
        self
    }

    /// Builder: set the surface friction randomizer.
    #[must_use]
    pub const fn with_surface_friction(mut self, sf: SurfaceFrictionRandomizer) -> Self {
        self.surface_friction = sf;
        self
    }

    /// Builder: set the geometry randomizer.
    #[must_use]
    pub const fn with_geometry(mut self, geometry: GeometryRandomizer) -> Self {
        self.geometry = geometry;
        self
    }

    /// Builder: set the external force randomizer.
    #[must_use]
    pub const fn with_external_force(mut self, ef: ExternalForceRandomizer) -> Self {
        self.external_force = ef;
        self
    }

    /// Builder: set initial seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Builder: enable or disable.
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// DomainRandState
// ---------------------------------------------------------------------------

/// Internal resource tracking the last episode number that was randomized.
#[derive(Resource, Default)]
struct DomainRandState {
    last_randomized_episode: u32,
}

// ---------------------------------------------------------------------------
// randomize_on_reset_system
// ---------------------------------------------------------------------------

/// System that randomizes physics parameters when a new episode starts.
///
/// Detects episode resets by comparing the current episode number against
/// the last randomized episode number. Randomizes actuators, mass,
/// surface friction, geometry scale, and external forces.
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
fn randomize_on_reset_system(
    config: Res<DomainRandConfig>,
    episode: Res<Episode>,
    mut state: ResMut<DomainRandState>,
    seed_hierarchy: Option<Res<SeedHierarchy>>,
    mut actuators: Query<&mut Actuator>,
    mut masses: Query<&mut clankers_core::physics::Mass>,
    mut surface_frictions: Query<&mut clankers_core::physics::SurfaceFriction>,
    mut geometry_transforms: Query<&mut Transform, With<clankers_core::physics::Mass>>,
    mut external_forces: Query<&mut clankers_core::physics::ExternalForce>,
) {
    if !config.enabled {
        return;
    }

    if episode.state != EpisodeState::Running {
        return;
    }

    if episode.episode_number <= state.last_randomized_episode {
        return;
    }

    state.last_randomized_episode = episode.episode_number;

    // Derive RNG seed: use SeedHierarchy when available and non-default,
    // otherwise fall back to config.seed + episode_number.
    let episode_seed = match &seed_hierarchy {
        Some(hierarchy) if hierarchy.root() != 0 => {
            hierarchy.subsystem_seed(0, u64::from(episode.episode_number), "domain_rand")
        }
        _ => config.seed.wrapping_add(u64::from(episode.episode_number)),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(episode_seed);

    for mut actuator in &mut actuators {
        config.actuator.randomize_actuator(&mut actuator, &mut rng);
    }

    for mut mass in &mut masses {
        config.mass.randomize(&mut mass, &mut rng);
    }

    for mut sf in &mut surface_frictions {
        config.surface_friction.randomize(&mut sf, &mut rng);
    }

    for mut transform in &mut geometry_transforms {
        config.geometry.randomize(&mut transform, &mut rng);
    }

    for mut ef in &mut external_forces {
        config.external_force.randomize(&mut ef, &mut rng);
    }
}

// ---------------------------------------------------------------------------
// ClankersDomainRandPlugin
// ---------------------------------------------------------------------------

/// Bevy plugin that adds domain randomization systems.
///
/// Randomizes actuator parameters on episode reset using configured ranges.
/// Runs in [`ClankersSet::Update`] (after episode state has been set).
pub struct ClankersDomainRandPlugin;

impl Plugin for ClankersDomainRandPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DomainRandConfig>()
            .init_resource::<DomainRandState>()
            .add_systems(
                Update,
                randomize_on_reset_system.in_set(ClankersSet::Update),
            );
    }
}

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::{
        ClankersDomainRandPlugin, DomainRandConfig,
        randomizers::{
            ActuatorRandomizer, ExternalForceRandomizer, FrictionRandomizer, GeometryRandomizer,
            MassRandomizer, MotorRandomizer, SurfaceFrictionRandomizer, TransmissionRandomizer,
        },
        ranges::{RandomizationRange, RangeError},
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};

    fn build_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(clankers_env::ClankersEnvPlugin);
        app.add_plugins(ClankersDomainRandPlugin);
        app.finish();
        app.cleanup();
        app
    }

    fn spawn_actuator(world: &mut World) {
        world.spawn((
            Actuator::default(),
            JointCommand::default(),
            JointState::default(),
            JointTorque::default(),
        ));
    }

    #[test]
    fn plugin_builds_without_panic() {
        let mut app = build_test_app();
        app.update();
        assert!(app.world().get_resource::<DomainRandConfig>().is_some());
    }

    #[test]
    fn randomization_skipped_when_disabled() {
        let mut app = build_test_app();
        spawn_actuator(app.world_mut());

        // Record the original max_torque before anything runs
        let original_max = {
            let a = app
                .world_mut()
                .query::<&Actuator>()
                .single(app.world())
                .unwrap();
            if let clankers_actuator_core::motor::MotorType::Ideal(m) = &a.motor {
                m.max_torque
            } else {
                panic!("expected Ideal motor");
            }
        };

        // Configure randomizer with large range but disabled
        app.world_mut().resource_mut::<DomainRandConfig>().enabled = false;
        app.world_mut().resource_mut::<DomainRandConfig>().actuator = ActuatorRandomizer {
            motor: randomizers::MotorRandomizer {
                max_torque: Some(ranges::RandomizationRange::uniform(100.0, 200.0).unwrap()),
                ..Default::default()
            },
            ..Default::default()
        };

        // Reset episode
        app.world_mut().resource_mut::<Episode>().reset(None);
        app.update();

        // Actuator should be unchanged
        let actuator = app
            .world_mut()
            .query::<&Actuator>()
            .single(app.world())
            .unwrap();
        if let clankers_actuator_core::motor::MotorType::Ideal(m) = &actuator.motor {
            assert!(
                (m.max_torque - original_max).abs() < f32::EPSILON,
                "expected unchanged, got {}",
                m.max_torque
            );
        }
    }

    #[test]
    fn randomization_applied_on_episode_reset() {
        let mut app = build_test_app();
        spawn_actuator(app.world_mut());

        app.world_mut().resource_mut::<DomainRandConfig>().actuator = ActuatorRandomizer {
            motor: randomizers::MotorRandomizer {
                max_torque: Some(ranges::RandomizationRange::uniform(100.0, 200.0).unwrap()),
                ..Default::default()
            },
            ..Default::default()
        };

        // Reset episode to trigger randomization
        app.world_mut().resource_mut::<Episode>().reset(None);
        app.update();

        let actuator = app
            .world_mut()
            .query::<&Actuator>()
            .single(app.world())
            .unwrap();
        if let clankers_actuator_core::motor::MotorType::Ideal(m) = &actuator.motor {
            assert!(
                m.max_torque >= 100.0 && m.max_torque < 200.0,
                "expected randomized max_torque in [100, 200), got {}",
                m.max_torque
            );
        } else {
            panic!("expected Ideal motor");
        }
    }

    #[test]
    fn randomization_deterministic_with_same_seed() {
        fn run_episode(seed: u64) -> f32 {
            let mut app = build_test_app();
            spawn_actuator(app.world_mut());

            let mut config = app.world_mut().resource_mut::<DomainRandConfig>();
            config.seed = seed;
            config.actuator = ActuatorRandomizer {
                motor: randomizers::MotorRandomizer {
                    max_torque: Some(ranges::RandomizationRange::uniform(1.0, 100.0).unwrap()),
                    ..Default::default()
                },
                ..Default::default()
            };

            app.world_mut().resource_mut::<Episode>().reset(None);
            app.update();

            let actuator = app
                .world_mut()
                .query::<&Actuator>()
                .single(app.world())
                .unwrap();
            if let clankers_actuator_core::motor::MotorType::Ideal(m) = &actuator.motor {
                m.max_torque
            } else {
                panic!("expected Ideal motor");
            }
        }

        let v1 = run_episode(42);
        let v2 = run_episode(42);
        let v3 = run_episode(99);

        assert!(
            (v1 - v2).abs() < f32::EPSILON,
            "same seed should give same result"
        );
        assert!(
            (v1 - v3).abs() > f32::EPSILON,
            "different seeds should differ"
        );
    }

    #[test]
    fn randomization_only_on_new_episodes() {
        let mut app = build_test_app();
        spawn_actuator(app.world_mut());

        app.world_mut().resource_mut::<DomainRandConfig>().actuator = ActuatorRandomizer {
            motor: randomizers::MotorRandomizer {
                max_torque: Some(ranges::RandomizationRange::uniform(100.0, 200.0).unwrap()),
                ..Default::default()
            },
            ..Default::default()
        };

        // First reset
        app.world_mut().resource_mut::<Episode>().reset(None);
        app.update();

        let first_val = {
            let actuator = app
                .world_mut()
                .query::<&Actuator>()
                .single(app.world())
                .unwrap();
            if let clankers_actuator_core::motor::MotorType::Ideal(m) = &actuator.motor {
                m.max_torque
            } else {
                panic!("expected Ideal motor");
            }
        };

        // Subsequent updates without reset should NOT re-randomize
        app.update();
        app.update();

        let after_val = {
            let actuator = app
                .world_mut()
                .query::<&Actuator>()
                .single(app.world())
                .unwrap();
            if let clankers_actuator_core::motor::MotorType::Ideal(m) = &actuator.motor {
                m.max_torque
            } else {
                panic!("expected Ideal motor");
            }
        };

        assert!(
            (first_val - after_val).abs() < f32::EPSILON,
            "should not re-randomize without new episode"
        );
    }

    #[test]
    fn multiple_actuators_randomized_differently() {
        let mut app = build_test_app();
        spawn_actuator(app.world_mut());
        spawn_actuator(app.world_mut());

        app.world_mut().resource_mut::<DomainRandConfig>().actuator = ActuatorRandomizer {
            motor: randomizers::MotorRandomizer {
                max_torque: Some(ranges::RandomizationRange::uniform(1.0, 1000.0).unwrap()),
                ..Default::default()
            },
            ..Default::default()
        };

        app.world_mut().resource_mut::<Episode>().reset(None);
        app.update();

        let mut values = Vec::new();
        for actuator in app.world_mut().query::<&Actuator>().iter(app.world()) {
            if let clankers_actuator_core::motor::MotorType::Ideal(m) = &actuator.motor {
                values.push(m.max_torque);
            }
        }

        assert_eq!(values.len(), 2);
        // With a range of 1-1000, two samples are extremely unlikely to be equal
        assert!(
            (values[0] - values[1]).abs() > f32::EPSILON,
            "two actuators should get different random values"
        );
    }
}
