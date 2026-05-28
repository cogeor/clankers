//! Built-in sensor implementations for common robotics observations.
//!
//! Sensors implement [`ObservationSensor`] from `clankers-core` and can
//! be registered with an [`ObservationBuffer`](crate::buffer::ObservationBuffer).
//!
//! # Layout-bound constructors (WS2 PR2)
//!
//! Each joint sensor (`JointStateSensor`, `JointCommandSensor`,
//! `JointTorqueSensor`, and their `Robot*` variants) ships a single
//! canonical constructor:
//!
//! - `new(layout: Arc<JointLayout>)` (or `new(RobotId, Arc<JointLayout>)`
//!   for the robot variants). Walks a layout-ordered `Vec<Entity>`
//!   snapshot taken at construction; missing components fill `NaN` so
//!   configuration drift is visible in the observation vector. The
//!   layout MUST be bound (have `entity = Some(_)` slots) BEFORE the
//!   sensor is built; see
//!   [`JointLayout::bind_entities`](clankers_core::layout::JointLayout::bind_entities).
//!
//! See `docs/plans/WS2-plan.md` § 5 PR2-1..PR2-3.

use std::sync::Arc;

use clankers_actuator::components::{JointCommand, JointState, JointTorque};
use clankers_core::{
    layout::JointLayout,
    physics::{ContactData, EndEffectorState, ImuData, LidarConfig, RaycastResult},
    traits::{ObservationSensor, Sensor},
    types::{Observation, RobotId},
};
use clankers_noise::prelude::NoiseModel;
use clankers_physics::rapier::RapierContext;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// SensorBuildError
// ---------------------------------------------------------------------------

// Wrapped in an inline module so `bevy::prelude::*` (imported above and
// shadowing `error` with a logging macro) does not collide with
// thiserror's `#[error(...)]` derive helper attribute.
mod build_error {
    use thiserror::Error;

    /// Failure mode for the fallible `try_new` constructors on
    /// layout-bound sensors (`JointStateSensor::try_new`, etc).
    ///
    /// The panicking `new(...)` constructors snapshot
    /// `layout.bound_entities()` at construction, which silently
    /// shrinks the observation dimension when the layout is unbound or
    /// only partially bound. `try_new` rejects those layouts before
    /// the snapshot is taken so the failure surfaces at sensor build
    /// time, not as a malformed observation downstream.
    #[derive(Debug, Clone, PartialEq, Eq, Error)]
    pub enum SensorBuildError {
        /// The supplied `JointLayout` has at least one slot whose
        /// `entity` is `None`. The sensor refuses to materialise
        /// because its observation dimension would be inconsistent
        /// with the layout's declared slot count.
        #[error(
            "sensor build: layout has {layout_len} slots but only {bound} are \
             bound; sensor requires every slot to be bound before construction"
        )]
        LayoutNotFullyBound {
            /// `layout.len()` — total number of joint slots.
            layout_len: usize,
            /// Count of slots whose `entity` is currently `Some(_)`.
            bound: usize,
        },
    }
}

pub use build_error::SensorBuildError;

fn entities_or_err(layout: &JointLayout) -> Result<Vec<Entity>, SensorBuildError> {
    let entities: Vec<Entity> = layout.bound_entities().collect();
    if entities.len() != layout.len() {
        return Err(SensorBuildError::LayoutNotFullyBound {
            layout_len: layout.len(),
            bound: entities.len(),
        });
    }
    Ok(entities)
}

// ---------------------------------------------------------------------------
// JointStateSensor
// ---------------------------------------------------------------------------

/// Reads `(position, velocity)` for every joint in a [`JointLayout`].
///
/// Produces `2 × layout.len()` values:
/// `[pos_0, vel_0, pos_1, vel_1, ...]`, indexed by layout slot.
///
/// The slot → entity mapping is cached at construction by snapshotting
/// `layout.bound_entities()` into a `Vec<Entity>`. The layout MUST
/// therefore be bound (see
/// [`JointLayout::bind_entities`](clankers_core::layout::JointLayout::bind_entities))
/// BEFORE the sensor is built; an unbound layout produces an empty
/// snapshot and the sensor emits a zero-length observation.
///
/// Entities missing a [`JointState`] component fill `NaN` in their two
/// slots, so a misconfigured layout produces a visible signature in the
/// observation vector rather than a silent skip.
pub struct JointStateSensor {
    /// Shared layout — kept alive so callers can introspect joint
    /// names / kinds via the sensor at runtime.
    layout: Arc<JointLayout>,
    /// Cached entity snapshot in layout order.
    entities: Vec<Entity>,
    /// Cached observation dimension (`2 * entities.len()`).
    dim: usize,
}

impl JointStateSensor {
    /// Build a sensor that walks the supplied layout's bound entities,
    /// in layout order, on every `read()`.
    ///
    /// Prefer [`Self::try_new`] in new code so unbound or partially
    /// bound layouts surface as [`SensorBuildError`] rather than a
    /// silent dimension shrink.
    ///
    /// # Panics
    ///
    /// Panics if the layout is not fully bound; see [`Self::try_new`].
    #[must_use]
    pub fn new(layout: Arc<JointLayout>) -> Self {
        Self::try_new(layout).expect("JointStateSensor::new: layout not fully bound")
    }

    /// Fallible constructor. Returns [`SensorBuildError::LayoutNotFullyBound`]
    /// when `layout.bound_entities().count() != layout.len()` — i.e. any
    /// joint slot still has `entity == None`.
    ///
    /// # Errors
    ///
    /// Returns [`SensorBuildError::LayoutNotFullyBound`] if any joint
    /// slot in the layout is unbound.
    pub fn try_new(layout: Arc<JointLayout>) -> Result<Self, SensorBuildError> {
        let entities = entities_or_err(&layout)?;
        let dim = entities.len() * 2;
        Ok(Self {
            layout,
            entities,
            dim,
        })
    }

    /// Borrow the layout this sensor was built from.
    #[must_use]
    pub fn layout(&self) -> &JointLayout {
        &self.layout
    }
}

impl Sensor for JointStateSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.dim);
        for &entity in &self.entities {
            if let Some(state) = world.get::<JointState>(entity) {
                data.push(state.position);
                data.push(state.velocity);
            } else {
                data.push(f32::NAN);
                data.push(f32::NAN);
            }
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "JointStateSensor"
    }
}

impl ObservationSensor for JointStateSensor {
    fn observation_dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// JointCommandSensor
// ---------------------------------------------------------------------------

/// Reads the current [`JointCommand`] value for every joint in a
/// [`JointLayout`].
///
/// Produces `layout.len()` values, indexed by layout slot. Missing
/// `JointCommand` components fill `NaN`. See [`JointStateSensor`] for
/// the constructor contract; this sensor mirrors it.
pub struct JointCommandSensor {
    layout: Arc<JointLayout>,
    entities: Vec<Entity>,
    dim: usize,
}

impl JointCommandSensor {
    /// Build a sensor that walks the supplied layout's bound entities,
    /// in layout order, on every `read()`.
    ///
    /// # Panics
    ///
    /// Panics if the layout is not fully bound; see [`Self::try_new`].
    #[must_use]
    pub fn new(layout: Arc<JointLayout>) -> Self {
        Self::try_new(layout).expect("JointCommandSensor::new: layout not fully bound")
    }

    /// Fallible constructor. See [`SensorBuildError`].
    ///
    /// # Errors
    ///
    /// Returns [`SensorBuildError::LayoutNotFullyBound`] if any joint
    /// slot is unbound.
    pub fn try_new(layout: Arc<JointLayout>) -> Result<Self, SensorBuildError> {
        let entities = entities_or_err(&layout)?;
        let dim = entities.len();
        Ok(Self {
            layout,
            entities,
            dim,
        })
    }

    /// Borrow the layout this sensor was built from.
    #[must_use]
    pub fn layout(&self) -> &JointLayout {
        &self.layout
    }
}

impl Sensor for JointCommandSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.dim);
        for &entity in &self.entities {
            if let Some(cmd) = world.get::<JointCommand>(entity) {
                data.push(cmd.value);
            } else {
                data.push(f32::NAN);
            }
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "JointCommandSensor"
    }
}

impl ObservationSensor for JointCommandSensor {
    fn observation_dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// JointTorqueSensor
// ---------------------------------------------------------------------------

/// Reads the most recent [`JointTorque`] value for every joint in a
/// [`JointLayout`].
///
/// Produces `layout.len()` values, indexed by layout slot. Missing
/// `JointTorque` components fill `NaN`. Mirrors [`JointStateSensor`]'s
/// constructor contract.
pub struct JointTorqueSensor {
    layout: Arc<JointLayout>,
    entities: Vec<Entity>,
    dim: usize,
}

impl JointTorqueSensor {
    /// Build a sensor that walks the supplied layout's bound entities,
    /// in layout order, on every `read()`.
    ///
    /// # Panics
    ///
    /// Panics if the layout is not fully bound; see [`Self::try_new`].
    #[must_use]
    pub fn new(layout: Arc<JointLayout>) -> Self {
        Self::try_new(layout).expect("JointTorqueSensor::new: layout not fully bound")
    }

    /// Fallible constructor. See [`SensorBuildError`].
    ///
    /// # Errors
    ///
    /// Returns [`SensorBuildError::LayoutNotFullyBound`] if any joint
    /// slot is unbound.
    pub fn try_new(layout: Arc<JointLayout>) -> Result<Self, SensorBuildError> {
        let entities = entities_or_err(&layout)?;
        let dim = entities.len();
        Ok(Self {
            layout,
            entities,
            dim,
        })
    }

    /// Borrow the layout this sensor was built from.
    #[must_use]
    pub fn layout(&self) -> &JointLayout {
        &self.layout
    }
}

impl Sensor for JointTorqueSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.dim);
        for &entity in &self.entities {
            if let Some(torque) = world.get::<JointTorque>(entity) {
                data.push(torque.value);
            } else {
                data.push(f32::NAN);
            }
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "JointTorqueSensor"
    }
}

impl ObservationSensor for JointTorqueSensor {
    fn observation_dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Robot-scoped sensors
// ---------------------------------------------------------------------------

/// Reads `(position, velocity)` for every joint in a [`JointLayout`]
/// that belongs to a specific robot.
///
/// Identical to [`JointStateSensor`] but additionally asserts (in debug
/// builds, via [`Sensor::read`]) that every layout-bound entity carries
/// a [`RobotId`] component matching `robot_id`. Catches the
/// multi-robot pitfall where two layouts share entity ids by accident.
pub struct RobotJointStateSensor {
    robot_id: RobotId,
    layout: Arc<JointLayout>,
    entities: Vec<Entity>,
    dim: usize,
}

impl RobotJointStateSensor {
    /// Build a sensor that walks the supplied layout's bound entities,
    /// in layout order. Pins the `RobotId` so [`Sensor::read`] can
    /// debug-assert every layout entity belongs to the expected robot.
    ///
    /// # Panics
    ///
    /// Panics if the layout is not fully bound; see [`Self::try_new`].
    #[must_use]
    pub fn new(robot_id: RobotId, layout: Arc<JointLayout>) -> Self {
        Self::try_new(robot_id, layout).expect("RobotJointStateSensor::new: layout not fully bound")
    }

    /// Fallible constructor. See [`SensorBuildError`].
    ///
    /// # Errors
    ///
    /// Returns [`SensorBuildError::LayoutNotFullyBound`] if any joint
    /// slot is unbound.
    pub fn try_new(robot_id: RobotId, layout: Arc<JointLayout>) -> Result<Self, SensorBuildError> {
        let entities = entities_or_err(&layout)?;
        let dim = entities.len() * 2;
        Ok(Self {
            robot_id,
            layout,
            entities,
            dim,
        })
    }

    /// Borrow the layout this sensor was built from.
    #[must_use]
    pub fn layout(&self) -> &JointLayout {
        &self.layout
    }

    /// The robot this sensor reads from.
    #[must_use]
    pub const fn robot_id(&self) -> RobotId {
        self.robot_id
    }
}

impl Sensor for RobotJointStateSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.dim);
        for &entity in &self.entities {
            debug_assert!(
                world
                    .get::<RobotId>(entity)
                    .is_none_or(|id| *id == self.robot_id),
                "RobotJointStateSensor: layout entity {entity:?} belongs to a different robot"
            );
            if let Some(state) = world.get::<JointState>(entity) {
                data.push(state.position);
                data.push(state.velocity);
            } else {
                data.push(f32::NAN);
                data.push(f32::NAN);
            }
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "RobotJointStateSensor"
    }
}

impl ObservationSensor for RobotJointStateSensor {
    fn observation_dim(&self) -> usize {
        self.dim
    }
}

/// Reads commands from every joint in a [`JointLayout`] that belongs
/// to a specific robot. Mirrors [`RobotJointStateSensor`].
pub struct RobotJointCommandSensor {
    robot_id: RobotId,
    layout: Arc<JointLayout>,
    entities: Vec<Entity>,
    dim: usize,
}

impl RobotJointCommandSensor {
    /// Build a sensor that walks the supplied layout's bound entities,
    /// in layout order.
    ///
    /// # Panics
    ///
    /// Panics if the layout is not fully bound; see [`Self::try_new`].
    #[must_use]
    pub fn new(robot_id: RobotId, layout: Arc<JointLayout>) -> Self {
        Self::try_new(robot_id, layout)
            .expect("RobotJointCommandSensor::new: layout not fully bound")
    }

    /// Fallible constructor. See [`SensorBuildError`].
    ///
    /// # Errors
    ///
    /// Returns [`SensorBuildError::LayoutNotFullyBound`] if any joint
    /// slot is unbound.
    pub fn try_new(robot_id: RobotId, layout: Arc<JointLayout>) -> Result<Self, SensorBuildError> {
        let entities = entities_or_err(&layout)?;
        let dim = entities.len();
        Ok(Self {
            robot_id,
            layout,
            entities,
            dim,
        })
    }

    /// Borrow the layout this sensor was built from.
    #[must_use]
    pub fn layout(&self) -> &JointLayout {
        &self.layout
    }

    /// The robot this sensor reads from.
    #[must_use]
    pub const fn robot_id(&self) -> RobotId {
        self.robot_id
    }
}

impl Sensor for RobotJointCommandSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.dim);
        for &entity in &self.entities {
            debug_assert!(
                world
                    .get::<RobotId>(entity)
                    .is_none_or(|id| *id == self.robot_id),
                "RobotJointCommandSensor: layout entity {entity:?} belongs to a different robot"
            );
            if let Some(cmd) = world.get::<JointCommand>(entity) {
                data.push(cmd.value);
            } else {
                data.push(f32::NAN);
            }
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "RobotJointCommandSensor"
    }
}

impl ObservationSensor for RobotJointCommandSensor {
    fn observation_dim(&self) -> usize {
        self.dim
    }
}

/// Reads torques from every joint in a [`JointLayout`] that belongs to
/// a specific robot. Mirrors [`RobotJointStateSensor`].
pub struct RobotJointTorqueSensor {
    robot_id: RobotId,
    layout: Arc<JointLayout>,
    entities: Vec<Entity>,
    dim: usize,
}

impl RobotJointTorqueSensor {
    /// Build a sensor that walks the supplied layout's bound entities,
    /// in layout order.
    ///
    /// # Panics
    ///
    /// Panics if the layout is not fully bound; see [`Self::try_new`].
    #[must_use]
    pub fn new(robot_id: RobotId, layout: Arc<JointLayout>) -> Self {
        Self::try_new(robot_id, layout)
            .expect("RobotJointTorqueSensor::new: layout not fully bound")
    }

    /// Fallible constructor. See [`SensorBuildError`].
    ///
    /// # Errors
    ///
    /// Returns [`SensorBuildError::LayoutNotFullyBound`] if any joint
    /// slot is unbound.
    pub fn try_new(robot_id: RobotId, layout: Arc<JointLayout>) -> Result<Self, SensorBuildError> {
        let entities = entities_or_err(&layout)?;
        let dim = entities.len();
        Ok(Self {
            robot_id,
            layout,
            entities,
            dim,
        })
    }

    /// Borrow the layout this sensor was built from.
    #[must_use]
    pub fn layout(&self) -> &JointLayout {
        &self.layout
    }

    /// The robot this sensor reads from.
    #[must_use]
    pub const fn robot_id(&self) -> RobotId {
        self.robot_id
    }
}

impl Sensor for RobotJointTorqueSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.dim);
        for &entity in &self.entities {
            debug_assert!(
                world
                    .get::<RobotId>(entity)
                    .is_none_or(|id| *id == self.robot_id),
                "RobotJointTorqueSensor: layout entity {entity:?} belongs to a different robot"
            );
            if let Some(torque) = world.get::<JointTorque>(entity) {
                data.push(torque.value);
            } else {
                data.push(f32::NAN);
            }
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "RobotJointTorqueSensor"
    }
}

impl ObservationSensor for RobotJointTorqueSensor {
    fn observation_dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// ImuSensor
// ---------------------------------------------------------------------------

/// Reads linear acceleration and angular velocity from all [`ImuData`] entities.
///
/// Produces `6 × n_imus` values:
/// `[accel_x_0, accel_y_0, accel_z_0, gyro_x_0, gyro_y_0, gyro_z_0, ...]`.
pub struct ImuSensor {
    /// Expected number of IMU entities.
    n_imus: usize,
}

impl ImuSensor {
    pub const fn new(n_imus: usize) -> Self {
        Self { n_imus }
    }
}

impl Sensor for ImuSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_imus * 6);
        let mut query = world.query::<&ImuData>();
        for imu in query.iter(world) {
            data.push(imu.linear_acceleration.x);
            data.push(imu.linear_acceleration.y);
            data.push(imu.linear_acceleration.z);
            data.push(imu.angular_velocity.x);
            data.push(imu.angular_velocity.y);
            data.push(imu.angular_velocity.z);
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "ImuSensor"
    }
}

impl ObservationSensor for ImuSensor {
    fn observation_dim(&self) -> usize {
        self.n_imus * 6
    }
}

// ---------------------------------------------------------------------------
// RobotImuSensor
// ---------------------------------------------------------------------------

/// Reads IMU data only from entities belonging to a specific robot.
///
/// Like [`ImuSensor`] but filtered by [`RobotId`].
pub struct RobotImuSensor {
    robot_id: RobotId,
    n_imus: usize,
}

impl RobotImuSensor {
    pub const fn new(robot_id: RobotId, n_imus: usize) -> Self {
        Self { robot_id, n_imus }
    }
}

impl Sensor for RobotImuSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_imus * 6);
        let mut query = world.query::<(&ImuData, &RobotId)>();
        for (imu, &id) in query.iter(world) {
            if id == self.robot_id {
                data.push(imu.linear_acceleration.x);
                data.push(imu.linear_acceleration.y);
                data.push(imu.linear_acceleration.z);
                data.push(imu.angular_velocity.x);
                data.push(imu.angular_velocity.y);
                data.push(imu.angular_velocity.z);
            }
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "RobotImuSensor"
    }
}

impl ObservationSensor for RobotImuSensor {
    fn observation_dim(&self) -> usize {
        self.n_imus * 6
    }
}

// ---------------------------------------------------------------------------
// ContactSensor
// ---------------------------------------------------------------------------

/// Reads contact normal forces from all [`ContactData`] entities.
///
/// Produces `3 × n_contacts` values: `[fx_0, fy_0, fz_0, fx_1, fy_1, fz_1, ...]`.
/// Zero force indicates no active contact.
pub struct ContactSensor {
    /// Expected number of contact-sensing entities.
    n_contacts: usize,
}

impl ContactSensor {
    pub const fn new(n_contacts: usize) -> Self {
        Self { n_contacts }
    }
}

impl Sensor for ContactSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_contacts * 3);
        let mut query = world.query::<&ContactData>();
        for contact in query.iter(world) {
            data.push(contact.normal_force.x);
            data.push(contact.normal_force.y);
            data.push(contact.normal_force.z);
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "ContactSensor"
    }
}

impl ObservationSensor for ContactSensor {
    fn observation_dim(&self) -> usize {
        self.n_contacts * 3
    }
}

// ---------------------------------------------------------------------------
// RobotContactSensor
// ---------------------------------------------------------------------------

/// Reads contact forces only from entities belonging to a specific robot.
///
/// Like [`ContactSensor`] but filtered by [`RobotId`].
pub struct RobotContactSensor {
    robot_id: RobotId,
    n_contacts: usize,
}

impl RobotContactSensor {
    pub const fn new(robot_id: RobotId, n_contacts: usize) -> Self {
        Self {
            robot_id,
            n_contacts,
        }
    }
}

impl Sensor for RobotContactSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_contacts * 3);
        let mut query = world.query::<(&ContactData, &RobotId)>();
        for (contact, &id) in query.iter(world) {
            if id == self.robot_id {
                data.push(contact.normal_force.x);
                data.push(contact.normal_force.y);
                data.push(contact.normal_force.z);
            }
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "RobotContactSensor"
    }
}

impl ObservationSensor for RobotContactSensor {
    fn observation_dim(&self) -> usize {
        self.n_contacts * 3
    }
}

// ---------------------------------------------------------------------------
// RaycastSensor
// ---------------------------------------------------------------------------

/// Reads hit distances from all [`RaycastResult`] entities.
///
/// Produces `n_rays` values (one distance per ray). Entities are read
/// in Bevy query iteration order; spawn consistently for determinism.
pub struct RaycastSensor {
    /// Total expected number of rays across all entities.
    n_rays: usize,
}

impl RaycastSensor {
    pub const fn new(n_rays: usize) -> Self {
        Self { n_rays }
    }
}

impl Sensor for RaycastSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_rays);
        let mut query = world.query::<&RaycastResult>();
        for result in query.iter(world) {
            data.extend_from_slice(&result.distances);
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "RaycastSensor"
    }
}

impl ObservationSensor for RaycastSensor {
    fn observation_dim(&self) -> usize {
        self.n_rays
    }
}

// ---------------------------------------------------------------------------
// RobotRaycastSensor
// ---------------------------------------------------------------------------

/// Reads raycast hit distances only from entities belonging to a specific robot.
///
/// Like [`RaycastSensor`] but filtered by [`RobotId`].
pub struct RobotRaycastSensor {
    robot_id: RobotId,
    n_rays: usize,
}

impl RobotRaycastSensor {
    pub const fn new(robot_id: RobotId, n_rays: usize) -> Self {
        Self { robot_id, n_rays }
    }
}

impl Sensor for RobotRaycastSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_rays);
        let mut query = world.query::<(&RaycastResult, &RobotId)>();
        for (result, &id) in query.iter(world) {
            if id == self.robot_id {
                data.extend_from_slice(&result.distances);
            }
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "RobotRaycastSensor"
    }
}

impl ObservationSensor for RobotRaycastSensor {
    fn observation_dim(&self) -> usize {
        self.n_rays
    }
}

// ---------------------------------------------------------------------------
// EndEffectorPoseSensor
// ---------------------------------------------------------------------------

/// Reads world-space position and orientation from all [`EndEffectorState`] entities.
///
/// Produces `7 × n_effectors` values:
/// `[x_0, y_0, z_0, qx_0, qy_0, qz_0, qw_0, ...]`.
pub struct EndEffectorPoseSensor {
    /// Expected number of end-effector entities.
    n_effectors: usize,
}

impl EndEffectorPoseSensor {
    pub const fn new(n_effectors: usize) -> Self {
        Self { n_effectors }
    }
}

impl Sensor for EndEffectorPoseSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_effectors * 7);
        let mut query = world.query::<&EndEffectorState>();
        for ee in query.iter(world) {
            data.push(ee.position.x);
            data.push(ee.position.y);
            data.push(ee.position.z);
            data.push(ee.orientation.x);
            data.push(ee.orientation.y);
            data.push(ee.orientation.z);
            data.push(ee.orientation.w);
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "EndEffectorPoseSensor"
    }
}

impl ObservationSensor for EndEffectorPoseSensor {
    fn observation_dim(&self) -> usize {
        self.n_effectors * 7
    }
}

// ---------------------------------------------------------------------------
// RobotEndEffectorPoseSensor
// ---------------------------------------------------------------------------

/// Reads end-effector pose only from entities belonging to a specific robot.
///
/// Like [`EndEffectorPoseSensor`] but filtered by [`RobotId`].
pub struct RobotEndEffectorPoseSensor {
    robot_id: RobotId,
    n_effectors: usize,
}

impl RobotEndEffectorPoseSensor {
    pub const fn new(robot_id: RobotId, n_effectors: usize) -> Self {
        Self {
            robot_id,
            n_effectors,
        }
    }
}

impl Sensor for RobotEndEffectorPoseSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_effectors * 7);
        let mut query = world.query::<(&EndEffectorState, &RobotId)>();
        for (ee, &id) in query.iter(world) {
            if id == self.robot_id {
                data.push(ee.position.x);
                data.push(ee.position.y);
                data.push(ee.position.z);
                data.push(ee.orientation.x);
                data.push(ee.orientation.y);
                data.push(ee.orientation.z);
                data.push(ee.orientation.w);
            }
        }
        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "RobotEndEffectorPoseSensor"
    }
}

impl ObservationSensor for RobotEndEffectorPoseSensor {
    fn observation_dim(&self) -> usize {
        self.n_effectors * 7
    }
}

// ---------------------------------------------------------------------------
// LidarSensor
// ---------------------------------------------------------------------------

/// CPU raycasting lidar sensor using the Rapier physics world.
///
/// Fires `num_channels × num_rays` rays from the given world-space origin
/// (defined by `config.origin_offset` added to `sensor_origin`) and returns
/// the hit distances as a flat `f32` array.  Rays that miss all colliders
/// within `config.max_range` produce `f32::NAN`.
///
/// Ray directions are computed as:
///
/// - azimuth  = −`half_fov` + `ray_idx`   × (2 × `half_fov` / (`num_rays` − 1))
/// - elevation = −`vertical_half_fov` + `ch_idx` × (2 × `vertical_half_fov` / (`num_channels` − 1))
///
/// and rotated into world space by `sensor_rotation`.  When `num_rays == 1`
/// the single ray points straight ahead (azimuth = 0).  Likewise for channels.
///
/// # Access
///
/// `read()` fetches the [`RapierContext`] resource from the world to perform
/// raycasts.  If the resource is absent the sensor returns an all-NaN vector.
pub struct LidarSensor {
    /// Lidar configuration (layout and range).
    pub config: LidarConfig,
    /// World-space origin of the sensor (body position + `origin_offset`).
    pub sensor_origin: Vec3,
    /// World-space rotation of the sensor frame.
    pub sensor_rotation: Quat,
}

impl LidarSensor {
    /// Create a new `LidarSensor` at the given world-space pose.
    pub const fn new(config: LidarConfig, sensor_origin: Vec3, sensor_rotation: Quat) -> Self {
        Self {
            config,
            sensor_origin,
            sensor_rotation,
        }
    }

    /// Compute the world-space ray direction for a given azimuth and elevation.
    fn ray_direction(rotation: Quat, azimuth: f32, elevation: f32) -> Vec3 {
        // Start with forward (+Z) and rotate by elevation then azimuth.
        let cos_el = elevation.cos();
        let sin_el = elevation.sin();
        let cos_az = azimuth.cos();
        let sin_az = azimuth.sin();

        // Local-frame direction (right-handed, +Z forward, +Y up).
        let local_dir = Vec3::new(sin_az * cos_el, sin_el, cos_el * cos_az);
        rotation * local_dir
    }
}

impl Sensor for LidarSensor {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let total = self.config.num_rays * self.config.num_channels;
        let mut data = vec![f32::NAN; total];

        let Some(ctx) = world.get_resource::<RapierContext>() else {
            return Observation::new(data);
        };

        let origin = self.config.origin_offset + self.sensor_origin;
        let nr = self.config.num_rays;
        let nc = self.config.num_channels;
        let hfov = self.config.half_fov;
        let vhfov = self.config.vertical_half_fov;
        let max_range = self.config.max_range;

        for ch in 0..nc {
            let elevation = if nc <= 1 {
                0.0_f32
            } else {
                #[allow(clippy::cast_precision_loss)]
                (ch as f32).mul_add(2.0 * vhfov / (nc - 1) as f32, -vhfov)
            };

            for ray_idx in 0..nr {
                let azimuth = if nr <= 1 {
                    0.0_f32
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    (ray_idx as f32).mul_add(2.0 * hfov / (nr - 1) as f32, -hfov)
                };

                let dir = Self::ray_direction(self.sensor_rotation, azimuth, elevation);

                // Convert bevy Vec3 → rapier Vector (nalgebra)
                let dist = ctx
                    .cast_ray_bevy(origin, dir, max_range, true)
                    .unwrap_or(f32::NAN);

                data[ch * nr + ray_idx] = dist;
            }
        }

        Observation::new(data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "LidarSensor"
    }
}

impl ObservationSensor for LidarSensor {
    fn observation_dim(&self) -> usize {
        self.config.num_rays * self.config.num_channels
    }
}

// ---------------------------------------------------------------------------
// NoisySensor
// ---------------------------------------------------------------------------

/// Wraps any [`ObservationSensor`] and adds noise to its output.
///
/// Uses a [`NoiseModel`] applied element-wise. An internal RNG ensures
/// noise is applied automatically through the standard `read()` path.
pub struct NoisySensor<S: ObservationSensor> {
    inner: S,
    noise: NoiseModel,
    rng: ChaCha8Rng,
}

impl<S: ObservationSensor> NoisySensor<S> {
    /// Create a noisy sensor with a random seed.
    pub fn new(inner: S, noise: NoiseModel) -> Self {
        Self {
            inner,
            noise,
            rng: ChaCha8Rng::from_entropy(),
        }
    }

    /// Create a noisy sensor with a deterministic seed.
    pub fn with_seed(inner: S, noise: NoiseModel, seed: u64) -> Self {
        Self {
            inner,
            noise,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl<S: ObservationSensor> Sensor for NoisySensor<S> {
    type Output = Observation;

    fn read(&mut self, world: &mut World) -> Observation {
        let mut obs = self.inner.read(world);
        // Apply noise through the standard read path.
        for val in obs.as_mut_slice() {
            *val = self.noise.apply(*val, &mut self.rng);
        }
        obs
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn episode_reset(&mut self) {
        self.noise.reset(&mut self.rng);
    }
}

impl<S: ObservationSensor> ObservationSensor for NoisySensor<S> {
    fn observation_dim(&self) -> usize {
        self.inner.observation_dim()
    }
}

impl<S: ObservationSensor> NoisySensor<S> {
    /// Apply noise to an observation in-place using the given RNG.
    pub fn apply_noise<R: rand::Rng + ?Sized>(&mut self, obs: &mut Observation, rng: &mut R) {
        for val in obs.as_mut_slice() {
            *val = self.noise.apply(*val, rng);
        }
    }

    /// Read from the world and apply noise.
    pub fn read_noisy<R: rand::Rng + ?Sized>(
        &mut self,
        world: &mut World,
        rng: &mut R,
    ) -> Observation {
        let mut obs = self.inner.read(world);
        self.apply_noise(&mut obs, rng);
        obs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clankers_actuator::components::Actuator;
    use clankers_core::layout::{JointKind, JointLayoutBuilder, JointSpec, JointSpecLimits};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Build a synthetic layout of `n` revolute joints bound to the
    /// supplied entity list. Used by tests that don't parse URDF.
    fn synthetic_layout(entities: &[Entity]) -> Arc<JointLayout> {
        let mut builder = JointLayoutBuilder::default();
        for (i, _) in entities.iter().enumerate() {
            builder = builder.push(JointSpec {
                name: format!("j{i}"),
                entity: None,
                joint_type: JointKind::Revolute,
                limits: JointSpecLimits::default(),
                axis: [0.0, 0.0, 1.0],
            });
        }
        let mut layout = builder.build();
        layout.bind_entities(entities);
        Arc::new(layout)
    }

    fn spawn_joint(world: &mut World, pos: f32, vel: f32, cmd: f32) -> Entity {
        world
            .spawn((
                Actuator::default(),
                JointCommand { value: cmd },
                JointState {
                    position: pos,
                    velocity: vel,
                },
                JointTorque::default(),
            ))
            .id()
    }

    // -- JointStateSensor --

    #[test]
    fn joint_state_sensor_reads_correctly() {
        let mut world = World::new();
        let e0 = spawn_joint(&mut world, 1.0, 2.0, 0.0);
        let e1 = spawn_joint(&mut world, 3.0, 4.0, 0.0);

        let layout = synthetic_layout(&[e0, e1]);
        let mut sensor = JointStateSensor::new(layout);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 4);
        let vals: Vec<f32> = obs.as_slice().to_vec();
        // Layout-ordered: [pos_0, vel_0, pos_1, vel_1]
        assert!((vals[0] - 1.0).abs() < f32::EPSILON);
        assert!((vals[1] - 2.0).abs() < f32::EPSILON);
        assert!((vals[2] - 3.0).abs() < f32::EPSILON);
        assert!((vals[3] - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn joint_state_sensor_empty_layout() {
        let mut world = World::new();
        let layout = synthetic_layout(&[]);
        let mut sensor = JointStateSensor::new(layout);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn joint_state_sensor_dim() {
        let mut world = World::new();
        let entities: Vec<Entity> = (0..5)
            .map(|_| spawn_joint(&mut world, 0.0, 0.0, 0.0))
            .collect();
        let layout = synthetic_layout(&entities);
        let sensor = JointStateSensor::new(layout);
        assert_eq!(sensor.observation_dim(), 10);
    }

    #[test]
    fn joint_state_sensor_name() {
        let mut world = World::new();
        let e = spawn_joint(&mut world, 0.0, 0.0, 0.0);
        let layout = synthetic_layout(&[e]);
        let sensor = JointStateSensor::new(layout);
        assert_eq!(sensor.name(), "JointStateSensor");
    }

    #[test]
    fn joint_state_sensor_try_new_rejects_unbound_layout() {
        // P0.4: layout with unbound slots must surface as
        // SensorBuildError::LayoutNotFullyBound, not silently shrink dim.
        let mut builder = JointLayoutBuilder::default();
        for i in 0..3 {
            builder = builder.push(JointSpec {
                name: format!("j{i}"),
                entity: None, // <- unbound
                joint_type: JointKind::Revolute,
                limits: JointSpecLimits::default(),
                axis: [0.0, 0.0, 1.0],
            });
        }
        let layout = Arc::new(builder.build());
        match JointStateSensor::try_new(layout) {
            Ok(_) => panic!("expected SensorBuildError::LayoutNotFullyBound"),
            Err(e) => assert_eq!(
                e,
                SensorBuildError::LayoutNotFullyBound {
                    layout_len: 3,
                    bound: 0,
                }
            ),
        }
    }

    #[test]
    fn joint_state_sensor_missing_component_fills_nan() {
        let mut world = World::new();
        // Spawn an entity without JointState — sensor must fill NaN.
        let bare = world.spawn(()).id();
        let layout = synthetic_layout(&[bare]);
        let mut sensor = JointStateSensor::new(layout);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 2);
        assert!(obs[0].is_nan());
        assert!(obs[1].is_nan());
    }

    // -- JointCommandSensor --

    #[test]
    fn joint_command_sensor_reads() {
        let mut world = World::new();
        let e0 = spawn_joint(&mut world, 0.0, 0.0, 5.0);
        let e1 = spawn_joint(&mut world, 0.0, 0.0, 10.0);

        let layout = synthetic_layout(&[e0, e1]);
        let mut sensor = JointCommandSensor::new(layout);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 2);
        let vals: Vec<f32> = obs.as_slice().to_vec();
        assert!((vals[0] - 5.0).abs() < f32::EPSILON);
        assert!((vals[1] - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn joint_command_sensor_dim() {
        let mut world = World::new();
        let entities: Vec<Entity> = (0..3)
            .map(|_| spawn_joint(&mut world, 0.0, 0.0, 0.0))
            .collect();
        let layout = synthetic_layout(&entities);
        let sensor = JointCommandSensor::new(layout);
        assert_eq!(sensor.observation_dim(), 3);
    }

    // -- JointTorqueSensor --

    #[test]
    fn joint_torque_sensor_reads() {
        let mut world = World::new();
        let e0 = world.spawn(JointTorque { value: 7.5 }).id();
        let e1 = world.spawn(JointTorque { value: -3.0 }).id();

        let layout = synthetic_layout(&[e0, e1]);
        let mut sensor = JointTorqueSensor::new(layout);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 2);
        let vals: Vec<f32> = obs.as_slice().to_vec();
        assert!((vals[0] - 7.5).abs() < f32::EPSILON);
        assert!((vals[1] + 3.0).abs() < f32::EPSILON);
    }

    // -- ImuSensor --

    fn spawn_imu(world: &mut World, accel: Vec3, gyro: Vec3) -> Entity {
        world.spawn(ImuData::new(accel, gyro)).id()
    }

    #[test]
    fn imu_sensor_reads_correctly() {
        let mut world = World::new();
        spawn_imu(
            &mut world,
            Vec3::new(0.0, -9.81, 0.0),
            Vec3::new(0.1, 0.2, 0.3),
        );

        let mut sensor = ImuSensor::new(1);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 6);
        let vals = obs.as_slice();
        assert!((vals[0] - 0.0).abs() < f32::EPSILON);
        assert!((vals[1] - (-9.81)).abs() < f32::EPSILON);
        assert!((vals[2] - 0.0).abs() < f32::EPSILON);
        assert!((vals[3] - 0.1).abs() < f32::EPSILON);
        assert!((vals[4] - 0.2).abs() < f32::EPSILON);
        assert!((vals[5] - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn imu_sensor_multiple_imus() {
        let mut world = World::new();
        spawn_imu(&mut world, Vec3::X, Vec3::Y);
        spawn_imu(&mut world, Vec3::Z, Vec3::NEG_X);

        let mut sensor = ImuSensor::new(2);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 12);
    }

    #[test]
    fn imu_sensor_empty_world() {
        let mut world = World::new();
        let mut sensor = ImuSensor::new(0);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn imu_sensor_dim() {
        let sensor = ImuSensor::new(2);
        assert_eq!(sensor.observation_dim(), 12);
    }

    #[test]
    fn imu_sensor_name() {
        let sensor = ImuSensor::new(1);
        assert_eq!(sensor.name(), "ImuSensor");
    }

    // -- RobotImuSensor --

    #[test]
    fn robot_imu_sensor_filters_by_id() {
        let mut world = World::new();
        world.spawn((ImuData::new(Vec3::X, Vec3::Y), RobotId(0)));
        world.spawn((ImuData::new(Vec3::Z, Vec3::NEG_X), RobotId(1)));
        world.spawn((ImuData::new(Vec3::NEG_Y, Vec3::NEG_Z), RobotId(0)));

        let mut sensor = RobotImuSensor::new(RobotId(0), 2);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 12);
        // Robot 1's data (Z, NEG_X) should not appear
        let vals = obs.as_slice();
        // Should not contain z=1.0 from accel of robot 1
        // Robot 0 entities: (X, Y) and (NEG_Y, NEG_Z)
        assert!(vals.contains(&1.0)); // X.x from first entity
        assert!(vals.contains(&-1.0)); // NEG_Y.y from third entity
    }

    #[test]
    fn robot_imu_sensor_empty_when_no_match() {
        let mut world = World::new();
        world.spawn((ImuData::default(), RobotId(0)));

        let mut sensor = RobotImuSensor::new(RobotId(99), 0);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn robot_imu_sensor_name_and_dim() {
        let sensor = RobotImuSensor::new(RobotId(0), 3);
        assert_eq!(sensor.name(), "RobotImuSensor");
        assert_eq!(sensor.observation_dim(), 18);
    }

    // -- ContactSensor --

    #[test]
    fn contact_sensor_reads_correctly() {
        let mut world = World::new();
        world.spawn(ContactData::new(Vec3::new(0.0, 50.0, 0.0)));

        let mut sensor = ContactSensor::new(1);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 3);
        assert!((obs[0] - 0.0).abs() < f32::EPSILON);
        assert!((obs[1] - 50.0).abs() < f32::EPSILON);
        assert!((obs[2] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn contact_sensor_multiple() {
        let mut world = World::new();
        world.spawn(ContactData::new(Vec3::X));
        world.spawn(ContactData::new(Vec3::Y));

        let mut sensor = ContactSensor::new(2);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 6);
    }

    #[test]
    fn contact_sensor_empty_world() {
        let mut world = World::new();
        let mut sensor = ContactSensor::new(0);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn contact_sensor_dim_and_name() {
        let sensor = ContactSensor::new(4);
        assert_eq!(sensor.observation_dim(), 12);
        assert_eq!(sensor.name(), "ContactSensor");
    }

    // -- RobotContactSensor --

    #[test]
    fn robot_contact_sensor_filters_by_id() {
        let mut world = World::new();
        world.spawn((ContactData::new(Vec3::new(0.0, 10.0, 0.0)), RobotId(0)));
        world.spawn((ContactData::new(Vec3::new(0.0, 20.0, 0.0)), RobotId(1)));

        let mut sensor = RobotContactSensor::new(RobotId(0), 1);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 3);
        assert!((obs[1] - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn robot_contact_sensor_empty_when_no_match() {
        let mut world = World::new();
        world.spawn((ContactData::default(), RobotId(0)));

        let mut sensor = RobotContactSensor::new(RobotId(99), 0);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn robot_contact_sensor_name_and_dim() {
        let sensor = RobotContactSensor::new(RobotId(0), 4);
        assert_eq!(sensor.name(), "RobotContactSensor");
        assert_eq!(sensor.observation_dim(), 12);
    }

    // -- RaycastSensor --

    #[test]
    fn raycast_sensor_reads_correctly() {
        let mut world = World::new();
        world.spawn(RaycastResult::new(vec![1.5, 3.0, 10.0], 10.0));

        let mut sensor = RaycastSensor::new(3);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 3);
        assert!((obs[0] - 1.5).abs() < f32::EPSILON);
        assert!((obs[1] - 3.0).abs() < f32::EPSILON);
        assert!((obs[2] - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn raycast_sensor_multiple_entities() {
        let mut world = World::new();
        world.spawn(RaycastResult::new(vec![1.0, 2.0], 5.0));
        world.spawn(RaycastResult::new(vec![3.0], 5.0));

        let mut sensor = RaycastSensor::new(3);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 3);
    }

    #[test]
    fn raycast_sensor_no_hits() {
        let mut world = World::new();
        world.spawn(RaycastResult::no_hits(4, 10.0));

        let mut sensor = RaycastSensor::new(4);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 4);
        for i in 0..4 {
            assert!((obs[i] - 10.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn raycast_sensor_empty_world() {
        let mut world = World::new();
        let mut sensor = RaycastSensor::new(0);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn raycast_sensor_dim_and_name() {
        let sensor = RaycastSensor::new(16);
        assert_eq!(sensor.observation_dim(), 16);
        assert_eq!(sensor.name(), "RaycastSensor");
    }

    // -- RobotRaycastSensor --

    #[test]
    fn robot_raycast_sensor_filters_by_id() {
        let mut world = World::new();
        world.spawn((RaycastResult::new(vec![1.0, 2.0], 5.0), RobotId(0)));
        world.spawn((RaycastResult::new(vec![3.0, 4.0], 5.0), RobotId(1)));

        let mut sensor = RobotRaycastSensor::new(RobotId(0), 2);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 2);
        assert!((obs[0] - 1.0).abs() < f32::EPSILON);
        assert!((obs[1] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn robot_raycast_sensor_empty_when_no_match() {
        let mut world = World::new();
        world.spawn((RaycastResult::no_hits(3, 10.0), RobotId(0)));

        let mut sensor = RobotRaycastSensor::new(RobotId(99), 0);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn robot_raycast_sensor_name_and_dim() {
        let sensor = RobotRaycastSensor::new(RobotId(0), 8);
        assert_eq!(sensor.name(), "RobotRaycastSensor");
        assert_eq!(sensor.observation_dim(), 8);
    }

    // -- EndEffectorPoseSensor --

    #[test]
    fn end_effector_pose_sensor_reads_correctly() {
        let mut world = World::new();
        let ee = EndEffectorState::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        world.spawn(ee);

        let mut sensor = EndEffectorPoseSensor::new(1);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 7);
        assert!((obs[0] - 1.0).abs() < f32::EPSILON);
        assert!((obs[1] - 2.0).abs() < f32::EPSILON);
        assert!((obs[2] - 3.0).abs() < f32::EPSILON);
        // Quat::IDENTITY = (0, 0, 0, 1)
        assert!((obs[3] - 0.0).abs() < f32::EPSILON);
        assert!((obs[6] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn end_effector_pose_sensor_multiple() {
        let mut world = World::new();
        world.spawn(EndEffectorState::default());
        world.spawn(EndEffectorState::default());

        let mut sensor = EndEffectorPoseSensor::new(2);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 14);
    }

    #[test]
    fn end_effector_pose_sensor_empty_world() {
        let mut world = World::new();
        let mut sensor = EndEffectorPoseSensor::new(0);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn end_effector_pose_sensor_dim_and_name() {
        let sensor = EndEffectorPoseSensor::new(2);
        assert_eq!(sensor.observation_dim(), 14);
        assert_eq!(sensor.name(), "EndEffectorPoseSensor");
    }

    // -- RobotEndEffectorPoseSensor --

    #[test]
    fn robot_end_effector_pose_sensor_filters_by_id() {
        let mut world = World::new();
        world.spawn((
            EndEffectorState::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY),
            RobotId(0),
        ));
        world.spawn((
            EndEffectorState::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY),
            RobotId(1),
        ));

        let mut sensor = RobotEndEffectorPoseSensor::new(RobotId(0), 1);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 7);
        assert!((obs[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn robot_end_effector_pose_sensor_empty_when_no_match() {
        let mut world = World::new();
        world.spawn((EndEffectorState::default(), RobotId(0)));

        let mut sensor = RobotEndEffectorPoseSensor::new(RobotId(99), 0);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn robot_end_effector_pose_sensor_name_and_dim() {
        let sensor = RobotEndEffectorPoseSensor::new(RobotId(0), 2);
        assert_eq!(sensor.name(), "RobotEndEffectorPoseSensor");
        assert_eq!(sensor.observation_dim(), 14);
    }

    // -- NoisySensor --

    #[test]
    fn noisy_sensor_applies_noise() {
        let mut world = World::new();
        let e = spawn_joint(&mut world, 1.0, 2.0, 0.0);

        let layout = synthetic_layout(&[e]);
        let noise = NoiseModel::gaussian(0.0, 0.1).unwrap();
        let mut sensor = NoisySensor::new(JointStateSensor::new(layout), noise);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let obs = sensor.read_noisy(&mut world, &mut rng);
        assert_eq!(obs.len(), 2);
        // Values should be near 1.0 and 2.0 but not exact (noise added)
        assert!((obs[0] - 1.0).abs() < 1.0);
        assert!((obs[1] - 2.0).abs() < 1.0);
    }

    #[test]
    fn noisy_sensor_deterministic_with_same_seed() {
        let mut world = World::new();
        let e = spawn_joint(&mut world, 5.0, 10.0, 0.0);

        let layout = synthetic_layout(&[e]);
        let noise = NoiseModel::gaussian(0.0, 1.0).unwrap();
        let mut sensor = NoisySensor::new(JointStateSensor::new(layout.clone()), noise);
        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let obs1 = sensor.read_noisy(&mut world, &mut rng1);

        let noise2 = NoiseModel::gaussian(0.0, 1.0).unwrap();
        let mut sensor2 = NoisySensor::new(JointStateSensor::new(layout), noise2);
        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let obs2 = sensor2.read_noisy(&mut world, &mut rng2);

        assert_eq!(obs1.as_slice(), obs2.as_slice());
    }

    #[test]
    fn noisy_sensor_delegates_name_and_dim() {
        let mut world = World::new();
        let entities: Vec<Entity> = (0..3)
            .map(|_| spawn_joint(&mut world, 0.0, 0.0, 0.0))
            .collect();
        let layout = synthetic_layout(&entities);
        let noise = NoiseModel::gaussian(0.0, 0.1).unwrap();
        let sensor = NoisySensor::new(JointStateSensor::new(layout), noise);
        assert_eq!(sensor.name(), "JointStateSensor");
        assert_eq!(sensor.observation_dim(), 6);
    }

    // -- Robot-scoped sensors --

    fn spawn_robot_joint(
        world: &mut World,
        robot_id: RobotId,
        pos: f32,
        vel: f32,
        cmd: f32,
    ) -> Entity {
        world
            .spawn((
                robot_id,
                Actuator::default(),
                JointCommand { value: cmd },
                JointState {
                    position: pos,
                    velocity: vel,
                },
                JointTorque { value: pos * 2.0 },
            ))
            .id()
    }

    #[test]
    fn robot_joint_state_sensor_filters_by_id() {
        let mut world = World::new();
        let r0_a = spawn_robot_joint(&mut world, RobotId(0), 1.0, 2.0, 0.0);
        spawn_robot_joint(&mut world, RobotId(1), 3.0, 4.0, 0.0);
        let r0_b = spawn_robot_joint(&mut world, RobotId(0), 5.0, 6.0, 0.0);

        let layout = synthetic_layout(&[r0_a, r0_b]);
        let mut sensor = RobotJointStateSensor::new(RobotId(0), layout);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 4);
        let vals: Vec<f32> = obs.as_slice().to_vec();
        assert!((vals[0] - 1.0).abs() < f32::EPSILON);
        assert!((vals[1] - 2.0).abs() < f32::EPSILON);
        assert!((vals[2] - 5.0).abs() < f32::EPSILON);
        assert!((vals[3] - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn robot_joint_command_sensor_filters_by_id() {
        let mut world = World::new();
        let r0 = spawn_robot_joint(&mut world, RobotId(0), 0.0, 0.0, 10.0);
        spawn_robot_joint(&mut world, RobotId(1), 0.0, 0.0, 20.0);

        let layout = synthetic_layout(&[r0]);
        let mut sensor = RobotJointCommandSensor::new(RobotId(0), layout);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 1);
        assert!((obs[0] - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn robot_joint_torque_sensor_filters_by_id() {
        let mut world = World::new();
        spawn_robot_joint(&mut world, RobotId(0), 5.0, 0.0, 0.0); // torque = 10.0
        let r1 = spawn_robot_joint(&mut world, RobotId(1), 3.0, 0.0, 0.0); // torque = 6.0

        let layout = synthetic_layout(&[r1]);
        let mut sensor = RobotJointTorqueSensor::new(RobotId(1), layout);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 1);
        assert!((obs[0] - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn robot_sensor_empty_layout() {
        let mut world = World::new();
        spawn_robot_joint(&mut world, RobotId(0), 1.0, 2.0, 3.0);

        let layout = synthetic_layout(&[]);
        let mut sensor = RobotJointStateSensor::new(RobotId(99), layout);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn robot_sensor_name_and_dim() {
        let mut world = World::new();
        let entities: Vec<Entity> = (0..4)
            .map(|_| spawn_robot_joint(&mut world, RobotId(0), 0.0, 0.0, 0.0))
            .collect();
        let layout3 = synthetic_layout(&entities[..3]);
        let s1 = RobotJointStateSensor::new(RobotId(0), layout3);
        assert_eq!(s1.name(), "RobotJointStateSensor");
        assert_eq!(s1.observation_dim(), 6);

        let layout2 = synthetic_layout(&entities[..2]);
        let s2 = RobotJointCommandSensor::new(RobotId(0), layout2);
        assert_eq!(s2.name(), "RobotJointCommandSensor");
        assert_eq!(s2.observation_dim(), 2);

        let layout4 = synthetic_layout(&entities);
        let s3 = RobotJointTorqueSensor::new(RobotId(0), layout4);
        assert_eq!(s3.name(), "RobotJointTorqueSensor");
        assert_eq!(s3.observation_dim(), 4);
    }

    // -- LidarSensor --

    #[test]
    fn lidar_config_default() {
        let cfg = LidarConfig::default();
        assert_eq!(cfg.num_rays, 64);
        assert_eq!(cfg.num_channels, 1);
        assert!((cfg.max_range - 10.0).abs() < f32::EPSILON);
        assert!((cfg.half_fov - std::f32::consts::PI).abs() < f32::EPSILON);
        assert!((cfg.vertical_half_fov - 0.0).abs() < f32::EPSILON);
        assert_eq!(cfg.origin_offset, Vec3::ZERO);
    }

    #[test]
    fn lidar_observation_dim() {
        let cfg = LidarConfig {
            num_rays: 16,
            num_channels: 4,
            ..LidarConfig::default()
        };
        let sensor = LidarSensor::new(cfg, Vec3::ZERO, Quat::IDENTITY);
        assert_eq!(sensor.observation_dim(), 64); // 16 × 4
    }

    #[test]
    fn lidar_sensor_name() {
        let sensor = LidarSensor::new(LidarConfig::default(), Vec3::ZERO, Quat::IDENTITY);
        assert_eq!(sensor.name(), "LidarSensor");
    }

    #[test]
    fn lidar_sensor_no_rapier_context_returns_nan() {
        // Without a RapierContext resource every ray should be NaN.
        let mut world = World::new();
        let cfg = LidarConfig {
            num_rays: 4,
            num_channels: 1,
            ..LidarConfig::default()
        };
        let mut sensor = LidarSensor::new(cfg, Vec3::ZERO, Quat::IDENTITY);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 4);
        for v in obs.as_slice() {
            assert!(v.is_nan(), "expected NaN but got {v}");
        }
    }

    #[test]
    fn lidar_sensor_single_ray_dim() {
        let cfg = LidarConfig {
            num_rays: 1,
            num_channels: 1,
            ..LidarConfig::default()
        };
        let sensor = LidarSensor::new(cfg, Vec3::ZERO, Quat::IDENTITY);
        assert_eq!(sensor.observation_dim(), 1);
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn sensor_types_are_send_sync() {
        assert_send_sync::<JointStateSensor>();
        assert_send_sync::<JointCommandSensor>();
        assert_send_sync::<JointTorqueSensor>();
        assert_send_sync::<ImuSensor>();
        assert_send_sync::<ContactSensor>();
        assert_send_sync::<RaycastSensor>();
        assert_send_sync::<EndEffectorPoseSensor>();
        assert_send_sync::<RobotJointStateSensor>();
        assert_send_sync::<RobotJointCommandSensor>();
        assert_send_sync::<RobotJointTorqueSensor>();
        assert_send_sync::<RobotImuSensor>();
        assert_send_sync::<RobotContactSensor>();
        assert_send_sync::<RobotRaycastSensor>();
        assert_send_sync::<RobotEndEffectorPoseSensor>();
        assert_send_sync::<LidarSensor>();
    }
}
