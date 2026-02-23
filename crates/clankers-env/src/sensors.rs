//! Built-in sensor implementations for common robotics observations.
//!
//! Sensors implement [`ObservationSensor`] from `clankers-core` and can
//! be registered with an [`ObservationBuffer`](crate::buffer::ObservationBuffer).

use clankers_actuator::components::{JointCommand, JointState, JointTorque};
use clankers_core::{
    traits::{ObservationSensor, Sensor},
    types::{Observation, RobotId},
};
use clankers_noise::prelude::NoiseModel;

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// JointStateSensor
// ---------------------------------------------------------------------------

/// Reads position and velocity from all entities with [`JointState`].
///
/// Produces `2 × n_joints` values: `[pos_0, vel_0, pos_1, vel_1, ...]`.
///
/// Entity order is determined by Bevy's query iteration order.  For determinism,
/// ensure entities are spawned in a consistent order.
pub struct JointStateSensor {
    /// Expected number of joints (for observation dimension).
    n_joints: usize,
}

impl JointStateSensor {
    pub const fn new(n_joints: usize) -> Self {
        Self { n_joints }
    }
}

impl Sensor for JointStateSensor {
    type Output = Observation;

    fn read(&self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_joints * 2);
        let mut query = world.query::<&JointState>();
        for state in query.iter(world) {
            data.push(state.position);
            data.push(state.velocity);
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
        self.n_joints * 2
    }
}

// ---------------------------------------------------------------------------
// JointCommandSensor
// ---------------------------------------------------------------------------

/// Reads the current command for all entities with [`JointCommand`].
///
/// Produces `n_joints` values: `[cmd_0, cmd_1, ...]`.
pub struct JointCommandSensor {
    n_joints: usize,
}

impl JointCommandSensor {
    pub const fn new(n_joints: usize) -> Self {
        Self { n_joints }
    }
}

impl Sensor for JointCommandSensor {
    type Output = Observation;

    fn read(&self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_joints);
        let mut query = world.query::<&JointCommand>();
        for cmd in query.iter(world) {
            data.push(cmd.value);
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
        self.n_joints
    }
}

// ---------------------------------------------------------------------------
// JointTorqueSensor
// ---------------------------------------------------------------------------

/// Reads computed torque from all entities with [`JointTorque`].
///
/// Produces `n_joints` values: `[torque_0, torque_1, ...]`.
pub struct JointTorqueSensor {
    n_joints: usize,
}

impl JointTorqueSensor {
    pub const fn new(n_joints: usize) -> Self {
        Self { n_joints }
    }
}

impl Sensor for JointTorqueSensor {
    type Output = Observation;

    fn read(&self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_joints);
        let mut query = world.query::<&JointTorque>();
        for torque in query.iter(world) {
            data.push(torque.value);
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
        self.n_joints
    }
}

// ---------------------------------------------------------------------------
// Robot-scoped sensors
// ---------------------------------------------------------------------------

/// Reads position and velocity only from joints belonging to a specific robot.
///
/// Like [`JointStateSensor`] but filtered by [`RobotId`], producing data for
/// a single robot in a multi-robot scene.
pub struct RobotJointStateSensor {
    robot_id: RobotId,
    n_joints: usize,
}

impl RobotJointStateSensor {
    pub const fn new(robot_id: RobotId, n_joints: usize) -> Self {
        Self { robot_id, n_joints }
    }
}

impl Sensor for RobotJointStateSensor {
    type Output = Observation;

    fn read(&self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_joints * 2);
        let mut query = world.query::<(&JointState, &RobotId)>();
        for (state, &id) in query.iter(world) {
            if id == self.robot_id {
                data.push(state.position);
                data.push(state.velocity);
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
        self.n_joints * 2
    }
}

/// Reads commands only from joints belonging to a specific robot.
pub struct RobotJointCommandSensor {
    robot_id: RobotId,
    n_joints: usize,
}

impl RobotJointCommandSensor {
    pub const fn new(robot_id: RobotId, n_joints: usize) -> Self {
        Self { robot_id, n_joints }
    }
}

impl Sensor for RobotJointCommandSensor {
    type Output = Observation;

    fn read(&self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_joints);
        let mut query = world.query::<(&JointCommand, &RobotId)>();
        for (cmd, &id) in query.iter(world) {
            if id == self.robot_id {
                data.push(cmd.value);
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
        self.n_joints
    }
}

/// Reads torques only from joints belonging to a specific robot.
pub struct RobotJointTorqueSensor {
    robot_id: RobotId,
    n_joints: usize,
}

impl RobotJointTorqueSensor {
    pub const fn new(robot_id: RobotId, n_joints: usize) -> Self {
        Self { robot_id, n_joints }
    }
}

impl Sensor for RobotJointTorqueSensor {
    type Output = Observation;

    fn read(&self, world: &mut World) -> Observation {
        let mut data = Vec::with_capacity(self.n_joints);
        let mut query = world.query::<(&JointTorque, &RobotId)>();
        for (torque, &id) in query.iter(world) {
            if id == self.robot_id {
                data.push(torque.value);
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
        self.n_joints
    }
}

// ---------------------------------------------------------------------------
// NoisySensor
// ---------------------------------------------------------------------------

/// Wraps any [`ObservationSensor`] and adds noise to its output.
///
/// Uses a [`NoiseModel`] applied element-wise.  The RNG must be provided
/// externally for determinism.
pub struct NoisySensor<S: ObservationSensor> {
    inner: S,
    noise: NoiseModel,
}

impl<S: ObservationSensor> NoisySensor<S> {
    pub const fn new(inner: S, noise: NoiseModel) -> Self {
        Self { inner, noise }
    }
}

impl<S: ObservationSensor> Sensor for NoisySensor<S> {
    type Output = Observation;

    fn read(&self, world: &mut World) -> Observation {
        // Read the clean observation; noise is applied via `read_noisy`.
        self.inner.read(world)
    }

    fn name(&self) -> &str {
        self.inner.name()
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
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

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
        spawn_joint(&mut world, 1.0, 2.0, 0.0);
        spawn_joint(&mut world, 3.0, 4.0, 0.0);

        let sensor = JointStateSensor::new(2);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 4);
        let vals: Vec<f32> = obs.as_slice().to_vec();
        assert!(vals.contains(&1.0));
        assert!(vals.contains(&2.0));
        assert!(vals.contains(&3.0));
        assert!(vals.contains(&4.0));
    }

    #[test]
    fn joint_state_sensor_empty_world() {
        let mut world = World::new();
        let sensor = JointStateSensor::new(0);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn joint_state_sensor_dim() {
        let sensor = JointStateSensor::new(5);
        assert_eq!(sensor.observation_dim(), 10);
    }

    #[test]
    fn joint_state_sensor_name() {
        let sensor = JointStateSensor::new(1);
        assert_eq!(sensor.name(), "JointStateSensor");
    }

    // -- JointCommandSensor --

    #[test]
    fn joint_command_sensor_reads() {
        let mut world = World::new();
        spawn_joint(&mut world, 0.0, 0.0, 5.0);
        spawn_joint(&mut world, 0.0, 0.0, 10.0);

        let sensor = JointCommandSensor::new(2);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 2);
        let vals: Vec<f32> = obs.as_slice().to_vec();
        assert!(vals.contains(&5.0));
        assert!(vals.contains(&10.0));
    }

    #[test]
    fn joint_command_sensor_dim() {
        let sensor = JointCommandSensor::new(3);
        assert_eq!(sensor.observation_dim(), 3);
    }

    // -- JointTorqueSensor --

    #[test]
    fn joint_torque_sensor_reads() {
        let mut world = World::new();
        world.spawn(JointTorque { value: 7.5 });
        world.spawn(JointTorque { value: -3.0 });

        let sensor = JointTorqueSensor::new(2);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 2);
        let vals: Vec<f32> = obs.as_slice().to_vec();
        assert!(vals.contains(&7.5));
        assert!(vals.contains(&-3.0));
    }

    // -- NoisySensor --

    #[test]
    fn noisy_sensor_applies_noise() {
        let mut world = World::new();
        spawn_joint(&mut world, 1.0, 2.0, 0.0);

        let noise = NoiseModel::gaussian(0.0, 0.1).unwrap();
        let mut sensor = NoisySensor::new(JointStateSensor::new(1), noise);

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
        spawn_joint(&mut world, 5.0, 10.0, 0.0);

        let noise = NoiseModel::gaussian(0.0, 1.0).unwrap();
        let mut sensor = NoisySensor::new(JointStateSensor::new(1), noise);
        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let obs1 = sensor.read_noisy(&mut world, &mut rng1);

        let noise2 = NoiseModel::gaussian(0.0, 1.0).unwrap();
        let mut sensor2 = NoisySensor::new(JointStateSensor::new(1), noise2);
        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let obs2 = sensor2.read_noisy(&mut world, &mut rng2);

        assert_eq!(obs1.as_slice(), obs2.as_slice());
    }

    #[test]
    fn noisy_sensor_delegates_name_and_dim() {
        let noise = NoiseModel::gaussian(0.0, 0.1).unwrap();
        let sensor = NoisySensor::new(JointStateSensor::new(3), noise);
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
        spawn_robot_joint(&mut world, RobotId(0), 1.0, 2.0, 0.0);
        spawn_robot_joint(&mut world, RobotId(1), 3.0, 4.0, 0.0);
        spawn_robot_joint(&mut world, RobotId(0), 5.0, 6.0, 0.0);

        let sensor = RobotJointStateSensor::new(RobotId(0), 2);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 4);
        let vals: Vec<f32> = obs.as_slice().to_vec();
        assert!(vals.contains(&1.0));
        assert!(vals.contains(&2.0));
        assert!(vals.contains(&5.0));
        assert!(vals.contains(&6.0));
        assert!(!vals.contains(&3.0));
        assert!(!vals.contains(&4.0));
    }

    #[test]
    fn robot_joint_command_sensor_filters_by_id() {
        let mut world = World::new();
        spawn_robot_joint(&mut world, RobotId(0), 0.0, 0.0, 10.0);
        spawn_robot_joint(&mut world, RobotId(1), 0.0, 0.0, 20.0);

        let sensor = RobotJointCommandSensor::new(RobotId(0), 1);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 1);
        assert!((obs[0] - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn robot_joint_torque_sensor_filters_by_id() {
        let mut world = World::new();
        spawn_robot_joint(&mut world, RobotId(0), 5.0, 0.0, 0.0); // torque = 10.0
        spawn_robot_joint(&mut world, RobotId(1), 3.0, 0.0, 0.0); // torque = 6.0

        let sensor = RobotJointTorqueSensor::new(RobotId(1), 1);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 1);
        assert!((obs[0] - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn robot_sensor_empty_when_no_match() {
        let mut world = World::new();
        spawn_robot_joint(&mut world, RobotId(0), 1.0, 2.0, 3.0);

        let sensor = RobotJointStateSensor::new(RobotId(99), 0);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn robot_sensor_name_and_dim() {
        let s1 = RobotJointStateSensor::new(RobotId(0), 3);
        assert_eq!(s1.name(), "RobotJointStateSensor");
        assert_eq!(s1.observation_dim(), 6);

        let s2 = RobotJointCommandSensor::new(RobotId(0), 2);
        assert_eq!(s2.name(), "RobotJointCommandSensor");
        assert_eq!(s2.observation_dim(), 2);

        let s3 = RobotJointTorqueSensor::new(RobotId(0), 4);
        assert_eq!(s3.name(), "RobotJointTorqueSensor");
        assert_eq!(s3.observation_dim(), 4);
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn sensor_types_are_send_sync() {
        assert_send_sync::<JointStateSensor>();
        assert_send_sync::<JointCommandSensor>();
        assert_send_sync::<JointTorqueSensor>();
        assert_send_sync::<RobotJointStateSensor>();
        assert_send_sync::<RobotJointCommandSensor>();
        assert_send_sync::<RobotJointTorqueSensor>();
    }
}
