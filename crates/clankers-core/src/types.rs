use std::collections::HashMap;

use bevy::ecs::entity::Entity;
use serde::{Deserialize, Serialize};

use crate::error::ValidationError;

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// Flat f32 vector representing environment state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    data: Vec<f32>,
}

impl Observation {
    pub const fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    pub const fn len(&self) -> usize {
        self.data.len()
    }

    pub const fn dim(&self) -> usize {
        self.data.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    pub fn values(&self) -> &[f32] {
        &self.data
    }
}

impl std::ops::Index<usize> for Observation {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        &self.data[i]
    }
}

impl std::ops::IndexMut<usize> for Observation {
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        &mut self.data[i]
    }
}

impl From<Vec<f32>> for Observation {
    fn from(data: Vec<f32>) -> Self {
        Self::new(data)
    }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// Control command sent to the environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    /// Continuous control values (typically normalized to [-1, 1]).
    Continuous(Vec<f32>),
    /// Single discrete choice in [0, n).
    Discrete(u64),
    /// Multiple independent discrete choices.
    MultiDiscrete(Vec<u64>),
}

impl Action {
    /// Create a continuous action. Alias kept for ergonomics.
    pub const fn new(data: Vec<f32>) -> Self {
        Self::Continuous(data)
    }

    /// Continuous action filled with zeros.
    pub fn zeros(len: usize) -> Self {
        Self::Continuous(vec![0.0; len])
    }

    /// Number of scalar elements.
    pub const fn len(&self) -> usize {
        match self {
            Self::Continuous(v) => v.len(),
            Self::Discrete(_) => 1,
            Self::MultiDiscrete(v) => v.len(),
        }
    }

    pub const fn dim(&self) -> usize {
        self.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Slice view for continuous actions. Panics on discrete variants.
    pub fn as_slice(&self) -> &[f32] {
        match self {
            Self::Continuous(v) => v.as_slice(),
            _ => panic!("as_slice() only valid for Action::Continuous"),
        }
    }

    /// Alias for `as_slice()`.
    pub fn values(&self) -> &[f32] {
        self.as_slice()
    }

    /// Mutable slice for continuous actions. Panics on discrete variants.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        match self {
            Self::Continuous(v) => v.as_mut_slice(),
            _ => panic!("as_mut_slice() only valid for Action::Continuous"),
        }
    }

    /// Consume into `Vec<f32>`. Panics on discrete variants.
    pub fn into_vec(self) -> Vec<f32> {
        match self {
            Self::Continuous(v) => v,
            _ => panic!("into_vec() only valid for Action::Continuous"),
        }
    }

    /// Clip continuous values to [-1, 1]. No-op for discrete variants.
    pub fn clip_normalized(&mut self) {
        if let Self::Continuous(v) = self {
            for val in v.iter_mut() {
                *val = val.clamp(-1.0, 1.0);
            }
        }
    }

    /// Scale continuous values from [-1, 1] to [low, high].
    pub fn scale(&self, low: &[f32], high: &[f32]) -> Vec<f32> {
        let s = self.as_slice();
        s.iter()
            .zip(low.iter().zip(high.iter()))
            .map(|(a, (l, h))| l + ((a + 1.0) / 2.0) * (h - l))
            .collect()
    }

    /// Validate action data (no NaN, no Inf in continuous).
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Continuous(v) => {
                for (i, val) in v.iter().enumerate() {
                    if val.is_nan() {
                        return Err(ValidationError::ActionContainsNan);
                    }
                    if val.is_infinite() {
                        return Err(ValidationError::ActionContainsInf);
                    }
                    let _ = i; // dim available if needed for OutOfBounds
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl From<Vec<f32>> for Action {
    fn from(data: Vec<f32>) -> Self {
        Self::Continuous(data)
    }
}

// ---------------------------------------------------------------------------
// ObservationSpace
// ---------------------------------------------------------------------------

/// Defines shape and bounds of valid observations. Follows Gymnasium conventions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObservationSpace {
    Box {
        low: Vec<f32>,
        high: Vec<f32>,
    },
    Discrete {
        n: usize,
    },
    MultiDiscrete {
        nvec: Vec<usize>,
    },
    MultiBinary {
        n: usize,
    },
    Image {
        height: u32,
        width: u32,
        channels: u32,
    },
    Dict {
        spaces: HashMap<String, Self>,
    },
}

impl ObservationSpace {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Self::Box { low, .. } => vec![low.len()],
            Self::Discrete { .. } => vec![1],
            Self::MultiDiscrete { nvec } => vec![nvec.len()],
            Self::MultiBinary { n } => vec![*n],
            Self::Image {
                height,
                width,
                channels,
            } => {
                vec![*height as usize, *width as usize, *channels as usize]
            }
            Self::Dict { .. } => vec![], // composite; query children
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Self::Dict { spaces } => spaces.values().map(Self::size).sum(),
            _ => self.shape().iter().product(),
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn contains(&self, obs: &Observation) -> bool {
        match self {
            Self::Box { low, high } => {
                obs.len() == low.len()
                    && obs
                        .as_slice()
                        .iter()
                        .zip(low.iter().zip(high.iter()))
                        .all(|(v, (l, h))| v >= l && v <= h)
            }
            Self::Discrete { n } => obs.len() == 1 && (obs[0] as usize) < *n,
            #[allow(clippy::float_cmp)]
            Self::MultiBinary { n } => {
                obs.len() == *n && obs.as_slice().iter().all(|v| *v == 0.0 || *v == 1.0)
            }
            _ => true,
        }
    }
}

// ---------------------------------------------------------------------------
// ActionSpace
// ---------------------------------------------------------------------------

/// Defines shape and bounds of valid actions. Follows Gymnasium conventions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionSpace {
    Box { low: Vec<f32>, high: Vec<f32> },
    Discrete { n: usize },
    MultiDiscrete { nvec: Vec<usize> },
    MultiBinary { n: usize },
    Dict { spaces: HashMap<String, Self> },
}

impl ActionSpace {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Self::Box { low, .. } => vec![low.len()],
            Self::Discrete { .. } => vec![1],
            Self::MultiDiscrete { nvec } => vec![nvec.len()],
            Self::MultiBinary { n } => vec![*n],
            Self::Dict { .. } => vec![],
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Self::Dict { spaces } => spaces.values().map(Self::size).sum(),
            _ => self.shape().iter().product(),
        }
    }

    /// Sample a random action. Takes `&mut impl Rng` for determinism.
    #[allow(clippy::cast_possible_truncation)]
    pub fn sample(&self, rng: &mut impl rand::Rng) -> Action {
        match self {
            Self::Box { low, high } => {
                let data: Vec<f32> = low
                    .iter()
                    .zip(high.iter())
                    .map(|(l, h)| rng.gen_range(*l..=*h))
                    .collect();
                Action::Continuous(data)
            }
            Self::Discrete { n } => Action::Discrete(rng.gen_range(0..*n as u64)),
            Self::MultiDiscrete { nvec } => {
                Action::MultiDiscrete(nvec.iter().map(|n| rng.gen_range(0..*n as u64)).collect())
            }
            Self::MultiBinary { n } => Action::Continuous(
                (0..*n)
                    .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
                    .collect(),
            ),
            Self::Dict { .. } => {
                panic!("sample() not supported for Dict spaces; sample each sub-space individually")
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn contains(&self, action: &Action) -> bool {
        match (self, action) {
            (Self::Box { low, high }, Action::Continuous(v)) => {
                v.len() == low.len()
                    && v.iter()
                        .zip(low.iter().zip(high.iter()))
                        .all(|(val, (l, h))| val >= l && val <= h)
            }
            (Self::Discrete { n }, Action::Discrete(v)) => (*v as usize) < *n,
            (Self::MultiDiscrete { nvec }, Action::MultiDiscrete(v)) => {
                v.len() == nvec.len()
                    && v.iter()
                        .zip(nvec.iter())
                        .all(|(val, n)| (*val as usize) < *n)
            }
            _ => false, // type mismatch
        }
    }
}

// ---------------------------------------------------------------------------
// StepResult / ResetResult
// ---------------------------------------------------------------------------

/// Result of `env.step(action)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub observation: Observation,
    pub reward: f32,
    /// Episode ended due to task success/failure.
    pub terminated: bool,
    /// Episode ended due to time limit.
    pub truncated: bool,
    pub info: StepInfo,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepInfo {
    pub episode_length: u32,
    pub episode_reward: f32,
    pub custom: HashMap<String, f32>,
}

/// Result of `env.reset()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetResult {
    pub observation: Observation,
    pub info: ResetInfo,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResetInfo {
    pub seed: Option<u64>,
    pub custom: HashMap<String, f32>,
}

// ---------------------------------------------------------------------------
// Entity Handles
// ---------------------------------------------------------------------------

/// Handle to a robot in the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RobotHandle(pub Entity);

/// Handle to a non-robot object in the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectHandle(pub Entity);

/// Handle to a sensor in the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SensorHandle(pub Entity);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Observation ----

    #[test]
    fn observation_new_and_len() {
        let obs = Observation::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(obs.len(), 3);
        assert_eq!(obs.dim(), 3);
        assert!(!obs.is_empty());
    }

    #[test]
    fn observation_zeros() {
        let obs = Observation::zeros(5);
        assert_eq!(obs.len(), 5);
        assert_eq!(obs.as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn observation_empty() {
        let obs = Observation::new(vec![]);
        assert!(obs.is_empty());
        assert_eq!(obs.len(), 0);
        assert_eq!(obs.dim(), 0);
    }

    #[test]
    fn observation_indexing() {
        let obs = Observation::new(vec![10.0, 20.0, 30.0]);
        assert!((obs[0] - 10.0).abs() < f32::EPSILON);
        assert!((obs[1] - 20.0).abs() < f32::EPSILON);
        assert!((obs[2] - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn observation_index_mut() {
        let mut obs = Observation::new(vec![1.0, 2.0, 3.0]);
        obs[1] = 99.0;
        assert!((obs[1] - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn observation_slicing() {
        let obs = Observation::new(vec![1.0, 2.0]);
        assert_eq!(obs.as_slice(), &[1.0, 2.0]);
        assert_eq!(obs.values(), &[1.0, 2.0]);
    }

    #[test]
    fn observation_mut_slice() {
        let mut obs = Observation::new(vec![1.0, 2.0]);
        obs.as_mut_slice()[0] = 5.0;
        assert!((obs[0] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn observation_into_vec() {
        let obs = Observation::new(vec![1.0, 2.0, 3.0]);
        let v = obs.into_vec();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn observation_from_vec() {
        let obs: Observation = vec![4.0, 5.0].into();
        assert_eq!(obs.len(), 2);
        assert!((obs[0] - 4.0).abs() < f32::EPSILON);
        assert!((obs[1] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn observation_clone_and_eq() {
        let obs1 = Observation::new(vec![1.0, 2.0]);
        let obs2 = obs1.clone();
        assert_eq!(obs1, obs2);
    }

    #[test]
    fn observation_serialize_roundtrip() {
        let obs = Observation::new(vec![1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&obs).unwrap();
        let obs2: Observation = serde_json::from_str(&json).unwrap();
        assert_eq!(obs, obs2);
    }

    // ---- Action ----

    #[test]
    fn action_continuous_new() {
        let action = Action::new(vec![0.5, -0.5]);
        assert_eq!(action.len(), 2);
        assert_eq!(action.dim(), 2);
        assert!(!action.is_empty());
        assert_eq!(action.as_slice(), &[0.5, -0.5]);
    }

    #[test]
    fn action_zeros() {
        let action = Action::zeros(3);
        assert_eq!(action.len(), 3);
        assert_eq!(action.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn action_discrete() {
        let action = Action::Discrete(5);
        assert_eq!(action.len(), 1);
        assert_eq!(action.dim(), 1);
        assert!(!action.is_empty());
    }

    #[test]
    fn action_multi_discrete() {
        let action = Action::MultiDiscrete(vec![1, 2, 3]);
        assert_eq!(action.len(), 3);
        assert_eq!(action.dim(), 3);
    }

    #[test]
    #[should_panic(expected = "as_slice() only valid for Action::Continuous")]
    fn action_discrete_as_slice_panics() {
        let action = Action::Discrete(1);
        let _ = action.as_slice();
    }

    #[test]
    #[should_panic(expected = "as_slice() only valid for Action::Continuous")]
    fn action_discrete_values_panics() {
        let action = Action::Discrete(1);
        let _ = action.values();
    }

    #[test]
    #[should_panic(expected = "as_mut_slice() only valid for Action::Continuous")]
    fn action_discrete_as_mut_slice_panics() {
        let mut action = Action::Discrete(1);
        let _ = action.as_mut_slice();
    }

    #[test]
    #[should_panic(expected = "into_vec() only valid for Action::Continuous")]
    fn action_discrete_into_vec_panics() {
        let action = Action::Discrete(1);
        let _ = action.into_vec();
    }

    #[test]
    #[should_panic(expected = "as_slice() only valid for Action::Continuous")]
    fn action_multi_discrete_as_slice_panics() {
        let action = Action::MultiDiscrete(vec![1, 2]);
        let _ = action.as_slice();
    }

    #[test]
    #[should_panic(expected = "as_mut_slice() only valid for Action::Continuous")]
    fn action_multi_discrete_as_mut_slice_panics() {
        let mut action = Action::MultiDiscrete(vec![1, 2]);
        let _ = action.as_mut_slice();
    }

    #[test]
    #[should_panic(expected = "into_vec() only valid for Action::Continuous")]
    fn action_multi_discrete_into_vec_panics() {
        let action = Action::MultiDiscrete(vec![1, 2]);
        let _ = action.into_vec();
    }

    #[test]
    fn action_clip_normalized_continuous() {
        let mut action = Action::new(vec![-2.0, 0.5, 1.5]);
        action.clip_normalized();
        assert_eq!(action.as_slice(), &[-1.0, 0.5, 1.0]);
    }

    #[test]
    fn action_clip_normalized_no_op_discrete() {
        let mut action = Action::Discrete(3);
        action.clip_normalized(); // should not panic
        assert_eq!(action, Action::Discrete(3));
    }

    #[test]
    fn action_clip_normalized_no_op_multi_discrete() {
        let mut action = Action::MultiDiscrete(vec![1, 2]);
        action.clip_normalized(); // should not panic
        assert_eq!(action, Action::MultiDiscrete(vec![1, 2]));
    }

    #[test]
    fn action_scale() {
        // value -1.0 -> low, 1.0 -> high, 0.0 -> midpoint
        let action = Action::new(vec![-1.0, 0.0, 1.0]);
        let low = [0.0, 0.0, 0.0];
        let high = [10.0, 10.0, 10.0];
        let scaled = action.scale(&low, &high);
        assert!((scaled[0] - 0.0).abs() < f32::EPSILON);
        assert!((scaled[1] - 5.0).abs() < f32::EPSILON);
        assert!((scaled[2] - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn action_validate_ok() {
        let action = Action::new(vec![0.5, -0.3, 1.0]);
        assert!(action.validate().is_ok());
    }

    #[test]
    fn action_validate_nan() {
        let action = Action::new(vec![0.5, f32::NAN, 1.0]);
        let err = action.validate().unwrap_err();
        assert_eq!(err, ValidationError::ActionContainsNan);
    }

    #[test]
    fn action_validate_inf() {
        let action = Action::new(vec![f32::INFINITY, 0.5]);
        let err = action.validate().unwrap_err();
        assert_eq!(err, ValidationError::ActionContainsInf);
    }

    #[test]
    fn action_validate_neg_inf() {
        let action = Action::new(vec![f32::NEG_INFINITY]);
        let err = action.validate().unwrap_err();
        assert_eq!(err, ValidationError::ActionContainsInf);
    }

    #[test]
    fn action_validate_discrete_always_ok() {
        let action = Action::Discrete(42);
        assert!(action.validate().is_ok());
    }

    #[test]
    fn action_validate_multi_discrete_always_ok() {
        let action = Action::MultiDiscrete(vec![1, 2, 3]);
        assert!(action.validate().is_ok());
    }

    #[test]
    fn action_from_vec() {
        let action: Action = vec![1.0, 2.0].into();
        assert_eq!(action, Action::Continuous(vec![1.0, 2.0]));
    }

    #[test]
    fn action_into_vec_roundtrip() {
        let data = vec![1.0, 2.0, 3.0];
        let action = Action::new(data.clone());
        assert_eq!(action.into_vec(), data);
    }

    #[test]
    fn action_as_mut_slice() {
        let mut action = Action::new(vec![1.0, 2.0]);
        action.as_mut_slice()[0] = 99.0;
        assert!((action.as_slice()[0] - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn action_serialize_roundtrip() {
        let action = Action::new(vec![0.1, 0.2]);
        let json = serde_json::to_string(&action).unwrap();
        let action2: Action = serde_json::from_str(&json).unwrap();
        assert_eq!(action, action2);

        let discrete = Action::Discrete(7);
        let json = serde_json::to_string(&discrete).unwrap();
        let discrete2: Action = serde_json::from_str(&json).unwrap();
        assert_eq!(discrete, discrete2);

        let md = Action::MultiDiscrete(vec![1, 2, 3]);
        let json = serde_json::to_string(&md).unwrap();
        let md2: Action = serde_json::from_str(&json).unwrap();
        assert_eq!(md, md2);
    }

    // ---- ObservationSpace ----

    #[test]
    fn obs_space_box_shape_size() {
        let space = ObservationSpace::Box {
            low: vec![-1.0, -1.0, -1.0],
            high: vec![1.0, 1.0, 1.0],
        };
        assert_eq!(space.shape(), vec![3]);
        assert_eq!(space.size(), 3);
    }

    #[test]
    fn obs_space_discrete_shape_size() {
        let space = ObservationSpace::Discrete { n: 10 };
        assert_eq!(space.shape(), vec![1]);
        assert_eq!(space.size(), 1);
    }

    #[test]
    fn obs_space_multi_discrete_shape_size() {
        let space = ObservationSpace::MultiDiscrete {
            nvec: vec![3, 4, 5],
        };
        assert_eq!(space.shape(), vec![3]);
        assert_eq!(space.size(), 3);
    }

    #[test]
    fn obs_space_multi_binary_shape_size() {
        let space = ObservationSpace::MultiBinary { n: 8 };
        assert_eq!(space.shape(), vec![8]);
        assert_eq!(space.size(), 8);
    }

    #[test]
    fn obs_space_image_shape_size() {
        let space = ObservationSpace::Image {
            height: 64,
            width: 64,
            channels: 3,
        };
        assert_eq!(space.shape(), vec![64, 64, 3]);
        assert_eq!(space.size(), 64 * 64 * 3);
    }

    #[test]
    fn obs_space_dict_shape_size() {
        let mut spaces = HashMap::new();
        spaces.insert(
            "pos".to_string(),
            ObservationSpace::Box {
                low: vec![0.0; 3],
                high: vec![1.0; 3],
            },
        );
        spaces.insert("label".to_string(), ObservationSpace::Discrete { n: 5 });
        let space = ObservationSpace::Dict { spaces };
        assert_eq!(space.shape(), Vec::<usize>::new());
        assert_eq!(space.size(), 4); // 3 from box + 1 from discrete
    }

    #[test]
    fn obs_space_box_contains() {
        let space = ObservationSpace::Box {
            low: vec![0.0, 0.0],
            high: vec![1.0, 1.0],
        };
        assert!(space.contains(&Observation::new(vec![0.5, 0.5])));
        assert!(space.contains(&Observation::new(vec![0.0, 1.0])));
        assert!(!space.contains(&Observation::new(vec![-0.1, 0.5])));
        assert!(!space.contains(&Observation::new(vec![0.5, 1.1])));
        // wrong dimension
        assert!(!space.contains(&Observation::new(vec![0.5])));
    }

    #[test]
    fn obs_space_discrete_contains() {
        let space = ObservationSpace::Discrete { n: 5 };
        assert!(space.contains(&Observation::new(vec![0.0])));
        assert!(space.contains(&Observation::new(vec![4.0])));
        assert!(!space.contains(&Observation::new(vec![5.0])));
        // wrong length
        assert!(!space.contains(&Observation::new(vec![0.0, 1.0])));
    }

    #[test]
    fn obs_space_multi_binary_contains() {
        let space = ObservationSpace::MultiBinary { n: 3 };
        assert!(space.contains(&Observation::new(vec![0.0, 1.0, 0.0])));
        assert!(!space.contains(&Observation::new(vec![0.0, 0.5, 1.0])));
        assert!(!space.contains(&Observation::new(vec![0.0, 1.0]))); // wrong length
    }

    // ---- ActionSpace ----

    #[test]
    fn action_space_box_shape_size() {
        let space = ActionSpace::Box {
            low: vec![-1.0, -1.0],
            high: vec![1.0, 1.0],
        };
        assert_eq!(space.shape(), vec![2]);
        assert_eq!(space.size(), 2);
    }

    #[test]
    fn action_space_discrete_shape_size() {
        let space = ActionSpace::Discrete { n: 4 };
        assert_eq!(space.shape(), vec![1]);
        assert_eq!(space.size(), 1);
    }

    #[test]
    fn action_space_multi_discrete_shape_size() {
        let space = ActionSpace::MultiDiscrete {
            nvec: vec![2, 3, 4],
        };
        assert_eq!(space.shape(), vec![3]);
        assert_eq!(space.size(), 3);
    }

    #[test]
    fn action_space_multi_binary_shape_size() {
        let space = ActionSpace::MultiBinary { n: 6 };
        assert_eq!(space.shape(), vec![6]);
        assert_eq!(space.size(), 6);
    }

    #[test]
    fn action_space_dict_shape_size() {
        let mut spaces = HashMap::new();
        spaces.insert(
            "motor".to_string(),
            ActionSpace::Box {
                low: vec![-1.0; 4],
                high: vec![1.0; 4],
            },
        );
        spaces.insert("grip".to_string(), ActionSpace::Discrete { n: 2 });
        let space = ActionSpace::Dict { spaces };
        assert_eq!(space.shape(), Vec::<usize>::new());
        assert_eq!(space.size(), 5); // 4 from box + 1 from discrete
    }

    #[test]
    fn action_space_sample_box() {
        let space = ActionSpace::Box {
            low: vec![-1.0, -2.0],
            high: vec![1.0, 2.0],
        };
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let action = space.sample(&mut rng);
            assert!(space.contains(&action));
            if let Action::Continuous(v) = &action {
                assert_eq!(v.len(), 2);
                assert!(v[0] >= -1.0 && v[0] <= 1.0);
                assert!(v[1] >= -2.0 && v[1] <= 2.0);
            } else {
                panic!("Expected Continuous action from Box space");
            }
        }
    }

    #[test]
    fn action_space_sample_discrete() {
        let space = ActionSpace::Discrete { n: 5 };
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let action = space.sample(&mut rng);
            assert!(space.contains(&action));
            if let Action::Discrete(v) = action {
                assert!(v < 5);
            } else {
                panic!("Expected Discrete action from Discrete space");
            }
        }
    }

    #[test]
    fn action_space_sample_multi_discrete() {
        let space = ActionSpace::MultiDiscrete {
            nvec: vec![3, 5, 2],
        };
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let action = space.sample(&mut rng);
            assert!(space.contains(&action));
            if let Action::MultiDiscrete(v) = &action {
                assert_eq!(v.len(), 3);
                assert!(v[0] < 3);
                assert!(v[1] < 5);
                assert!(v[2] < 2);
            } else {
                panic!("Expected MultiDiscrete action from MultiDiscrete space");
            }
        }
    }

    #[test]
    fn action_space_sample_multi_binary() {
        let space = ActionSpace::MultiBinary { n: 4 };
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let action = space.sample(&mut rng);
            if let Action::Continuous(v) = &action {
                assert_eq!(v.len(), 4);
                for val in v {
                    assert!(val.abs() < f32::EPSILON || (*val - 1.0).abs() < f32::EPSILON);
                }
            } else {
                panic!("Expected Continuous action from MultiBinary space");
            }
        }
    }

    #[test]
    #[should_panic(expected = "sample() not supported for Dict spaces")]
    fn action_space_sample_dict_panics() {
        let space = ActionSpace::Dict {
            spaces: HashMap::new(),
        };
        let mut rng = rand::thread_rng();
        let _ = space.sample(&mut rng);
    }

    #[test]
    fn action_space_box_contains() {
        let space = ActionSpace::Box {
            low: vec![0.0, 0.0],
            high: vec![1.0, 1.0],
        };
        assert!(space.contains(&Action::Continuous(vec![0.5, 0.5])));
        assert!(space.contains(&Action::Continuous(vec![0.0, 1.0])));
        assert!(!space.contains(&Action::Continuous(vec![-0.1, 0.5])));
        assert!(!space.contains(&Action::Continuous(vec![0.5, 1.1])));
        // wrong type
        assert!(!space.contains(&Action::Discrete(0)));
        // wrong dimension
        assert!(!space.contains(&Action::Continuous(vec![0.5])));
    }

    #[test]
    fn action_space_discrete_contains() {
        let space = ActionSpace::Discrete { n: 3 };
        assert!(space.contains(&Action::Discrete(0)));
        assert!(space.contains(&Action::Discrete(2)));
        assert!(!space.contains(&Action::Discrete(3)));
        // wrong type
        assert!(!space.contains(&Action::Continuous(vec![0.0])));
    }

    #[test]
    fn action_space_multi_discrete_contains() {
        let space = ActionSpace::MultiDiscrete { nvec: vec![3, 5] };
        assert!(space.contains(&Action::MultiDiscrete(vec![0, 4])));
        assert!(!space.contains(&Action::MultiDiscrete(vec![3, 0])));
        assert!(!space.contains(&Action::MultiDiscrete(vec![0, 5])));
        // wrong length
        assert!(!space.contains(&Action::MultiDiscrete(vec![0])));
        // wrong type
        assert!(!space.contains(&Action::Discrete(0)));
    }

    // ---- StepResult / ResetResult ----

    #[test]
    fn step_result_construction() {
        let result = StepResult {
            observation: Observation::new(vec![1.0, 2.0]),
            reward: 1.5,
            terminated: false,
            truncated: true,
            info: StepInfo {
                episode_length: 100,
                episode_reward: 50.0,
                custom: HashMap::new(),
            },
        };
        assert!((result.reward - 1.5).abs() < f32::EPSILON);
        assert!(!result.terminated);
        assert!(result.truncated);
        assert_eq!(result.info.episode_length, 100);
    }

    #[test]
    fn step_result_serialize_roundtrip() {
        let mut custom = HashMap::new();
        custom.insert("distance".to_string(), 0.42);
        let result = StepResult {
            observation: Observation::new(vec![1.0]),
            reward: 0.5,
            terminated: true,
            truncated: false,
            info: StepInfo {
                episode_length: 50,
                episode_reward: 25.0,
                custom,
            },
        };
        let json = serde_json::to_string(&result).unwrap();
        let result2: StepResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.observation, result2.observation);
        assert!((result.reward - result2.reward).abs() < f32::EPSILON);
        assert_eq!(result.terminated, result2.terminated);
        assert_eq!(result.truncated, result2.truncated);
        assert_eq!(result.info.episode_length, result2.info.episode_length);
    }

    #[test]
    fn reset_result_construction() {
        let result = ResetResult {
            observation: Observation::zeros(4),
            info: ResetInfo {
                seed: Some(42),
                custom: HashMap::new(),
            },
        };
        assert_eq!(result.observation.len(), 4);
        assert_eq!(result.info.seed, Some(42));
    }

    #[test]
    fn reset_result_serialize_roundtrip() {
        let result = ResetResult {
            observation: Observation::new(vec![0.0, 1.0]),
            info: ResetInfo {
                seed: None,
                custom: HashMap::new(),
            },
        };
        let json = serde_json::to_string(&result).unwrap();
        let result2: ResetResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.observation, result2.observation);
        assert_eq!(result.info.seed, result2.info.seed);
    }

    #[test]
    fn step_info_default() {
        let info = StepInfo::default();
        assert_eq!(info.episode_length, 0);
        assert!((info.episode_reward - 0.0).abs() < f32::EPSILON);
        assert!(info.custom.is_empty());
    }

    #[test]
    fn reset_info_default() {
        let info = ResetInfo::default();
        assert_eq!(info.seed, None);
        assert!(info.custom.is_empty());
    }

    // ---- Entity Handles ----

    #[test]
    fn robot_handle_copy_semantics() {
        let entity = Entity::from_bits(42);
        let handle = RobotHandle(entity);
        let handle2 = handle; // Copy
        let handle3 = handle; // Still valid because Copy
        assert_eq!(handle2, handle3);
        assert_eq!(handle.0, entity);
    }

    #[test]
    fn object_handle_copy_semantics() {
        let entity = Entity::from_bits(99);
        let handle = ObjectHandle(entity);
        let handle2 = handle;
        let handle3 = handle;
        assert_eq!(handle2, handle3);
        assert_eq!(handle.0, entity);
    }

    #[test]
    fn sensor_handle_copy_semantics() {
        let entity = Entity::from_bits(7);
        let handle = SensorHandle(entity);
        let handle2 = handle;
        let handle3 = handle;
        assert_eq!(handle2, handle3);
        assert_eq!(handle.0, entity);
    }

    #[test]
    fn handles_hash() {
        use std::collections::HashSet;
        let e1 = Entity::from_bits(1);
        let e2 = Entity::from_bits(2);
        let mut set = HashSet::new();
        set.insert(RobotHandle(e1));
        set.insert(RobotHandle(e2));
        set.insert(RobotHandle(e1)); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn handles_debug_format() {
        let entity = Entity::from_bits(10);
        let r = format!("{:?}", RobotHandle(entity));
        assert!(r.contains("RobotHandle"));
        let o = format!("{:?}", ObjectHandle(entity));
        assert!(o.contains("ObjectHandle"));
        let s = format!("{:?}", SensorHandle(entity));
        assert!(s.contains("SensorHandle"));
    }
}
