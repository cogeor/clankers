//! Structure-of-Arrays (`SoA`) buffers for batched `VecEnv` data.
//!
//! Stores observations and done flags in flat column-major layout
//! for efficient batched training: `[num_envs, dim]` for observations,
//! `[num_envs]` for flags.

use clankers_core::types::Observation;

// ---------------------------------------------------------------------------
// VecObsBuffer
// ---------------------------------------------------------------------------

/// Batched observation buffer with shape `[num_envs, obs_dim]`.
///
/// Stores observations for all environments in a single flat `Vec<f32>`.
///
/// # Example
///
/// ```
/// use clankers_core::types::Observation;
/// use clankers_env::vec_buffer::VecObsBuffer;
///
/// let mut buf = VecObsBuffer::new(2, 3);
/// buf.set(0, &Observation::new(vec![1.0, 2.0, 3.0]));
/// assert_eq!(buf.get(0).as_slice(), &[1.0, 2.0, 3.0]);
/// ```
#[derive(Debug, Clone)]
pub struct VecObsBuffer {
    data: Vec<f32>,
    num_envs: usize,
    obs_dim: usize,
}

impl VecObsBuffer {
    /// Create a zeroed buffer for `num_envs` environments with `obs_dim` dimensions.
    #[must_use]
    pub fn new(num_envs: usize, obs_dim: usize) -> Self {
        Self {
            data: vec![0.0; num_envs * obs_dim],
            num_envs,
            obs_dim,
        }
    }

    /// Number of environments.
    #[must_use]
    pub const fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Observation dimension per environment.
    #[must_use]
    pub const fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Set the observation for environment `env_idx`.
    ///
    /// # Panics
    ///
    /// Panics if `env_idx >= num_envs` or observation length != `obs_dim`.
    pub fn set(&mut self, env_idx: usize, obs: &Observation) {
        assert!(env_idx < self.num_envs, "env_idx out of bounds");
        assert_eq!(
            obs.len(),
            self.obs_dim,
            "observation dim mismatch: expected {}, got {}",
            self.obs_dim,
            obs.len()
        );
        let start = env_idx * self.obs_dim;
        self.data[start..start + self.obs_dim].copy_from_slice(obs.as_slice());
    }

    /// Get the observation for environment `env_idx`.
    ///
    /// # Panics
    ///
    /// Panics if `env_idx >= num_envs`.
    #[must_use]
    pub fn get(&self, env_idx: usize) -> Observation {
        assert!(env_idx < self.num_envs, "env_idx out of bounds");
        let start = env_idx * self.obs_dim;
        Observation::new(self.data[start..start + self.obs_dim].to_vec())
    }

    /// Raw flat buffer `[num_envs * obs_dim]`.
    #[must_use]
    pub fn as_flat(&self) -> &[f32] {
        &self.data
    }

    /// Zero out all observations.
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// VecDoneBuffer
// ---------------------------------------------------------------------------

/// Batched done/truncated flag buffer with shape `[num_envs]`.
#[derive(Debug, Clone)]
pub struct VecDoneBuffer {
    terminated: Vec<bool>,
    truncated: Vec<bool>,
}

impl VecDoneBuffer {
    /// Create a buffer with all flags false.
    #[must_use]
    pub fn new(num_envs: usize) -> Self {
        Self {
            terminated: vec![false; num_envs],
            truncated: vec![false; num_envs],
        }
    }

    /// Set done flags for environment `env_idx`.
    pub fn set(&mut self, env_idx: usize, terminated: bool, truncated: bool) {
        self.terminated[env_idx] = terminated;
        self.truncated[env_idx] = truncated;
    }

    /// Get terminated flag for environment `env_idx`.
    #[must_use]
    pub fn terminated(&self, env_idx: usize) -> bool {
        self.terminated[env_idx]
    }

    /// Get truncated flag for environment `env_idx`.
    #[must_use]
    pub fn truncated(&self, env_idx: usize) -> bool {
        self.truncated[env_idx]
    }

    /// Whether any env is done (terminated or truncated) at `env_idx`.
    #[must_use]
    pub fn is_done(&self, env_idx: usize) -> bool {
        self.terminated[env_idx] || self.truncated[env_idx]
    }

    /// Flat terminated flags.
    #[must_use]
    pub fn terminated_flat(&self) -> &[bool] {
        &self.terminated
    }

    /// Flat truncated flags.
    #[must_use]
    pub fn truncated_flat(&self) -> &[bool] {
        &self.truncated
    }

    /// Reset all flags to false.
    pub fn clear(&mut self) {
        self.terminated.fill(false);
        self.truncated.fill(false);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- VecObsBuffer ----

    #[test]
    fn obs_buffer_set_get() {
        let mut buf = VecObsBuffer::new(3, 2);
        buf.set(0, &Observation::new(vec![1.0, 2.0]));
        buf.set(1, &Observation::new(vec![3.0, 4.0]));
        buf.set(2, &Observation::new(vec![5.0, 6.0]));

        assert_eq!(buf.get(0).as_slice(), &[1.0, 2.0]);
        assert_eq!(buf.get(1).as_slice(), &[3.0, 4.0]);
        assert_eq!(buf.get(2).as_slice(), &[5.0, 6.0]);
    }

    #[test]
    fn obs_buffer_flat_layout() {
        let mut buf = VecObsBuffer::new(2, 3);
        buf.set(0, &Observation::new(vec![1.0, 2.0, 3.0]));
        buf.set(1, &Observation::new(vec![4.0, 5.0, 6.0]));

        // Row-major: [env0_obs..., env1_obs...]
        assert_eq!(buf.as_flat(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn obs_buffer_clear() {
        let mut buf = VecObsBuffer::new(2, 2);
        buf.set(0, &Observation::new(vec![1.0, 2.0]));
        buf.clear();
        assert_eq!(buf.get(0).as_slice(), &[0.0, 0.0]);
    }

    #[test]
    fn obs_buffer_dimensions() {
        let buf = VecObsBuffer::new(4, 10);
        assert_eq!(buf.num_envs(), 4);
        assert_eq!(buf.obs_dim(), 10);
    }

    // ---- VecDoneBuffer ----

    #[test]
    fn done_buffer_set_get() {
        let mut buf = VecDoneBuffer::new(3);
        buf.set(0, true, false);
        buf.set(1, false, true);
        buf.set(2, false, false);

        assert!(buf.terminated(0));
        assert!(!buf.truncated(0));
        assert!(!buf.terminated(1));
        assert!(buf.truncated(1));
        assert!(!buf.is_done(2));
    }

    #[test]
    fn done_buffer_is_done() {
        let mut buf = VecDoneBuffer::new(2);
        buf.set(0, true, false);
        buf.set(1, false, true);
        assert!(buf.is_done(0));
        assert!(buf.is_done(1));
    }

    #[test]
    fn done_buffer_clear() {
        let mut buf = VecDoneBuffer::new(2);
        buf.set(0, true, true);
        buf.clear();
        assert!(!buf.terminated(0));
        assert!(!buf.truncated(0));
    }

    #[test]
    fn done_buffer_flat_views() {
        let mut buf = VecDoneBuffer::new(3);
        buf.set(1, true, false);
        assert_eq!(buf.terminated_flat(), &[false, true, false]);
        assert_eq!(buf.truncated_flat(), &[false, false, false]);
    }
}
