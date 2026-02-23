//! Mathematical noise models for sensor simulation, action perturbation, and
//! domain randomization.
//!
//! `clankers-noise` provides reusable noise primitives for sim-to-real transfer.
//! All noise sampling requires an explicit RNG parameter for deterministic,
//! reproducible simulations.
//!
//! # Architecture
//!
//! - [`NoiseModel`](model::NoiseModel) is an enum with six variants (Gaussian,
//!   Uniform, Bias, Drift, Quantization, Chain).  All dispatch is static via
//!   `match` — no trait objects.
//! - [`VectorNoiseModel`](vector::VectorNoiseModel) wraps scalar models for
//!   multi-axis sensors.
//! - [`presets`] provides ready-made noise configurations for MEMS IMUs,
//!   encoders, and force sensors.
//!
//! # Quick Start
//!
//! ```
//! use clankers_noise::prelude::*;
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha8Rng;
//!
//! let mut rng = ChaCha8Rng::seed_from_u64(42);
//! let mut noise = NoiseModel::gaussian_zero_mean(0.01).unwrap();
//! let noisy = 1.0 + noise.sample(&mut rng);
//! ```

pub mod model;
pub mod presets;
pub mod vector;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::model::{NoiseError, NoiseModel};
    pub use crate::presets;
    pub use crate::vector::{IndependentAxesNoise, VectorNoiseModel};
}
