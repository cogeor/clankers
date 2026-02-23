# Rust Best Practices for Clankerss

A comprehensive reference for writing production-quality Rust in the Clankerss robotics
simulation library. Every contributor and AI agent working on this codebase must follow
these conventions.

**Workspace:** 13 crates under `crates/` plus apps under `apps/`
**Stack:** Bevy 0.17.3, bevy_rapier3d 0.32, Edition 2024, MSRV 1.88+
**License:** MIT
**Performance target:** 100k+ physics steps/sec, deterministic

---

## Table of Contents

1. [Code Organization](#1-code-organization)
2. [Type Design](#2-type-design)
3. [Error Handling](#3-error-handling)
4. [Memory and Performance](#4-memory-and-performance)
5. [Concurrency and Thread Safety](#5-concurrency-and-thread-safety)
6. [Determinism and Reproducibility](#6-determinism-and-reproducibility)
7. [Bevy ECS Conventions](#7-bevy-ecs-conventions)
8. [Testing](#8-testing)
9. [Documentation](#9-documentation)
10. [CI/CD and Linting](#10-cicd-and-linting)
11. [API Design](#11-api-design)
12. [Dependencies](#12-dependencies)

---

## 1. Code Organization

### Module Structure

Each source file addresses exactly one concern. A crate's `lib.rs` re-exports the
public surface and contains no logic beyond `mod` declarations and `pub use` items.

```
crates/clankers-core/src/
    lib.rs          # mod declarations, pub use, prelude
    error.rs        # ClankersError enum
    config.rs       # SimConfig, builder
    time.rs         # SimTime, Tick, Duration
    types.rs        # JointId, LinkId, EntityHandle
    traits.rs       # Steppable, Observable, Actuatable
    plugin.rs       # CorePlugin
```

A file should not exceed roughly 500 lines. When it grows beyond that, split it into a
submodule directory:

```
crates/clankers-core/src/
    config/
        mod.rs      # pub use, shared types
        builder.rs  # SimConfigBuilder
        validate.rs # validation logic
```

### Prelude Pattern

Every crate exposes a `prelude` module that contains the types most consumers need.
Downstream crates import the prelude rather than cherry-picking individual types.

```rust
// crates/clankers-core/src/lib.rs
pub mod prelude {
    pub use crate::config::SimConfig;
    pub use crate::error::{ClankersError, ClankersResult};
    pub use crate::time::{SimTime, Tick};
    pub use crate::traits::{Actuatable, Observable, Steppable};
    pub use crate::types::{EntityHandle, JointId, LinkId};
}
```

```rust
// downstream crate
use clankers_core::prelude::*;
```

Only re-export types that belong to the crate's public API. Internal helpers, builder
intermediates, and error variants that consumers never match on stay out of the prelude.

### Visibility Rules

| Visibility    | When to use                                              |
|---------------|----------------------------------------------------------|
| `pub`         | Part of the crate's external API, documented, stable     |
| `pub(crate)`  | Shared between modules inside the same crate             |
| private       | Default. Implementation details within a single module   |

Never use `pub(super)` -- it creates confusing coupling. If two sibling modules share a
type, promote it to `pub(crate)`.

Fields of public structs are private by default. Expose them only through accessor
methods or make the struct `#[non_exhaustive]` so fields can be added without breaking
changes.

```rust
/// Simulation configuration. Constructed via [`SimConfigBuilder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SimConfig {
    dt_ns: u64,
    gravity: [f32; 3],
    seed: u64,
}

impl SimConfig {
    /// Physics timestep in nanoseconds.
    pub fn dt_ns(&self) -> u64 {
        self.dt_ns
    }
}
```

### Feature Flag Conventions

Feature names are lowercase with hyphens. Each optional heavyweight dependency gets its
own feature. Features are additive only -- enabling a feature must never remove
functionality.

```toml
[features]
default = []
cuda = ["dep:cust"]
tensorrt = ["dep:tensorrt-rs", "cuda"]
render = ["dep:bevy/default"]
```

Guard feature-gated code with `cfg` attributes, not `cfg!` macros, so unused code is
excluded from compilation entirely:

```rust
#[cfg(feature = "cuda")]
mod cuda_backend;

#[cfg(feature = "cuda")]
pub use cuda_backend::CudaPolicy;
```

---

## 2. Type Design

### Copy vs Clone

| Use `Copy`  | Use `Clone` (not `Copy`)                          |
|-------------|----------------------------------------------------|
| IDs, handles, indices, ticks | Strings, Vecs, Configs, anything heap-allocated |
| Small structs (up to ~64 bytes) of Copy fields | Anything containing a non-Copy field |
| Newtypes around primitive numerics | Smart pointers (`Arc<T>`, `Box<T>`) |

When in doubt, derive `Clone` only. `Copy` is a permanent API commitment -- removing it
later is a breaking change.

### Enum vs Trait Object vs Generics

| Approach      | When to use                                                    |
|---------------|----------------------------------------------------------------|
| Enum          | Finite, known set of variants. Preferred in Clankerss.          |
| Generics      | Algorithmic abstraction where monomorphization is acceptable   |
| Trait object  | Plugin-style extensibility where variants are truly open-ended |

Prefer enums for anything where the full set of variants is known at compile time. Enums
are exhaustively matchable, cheaper than dynamic dispatch, and friendlier to
serialization.

```rust
/// Actuator command. Finite set, use an enum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActuatorCommand {
    Position(f32),
    Velocity(f32),
    Torque(f32),
}

/// Policy inference. Open-ended, use a trait.
pub trait PolicyBackend: Send + Sync {
    fn infer(&self, obs: &Observation) -> ClankersResult<Action>;
}
```

### Newtype Pattern

Wrap raw primitives to prevent misuse. A `JointId` and a `LinkId` are both `u32`
internally, but they must never be confused.

```rust
/// Identifies a joint within an articulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JointId(u32);

impl JointId {
    /// Creates a new joint identifier.
    ///
    /// # Errors
    ///
    /// Returns `ClankersError::InvalidId` if `raw` exceeds the maximum joint count.
    pub fn new(raw: u32) -> ClankersResult<Self> {
        if raw > MAX_JOINTS {
            return Err(ClankersError::InvalidId {
                kind: "joint",
                value: raw,
            });
        }
        Ok(Self(raw))
    }

    /// Returns the raw numeric value.
    pub fn raw(self) -> u32 {
        self.0
    }
}
```

The newtype derives `Copy` because `u32` is `Copy`, derives `Hash` for use in maps, and
validates at construction so all downstream code can trust the invariant.

### Standard Derive Conventions

Derive traits in a consistent order on every type. Only derive what the type genuinely
needs.

```rust
// Full ceremony for a public value type:
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]

// For a type used as a map key:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]

// For an internal-only helper:
#[derive(Debug, Clone)]
```

**Mandatory for all public types:** `Debug`. There is no exception.

**Mandatory for all serializable types:** `Serialize, Deserialize` from serde.

**Never derive `Default` on types where a "zero" value is not meaningful.** A
`SimConfig` with `dt_ns = 0` is nonsensical. Force construction through a builder or
constructor.

### Serde Conventions

Use `#[serde(rename_all = "snake_case")]` on enums so serialized output matches Rust
naming. Use `#[serde(deny_unknown_fields)]` on config structs to catch typos.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimConfig {
    pub dt_ns: u64,
    #[serde(default = "default_gravity")]
    pub gravity: [f32; 3],
    pub seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlMode {
    Position,
    Velocity,
    Torque,
}

fn default_gravity() -> [f32; 3] {
    [0.0, -9.81, 0.0]
}
```

---

## 3. Error Handling

### Library Errors with thiserror

Every crate defines its own error enum using `thiserror`. The root crate
(`clankers-core`) defines `ClankersError` and `ClankersResult<T>`. Other crates define
crate-specific errors that convert into `ClankersError` via `#[from]`.

```rust
// crates/clankers-core/src/error.rs
use thiserror::Error;

/// Top-level error type for the Clankerss library.
#[derive(Debug, Error)]
pub enum ClankersError {
    #[error("invalid {kind} id: {value}")]
    InvalidId { kind: &'static str, value: u32 },

    #[error("configuration error: {0}")]
    Config(String),

    #[error("physics error: {0}")]
    Physics(String),

    #[error("URDF parse error: {0}")]
    Urdf(#[from] UrdfError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Convenience alias used throughout the library.
pub type ClankersResult<T> = Result<T, ClankersError>;
```

```rust
// crates/clankers-urdf/src/error.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum UrdfError {
    #[error("missing required element <{element}> in URDF")]
    MissingElement { element: String },

    #[error("invalid joint axis [{0}, {1}, {2}]: must be unit vector")]
    InvalidAxis(f32, f32, f32),
}
```

### Rules

1. **Never use `anyhow` in library code.** `anyhow` erases type information. It is
   acceptable only in binary targets (apps) and test helpers.

2. **Never panic in library code.** No `unwrap()`, no `expect()`, no `panic!()`, no
   `unreachable!()` on reachable paths. Use `Result` for fallible operations and
   `debug_assert!` for invariants that should be impossible.

3. **One error enum per crate.** Crate-level errors convert into `ClankersError` via the
   `#[from]` attribute. Do not define separate error types per module unless the module
   is very large and the errors are truly disjoint.

4. **Validate at construction time.** If a value can be invalid, reject it in the
   constructor or builder. Once a type is constructed, all code that holds it can trust
   the invariants without re-checking.

```rust
impl SimConfigBuilder {
    pub fn build(self) -> ClankersResult<SimConfig> {
        if self.dt_ns == 0 {
            return Err(ClankersError::Config(
                "timestep dt_ns must be positive".into(),
            ));
        }
        if self.dt_ns > 1_000_000_000 {
            return Err(ClankersError::Config(
                "timestep dt_ns exceeds 1 second".into(),
            ));
        }
        Ok(SimConfig {
            dt_ns: self.dt_ns,
            gravity: self.gravity,
            seed: self.seed,
        })
    }
}
```

5. **Use `#[must_use]` on Result-returning functions.** Clippy enforces this by default
   for `Result`, but adding the attribute explicitly documents intent and triggers
   warnings even if clippy is bypassed.

```rust
#[must_use = "this returns the result of the operation, not modifying in place"]
pub fn validate(&self) -> ClankersResult<()> { ... }
```

---

## 4. Memory and Performance

### Pre-allocate and Reuse Buffers

Allocations in hot loops destroy throughput. Pre-allocate buffers at initialization and
reuse them via `clear()`, which deallocates nothing.

```rust
pub struct PhysicsSolver {
    /// Reusable scratch buffer for constraint forces.
    force_scratch: Vec<f32>,
}

impl PhysicsSolver {
    pub fn new(max_contacts: usize) -> Self {
        Self {
            force_scratch: Vec::with_capacity(max_contacts * 3),
        }
    }

    pub fn solve(&mut self, contacts: &[Contact]) {
        self.force_scratch.clear(); // length = 0, capacity unchanged
        self.force_scratch
            .extend(contacts.iter().map(|c| c.normal_force));
        // ... use force_scratch ...
    }
}
```

### Function Signatures: Borrow over Own

Accept borrowed slices in function signatures. This avoids forcing callers to allocate
and lets the function work with any contiguous storage.

```rust
// Good: accepts any contiguous f32 storage
pub fn compute_reward(joint_positions: &[f32], target: &[f32]) -> f32 { ... }

// Bad: forces callers to own a Vec
pub fn compute_reward(joint_positions: Vec<f32>, target: Vec<f32>) -> f32 { ... }
```

When returning data, return `Vec<T>` if the caller will own it. Return `&[T]` if the
data lives in the struct.

### Small Data: Copy, Not Arc

For types that are 64 bytes or smaller and contain only `Copy` fields, derive `Copy` and
pass by value. The copy is cheaper than the indirection of a reference or the atomic
reference counting of `Arc`.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointState {
    pub position: f32,
    pub velocity: f32,
    pub effort: f32,
}
```

### Expensive Shared Data: Arc

For large read-only data shared across systems (mesh data, policy weights, URDF trees),
wrap in `Arc`. Never wrap mutable data in `Arc<Mutex<T>>` without strong justification.

```rust
/// Parsed robot description, shared by physics and rendering.
#[derive(Debug, Clone)]
pub struct RobotDescription {
    inner: Arc<RobotDescriptionInner>,
}
```

### SoA vs AoS

For batch operations over many entities (100+ joints, 1000+ contacts), prefer
Structure-of-Arrays (SoA) over Array-of-Structures (AoS). SoA enables SIMD, improves
cache locality for per-field iteration, and aligns with ECS principles.

```rust
// AoS: poor cache locality when iterating only positions
struct JointStates {
    joints: Vec<JointState>, // position, velocity, effort interleaved
}

// SoA: each field is contiguous in memory
struct JointStates {
    positions: Vec<f32>,
    velocities: Vec<f32>,
    efforts: Vec<f32>,
}

impl JointStates {
    fn with_capacity(n: usize) -> Self {
        Self {
            positions: Vec::with_capacity(n),
            velocities: Vec::with_capacity(n),
            efforts: Vec::with_capacity(n),
        }
    }

    fn clear(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.efforts.clear();
    }
}
```

### Zero-Copy Patterns

Use indices and slices instead of cloning data between subsystems. An `Observation` can
reference a slice of a flat buffer rather than owning a copy.

```rust
/// A view into the observation buffer. Does not own the data.
pub struct ObservationSlice<'a> {
    pub joint_positions: &'a [f32],
    pub joint_velocities: &'a [f32],
    pub imu_readings: &'a [f32],
}
```

When lifetimes become unwieldy, use index ranges into a shared flat buffer:

```rust
pub struct ObservationLayout {
    pub joint_pos_range: std::ops::Range<usize>,
    pub joint_vel_range: std::ops::Range<usize>,
    pub imu_range: std::ops::Range<usize>,
    pub total_size: usize,
}

impl ObservationLayout {
    pub fn joint_positions<'a>(&self, buffer: &'a [f32]) -> &'a [f32] {
        &buffer[self.joint_pos_range.clone()]
    }
}
```

### Avoid Allocations in Hot Loops Checklist

- No `format!()` or string building.
- No `Vec::new()` -- use pre-allocated scratch buffers.
- No `Box::new()` -- use arena allocation or stack if possible.
- No `HashMap` lookups in per-step code -- use flat arrays indexed by entity ID.
- No trait objects in per-step code -- use enums or generics for static dispatch.
- Profile with `cargo flamegraph` before optimizing. Measure, do not guess.

---

## 5. Concurrency and Thread Safety

### Send + Sync on All Public Types

Every public type must be `Send + Sync`. This is required for Bevy's multithreaded
scheduler to move components and resources across threads. Add static assertions in
tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn public_types_are_send_sync() {
        assert_send_sync::<SimConfig>();
        assert_send_sync::<JointId>();
        assert_send_sync::<ClankersError>();
    }
}
```

If a type cannot be `Send + Sync`, it must not be public. Redesign it.

### No Interior Mutability Without Justification

`Cell`, `RefCell`, `Mutex`, and `RwLock` hide mutation from the type system. In an ECS
architecture, mutation is explicit through system parameters (`&mut T`, `ResMut<T>`).
Interior mutability undermines Bevy's change detection and parallel scheduling.

Use interior mutability only when:
- Caching a computed value (lazy initialization with `OnceLock`).
- Interfacing with a C library that requires `&self` but mutates internal state.

Document every use of interior mutability with a comment explaining why it is necessary.

### Channel-Based Communication

When subsystems must communicate outside the ECS, prefer channels over shared mutexes.
Channels decouple producers from consumers and avoid deadlocks.

```rust
use std::sync::mpsc;

pub struct TelemetryRecorder {
    sender: mpsc::Sender<TelemetryEvent>,
}

pub struct TelemetryWriter {
    receiver: mpsc::Receiver<TelemetryEvent>,
}
```

For high-throughput scenarios, use `crossbeam-channel` (if added as a dependency) or
Bevy's built-in event system.

### Bevy's Task Pool

For parallel computation that does not fit the ECS system model (batch policy inference,
parallel environment stepping), use Bevy's `ComputeTaskPool`:

```rust
use bevy::tasks::ComputeTaskPool;

fn parallel_reset(
    pool: Res<ComputeTaskPool>,
    mut envs: Query<&mut Environment>,
) {
    pool.scope(|scope| {
        for mut env in &mut envs {
            scope.spawn(async move {
                env.reset();
            });
        }
    });
}
```

Never spawn raw `std::thread` instances. Bevy manages thread pools; spawning additional
threads wastes resources and bypasses the scheduler.

---

## 6. Determinism and Reproducibility

Determinism is non-negotiable. Given the same seed and configuration, a simulation must
produce bit-identical results across runs, platforms, and thread counts.

### RNG: ChaCha8Rng with Explicit Seeds

Use `ChaCha8Rng` from `rand_chacha`. It is deterministic, fast, and portable. Never use
`thread_rng()`, `StdRng::from_entropy()`, or any entropy-seeded generator.

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct SimRng {
    rng: ChaCha8Rng,
}

impl SimRng {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}
```

### Seed Hierarchy

Seeds flow deterministically from a root seed. Each subsystem derives its seed from the
parent, ensuring that adding a new sensor does not change the noise of existing sensors.

```
run_seed (from user)
  |
  +-- env_seed = hash(run_seed, env_index)
       |
       +-- world_seed = hash(env_seed, "world")
       |    |
       |    +-- physics_seed = hash(world_seed, "physics")
       |    +-- domain_rand_seed = hash(world_seed, "domain_rand")
       |
       +-- sensor_seed = hash(env_seed, "sensors")
            |
            +-- imu_seed = hash(sensor_seed, "imu")
            +-- joint_encoder_seed = hash(sensor_seed, "joint_encoder")
```

Implement the hierarchy with a stable hash function:

```rust
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Derives a child seed from a parent seed and a label.
/// Deterministic: same inputs always produce the same output.
pub fn derive_seed(parent: u64, label: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    parent.hash(&mut hasher);
    label.hash(&mut hasher);
    hasher.finish()
}
```

### Integer Time

Floating-point accumulation drifts. Use integer nanoseconds for all time tracking.

```rust
/// Simulation time in integer nanoseconds. Never floating-point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SimTime {
    nanos: u64,
}

impl SimTime {
    pub const ZERO: Self = Self { nanos: 0 };

    pub fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }

    pub fn from_secs_f64(secs: f64) -> Self {
        Self {
            nanos: (secs * 1_000_000_000.0) as u64,
        }
    }

    pub fn as_secs_f64(self) -> f64 {
        self.nanos as f64 / 1_000_000_000.0
    }

    pub fn advance(self, dt_ns: u64) -> Self {
        Self {
            nanos: self.nanos + dt_ns,
        }
    }
}
```

Convert to `f64` only at the boundary when passing to rendering or logging. Never
accumulate floating-point time.

### Determinism Checklist

- No `HashMap` iteration order in logic (use `BTreeMap` or sorted keys).
- No `thread_rng()` or `StdRng::from_entropy()`.
- No `Instant::now()` in simulation logic.
- No `par_iter()` that mutates shared state.
- No floating-point time accumulation.
- Test determinism explicitly: run twice with the same seed, assert bitwise equality.

---

## 7. Bevy ECS Conventions

### Component vs Resource Decision Matrix

| Data belongs to...             | Use          | Example                           |
|-------------------------------|--------------|-----------------------------------|
| A single entity (robot, joint) | `Component`  | `JointState`, `LinkPose`          |
| The simulation as a whole      | `Resource`   | `SimConfig`, `SimTime`, `SimRng`  |
| Communication between systems  | `Event`      | `EpisodeReset`, `RewardSignal`    |

### SystemSet for Ordering

All Clankerss systems register into `ClankersSet` variants for deterministic ordering:

```rust
use bevy::prelude::*;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClankersSet {
    /// Read inputs, apply commands.
    Input,
    /// Step physics.
    Physics,
    /// Read sensors, compute observations.
    Observe,
    /// Compute rewards, check termination.
    Evaluate,
    /// Domain randomization, resets.
    Reset,
}

impl Plugin for CorePlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            FixedUpdate,
            (
                ClankersSet::Input,
                ClankersSet::Physics,
                ClankersSet::Observe,
                ClankersSet::Evaluate,
                ClankersSet::Reset,
            )
                .chain(),
        );
    }
}
```

Systems declare their set membership explicitly:

```rust
app.add_systems(
    FixedUpdate,
    apply_joint_commands.in_set(ClankersSet::Input),
);
```

### Change Detection

Use `Added<T>` and `Changed<T>` filters to avoid redundant work:

```rust
fn on_robot_spawned(
    query: Query<(Entity, &RobotDescription), Added<RobotDescription>>,
) {
    for (entity, desc) in &query {
        // Only runs the first frame this component exists.
        info!("Robot spawned: {:?}", entity);
    }
}
```

Never poll for changes manually. Bevy's change detection is zero-cost when nothing
changes.

### Plugin Pattern

Each crate exposes exactly one plugin (or a small set of related plugins). The plugin
registers all systems, resources, events, and components for that crate.

```rust
// crates/clankers-env/src/plugin.rs
pub struct EnvPlugin;

impl Plugin for EnvPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<EpisodeReset>()
            .init_resource::<EpisodeState>()
            .add_systems(
                FixedUpdate,
                (
                    check_termination.in_set(ClankersSet::Evaluate),
                    reset_environment
                        .in_set(ClankersSet::Reset)
                        .run_if(on_event::<EpisodeReset>),
                ),
            );
    }
}
```

The top-level `clankers-sim` crate composes all plugins:

```rust
// crates/clankers-sim/src/lib.rs
pub struct ClankersPlugins;

impl PluginGroup for ClankersPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(CorePlugin)
            .add(EnvPlugin)
            .add(ActuatorPlugin)
            .add(NoisePlugin)
            .add(DomainRandPlugin)
            .add(PolicyPlugin)
    }
}
```

---

## 8. Testing

### Unit Tests

Place unit tests in a `#[cfg(test)]` module at the bottom of the source file they test.
Unit tests have access to `pub(crate)` items, so they can test internals.

```rust
// crates/clankers-core/src/time.rs

pub struct SimTime { ... }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_adds_dt() {
        let t = SimTime::from_nanos(1_000_000);
        let t2 = t.advance(500_000);
        assert_eq!(t2.as_nanos(), 1_500_000);
    }

    #[test]
    fn from_secs_f64_round_trips() {
        let t = SimTime::from_secs_f64(1.5);
        let diff = (t.as_secs_f64() - 1.5).abs();
        assert!(diff < 1e-9);
    }
}
```

### Integration Tests

Place integration tests in `tests/` at the crate root. They test the public API only
and verify cross-module interactions.

```rust
// crates/clankers-core/tests/config_validation.rs
use clankers_core::prelude::*;

#[test]
fn zero_timestep_rejected() {
    let result = SimConfigBuilder::new().dt_ns(0).build();
    assert!(result.is_err());
}
```

### Workspace-Level Integration Tests

Full-stack tests that exercise multiple crates live in the top-level `tests/` directory.

```
tests/
    determinism.rs      # Two runs with same seed produce identical output
    episode_lifecycle.rs # Spawn, step, reset, step
```

### Parameterized Tests with rstest

Use `rstest` for test cases that vary inputs:

```rust
use rstest::rstest;

#[rstest]
#[case(0.001, 1_000_000)]
#[case(0.01, 10_000_000)]
#[case(1.0, 1_000_000_000)]
fn secs_to_nanos(#[case] secs: f64, #[case] expected_ns: u64) {
    let t = SimTime::from_secs_f64(secs);
    assert_eq!(t.as_nanos(), expected_ns);
}
```

### Determinism Tests

Every simulation feature must include a determinism test. Run the same scenario twice
with the same seed and assert bitwise equality:

```rust
#[test]
fn physics_is_deterministic() {
    let seed = 42;
    let result_a = run_simulation(seed, 1000);
    let result_b = run_simulation(seed, 1000);
    assert_eq!(result_a.joint_positions, result_b.joint_positions);
    assert_eq!(result_a.rewards, result_b.rewards);
}
```

### Shared Fixtures

The `clankers-test-utils` crate provides shared test fixtures, builders, and assertion
helpers. It is a `[dev-dependencies]` only crate and must never appear in non-test code.

```rust
// crates/clankers-test-utils/src/lib.rs
pub fn default_test_config() -> SimConfig {
    SimConfigBuilder::new()
        .dt_ns(1_000_000) // 1ms
        .seed(12345)
        .build()
        .expect("test config is valid")
}

pub fn assert_deterministic<F>(f: F, seed: u64, steps: usize)
where
    F: Fn(u64, usize) -> Vec<f32>,
{
    let a = f(seed, steps);
    let b = f(seed, steps);
    assert_eq!(a, b, "simulation is not deterministic");
}
```

### Test Naming

Test names describe the scenario and expected outcome, separated by underscores:

```rust
#[test]
fn joint_id_rejects_value_above_maximum() { ... }

#[test]
fn sim_time_advance_does_not_overflow_at_max() { ... }

#[test]
fn noise_model_produces_identical_output_with_same_seed() { ... }
```

---

## 9. Documentation

### Doc Comments on All Public Items

Every `pub` function, struct, enum, trait, and module has a doc comment. No exceptions.

```rust
/// A rigid body link in the kinematic tree.
///
/// Each link has a unique [`LinkId`] and an associated pose in world coordinates.
/// Links are connected by [`Joint`]s and form a tree rooted at the base link.
///
/// # Examples
///
/// ```
/// use clankers_core::prelude::*;
///
/// let link = Link::new(LinkId::new(0).unwrap(), "base_link");
/// assert_eq!(link.name(), "base_link");
/// ```
#[derive(Debug, Clone)]
pub struct Link { ... }
```

### Module-Level Documentation

Each module file begins with `//!` doc comments that explain the module's purpose, its
role in the architecture, and a brief usage example.

```rust
//! Joint state management and kinematic computations.
//!
//! This module provides [`JointState`] for tracking joint positions, velocities,
//! and efforts, along with forward kinematics utilities.
//!
//! # Architecture
//!
//! Joint states are stored as Bevy components on joint entities. The physics
//! system writes to them each tick, and sensor systems read from them.
```

### Private Items

Do not write doc comments (`///`) on private items. Use regular comments (`//`) for
implementation notes on private functions and fields:

```rust
struct InternalCache {
    // Lazily computed inverse kinematics solution.
    ik_cache: Option<IkSolution>,
}

// Computes the Jacobian transpose for the given chain.
// Uses the recursive Newton-Euler formulation.
fn compute_jacobian_transpose(chain: &KinematicChain) -> Matrix {
    ...
}
```

### `#[must_use]` Annotation

Apply `#[must_use]` to:
- All `Result`-returning public functions (clippy covers this, but be explicit).
- Builder methods that return `Self`.
- Pure functions whose return value is the entire point of calling them.

```rust
impl SimConfigBuilder {
    #[must_use]
    pub fn dt_ns(mut self, dt_ns: u64) -> Self {
        self.dt_ns = dt_ns;
        self
    }
}
```

### Documentation Checklist

- `///` on every `pub` item.
- `//!` at the top of every module.
- `# Examples` section on types that have non-obvious construction.
- `# Errors` section on fallible functions listing when each error variant occurs.
- `# Panics` section if a function can panic (it should not, but document it if it does).
- No broken intra-doc links (`cargo doc --no-deps` must succeed without warnings).

---

## 10. CI/CD and Linting

### Workspace Lint Configuration

Lint configuration lives in `Cargo.toml` at the workspace level so every crate inherits
the same rules:

```toml
# Cargo.toml (workspace root)
[workspace.lints.rust]
unsafe_code = "forbid"

[workspace.lints.clippy]
all = { level = "deny", priority = -1 }
pedantic = { level = "warn", priority = -1 }
nursery = { level = "warn", priority = -1 }
module_name_repetitions = "allow"
must_use_candidate = "allow"
missing_errors_doc = "allow"
missing_panics_doc = "allow"
```

Each crate inherits these lints:

```toml
# crates/clankers-core/Cargo.toml
[lints]
workspace = true
```

### unsafe_code = "forbid"

There is no escape hatch. No `unsafe` block, no `unsafe fn`, no `unsafe impl`. If a
dependency requires unsafe, it must be behind a well-audited wrapper. The Bevy and
Rapier ecosystems handle low-level unsafe internally; Clankerss code stays safe.

### rustfmt

The workspace `rustfmt.toml` enforces consistent formatting:

```toml
max_width = 100
use_field_init_shorthand = true
use_try_shorthand = true
```

All code must pass `cargo fmt --check` before merging. No manual formatting overrides.

### cargo deny

Use `cargo deny` to audit dependencies for:
- Known vulnerabilities (advisories).
- Duplicate versions of the same crate.
- Disallowed licenses.

```toml
# deny.toml
[advisories]
vulnerability = "deny"

[licenses]
allow = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib"]

[bans]
multiple-versions = "warn"
```

### CI Pipeline Stages

The CI pipeline runs these checks in order. All must pass.

1. `cargo fmt --all --check` -- formatting.
2. `cargo clippy --workspace --all-targets -- -D warnings` -- linting.
3. `cargo test --workspace` -- all tests.
4. `cargo doc --workspace --no-deps` -- documentation builds without warnings.
5. `cargo deny check` -- dependency audit.

### MSRV Verification

CI includes a job that builds with the declared MSRV (1.88) to ensure no accidental use
of newer features:

```yaml
- name: Check MSRV
  run: |
    rustup install 1.88
    cargo +1.88 check --workspace
```

---

## 11. API Design

### Builder Pattern

Use the builder pattern for any struct with more than three configuration fields or
where defaults are meaningful. Builders consume `self` (not `&mut self`) to enable
method chaining and prevent reuse of a partially-configured builder.

```rust
pub struct SimConfig {
    dt_ns: u64,
    gravity: [f32; 3],
    seed: u64,
    max_steps: u64,
}

pub struct SimConfigBuilder {
    dt_ns: u64,
    gravity: [f32; 3],
    seed: u64,
    max_steps: u64,
}

impl SimConfigBuilder {
    pub fn new() -> Self {
        Self {
            dt_ns: 1_000_000, // 1ms default
            gravity: [0.0, -9.81, 0.0],
            seed: 0,
            max_steps: 10_000,
        }
    }

    #[must_use]
    pub fn dt_ns(mut self, dt_ns: u64) -> Self {
        self.dt_ns = dt_ns;
        self
    }

    #[must_use]
    pub fn gravity(mut self, gravity: [f32; 3]) -> Self {
        self.gravity = gravity;
        self
    }

    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    #[must_use]
    pub fn max_steps(mut self, max_steps: u64) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Validates and builds the configuration.
    ///
    /// # Errors
    ///
    /// Returns `ClankersError::Config` if:
    /// - `dt_ns` is zero.
    /// - `dt_ns` exceeds one second.
    /// - `max_steps` is zero.
    pub fn build(self) -> ClankersResult<SimConfig> {
        if self.dt_ns == 0 {
            return Err(ClankersError::Config("dt_ns must be > 0".into()));
        }
        if self.dt_ns > 1_000_000_000 {
            return Err(ClankersError::Config("dt_ns must be <= 1s".into()));
        }
        if self.max_steps == 0 {
            return Err(ClankersError::Config("max_steps must be > 0".into()));
        }
        Ok(SimConfig {
            dt_ns: self.dt_ns,
            gravity: self.gravity,
            seed: self.seed,
            max_steps: self.max_steps,
        })
    }
}
```

### Enum Over Trait Objects for Finite Variants

When the set of variants is known at compile time, use an enum. Enums are matchable,
serializable, `Copy`-able (if variants are `Copy`), and have no dynamic dispatch
overhead.

```rust
// Good: finite set of control modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlMode {
    Position,
    Velocity,
    Torque,
}

// Good: match is exhaustive, compiler catches missing arms
fn apply_command(mode: ControlMode, value: f32) -> f32 {
    match mode {
        ControlMode::Position => clamp_position(value),
        ControlMode::Velocity => clamp_velocity(value),
        ControlMode::Torque => clamp_torque(value),
    }
}
```

Reserve trait objects (`Box<dyn Trait>`) for genuinely open-ended extensibility points
like user-provided policy backends.

### Composition Over Inheritance (Chain Pattern)

Rust has no inheritance. Build complex behavior by composing smaller components. Use
the "chain" pattern where stages are explicit and ordered:

```rust
/// A processing pipeline built from composable stages.
pub struct ObservationPipeline {
    stages: Vec<Box<dyn ObservationStage>>,
}

impl ObservationPipeline {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    #[must_use]
    pub fn add_stage(mut self, stage: impl ObservationStage + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    pub fn process(&self, raw: &mut Observation) -> ClankersResult<()> {
        for stage in &self.stages {
            stage.apply(raw)?;
        }
        Ok(())
    }
}
```

### Borrowing Over Copying

Public API functions accept references to non-`Copy` types. This avoids forcing callers
to clone and clearly communicates that the function does not consume the value.

```rust
// Good: borrows the observation, caller retains ownership
pub fn compute_reward(obs: &Observation) -> f32 { ... }

// Bad: takes ownership unnecessarily, forces clone at call site
pub fn compute_reward(obs: Observation) -> f32 { ... }
```

Pass by value only for `Copy` types and when the function genuinely needs ownership
(e.g., to store the value in a struct).

### Fail-Fast Validation

Provide a `validate()` method on configuration types that checks all invariants. Call
it at construction time. Users who build configs programmatically get errors immediately
rather than deep inside a simulation run.

```rust
impl SimConfig {
    /// Validates all configuration invariants.
    ///
    /// Called automatically by [`SimConfigBuilder::build`].
    /// Call manually when deserializing from external sources.
    ///
    /// # Errors
    ///
    /// Returns `ClankersError::Config` describing the first invalid field.
    pub fn validate(&self) -> ClankersResult<()> {
        if self.dt_ns == 0 {
            return Err(ClankersError::Config("dt_ns must be > 0".into()));
        }
        if self.dt_ns > 1_000_000_000 {
            return Err(ClankersError::Config("dt_ns must be <= 1s".into()));
        }
        let mag_sq = self.gravity.iter().map(|g| g * g).sum::<f32>();
        if mag_sq < 1e-6 {
            return Err(ClankersError::Config("gravity magnitude near zero".into()));
        }
        Ok(())
    }
}
```

### Extension Traits

When adding methods to foreign types (Bevy's `Transform`, Rapier's `RigidBody`), use
extension traits in a clearly named module:

```rust
// crates/clankers-core/src/transform_ext.rs

/// Extension methods for Bevy's `Transform`.
pub trait TransformExt {
    /// Extracts the yaw angle (rotation around Y) in radians.
    fn yaw(&self) -> f32;
}

impl TransformExt for bevy::prelude::Transform {
    fn yaw(&self) -> f32 {
        let (_, yaw, _) = self.rotation.to_euler(bevy::math::EulerRot::XYZ);
        yaw
    }
}
```

---

## 12. Dependencies

### Minimize External Dependencies

Every dependency is an attack surface, a compilation cost, and a maintenance burden.
Before adding a crate, ask:
1. Can we implement this in under 100 lines?
2. Is this crate well-maintained (recent commits, low issue count)?
3. Does it pull in a large transitive dependency tree?

### Workspace Dependency Inheritance

All external dependencies are declared once in the workspace `Cargo.toml` under
`[workspace.dependencies]`. Crates reference them without version numbers.

```toml
# Workspace Cargo.toml
[workspace.dependencies]
bevy = "0.17.3"
serde = { version = "1", features = ["derive"] }
thiserror = "2"
rand = "0.8"
rand_chacha = "0.3"
```

```toml
# crates/clankers-core/Cargo.toml
[dependencies]
bevy = { workspace = true }
serde = { workspace = true }
thiserror = { workspace = true }
```

This ensures version consistency across the entire workspace and makes upgrades a
single-line change.

### Feature-Gate Heavy Optional Dependencies

Large or platform-specific dependencies hide behind feature flags. A user who does not
need CUDA should not compile CUDA bindings.

```toml
[features]
default = []
cuda = ["dep:cust"]
tensorrt = ["dep:tensorrt-rs", "cuda"]
render = ["bevy/default"]

[dependencies]
cust = { version = "0.3", optional = true }
tensorrt-rs = { version = "0.2", optional = true }
```

### Pin Major Versions Only

Specify dependencies with major version only (`"1"`, `"0.8"`). Let Cargo resolve
compatible minor and patch updates. Pin exact versions only when a specific version is
known to have a required fix or when reproducibility of CI builds is paramount (use
`Cargo.lock` for that).

```toml
# Good: allows compatible updates
serde = "1"
rand = "0.8"

# Acceptable for frameworks with breaking minor releases in 0.x:
bevy = "0.17.3"

# Bad: unnecessarily restrictive
serde = "=1.0.197"
```

### Cargo.lock

The `Cargo.lock` file is committed to version control. Even though Clankerss is a
library workspace, the lock file ensures CI builds are reproducible and developers get
identical dependency trees.

### Dev-Dependencies

Test-only crates go in `[dev-dependencies]`. They are not compiled into the library and
do not affect downstream users.

```toml
[dev-dependencies]
rstest = "0.25"
approx = "0.5"
clankers-test-utils = { workspace = true }
```

Never use `anyhow` in library code, but it is acceptable as a dev-dependency for test
convenience.

---

## Summary of Non-Negotiable Rules

These rules have no exceptions. Violating them blocks a merge.

1. `unsafe_code = "forbid"` -- no unsafe anywhere.
2. No `unwrap()` or `expect()` in library code -- use `Result`.
3. No `thread_rng()` or entropy-seeded RNG -- determinism is sacred.
4. No floating-point time accumulation -- integer nanoseconds only.
5. `Debug` derived on every public type.
6. Doc comments on every public item.
7. All public types are `Send + Sync`.
8. All tests pass, all clippy warnings resolved before merge.
9. `cargo fmt --check` passes with no diff.
10. One plugin per crate, registered in `ClankersPlugins`.
