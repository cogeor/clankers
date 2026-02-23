# Clankerss Implementation Plan

## 1. Overview

Clankerss is a modular robotics simulation framework written in Rust, built on Bevy 0.17 and bevy_rapier3d 0.32. It replaces the role that Isaac Sim/Gym fills for NVIDIA hardware, running on commodity hardware with no vendor lock-in and shipping as a single binary. A user describes a robot and a task, Clankerss simulates the physics, and a Python training script on the other end of a TCP socket learns a policy. Once trained, the policy is exported to ONNX and runs natively in Rust.

The workspace contains 14 crates (13 library crates + 1 application binary) organized under `crates/` and `apps/`.

### Implementation Approach

Each module is implemented one at a time, following the dependency graph bottom-up. A module is considered complete only when:

1. All public types, traits, and functions are implemented.
2. All unit tests pass.
3. All integration tests pass.
4. `cargo clippy` (pedantic + nursery) reports zero warnings.
5. `cargo fmt --check` passes.
6. No `unsafe` code exists (enforced by `#[forbid(unsafe_code)]` at workspace level).

No module in a later phase begins until every module in the current phase meets these criteria.

---

## 2. Crate Dependency Graph

The workspace has a clear layered dependency structure. Leaf crates have zero internal dependencies. Higher-level crates compose them.

```
clankers-sim (top-level plugin)
 |
 +-- clankers-core ................. (bevy, serde, thiserror, rand, rand_chacha, toml)
 +-- clankers-noise ................ (rand, rand_distr)
 +-- clankers-env .................. (core, noise, bevy, bevy_rapier3d)
 +-- clankers-actuator-core ........ (no deps -- pure math)
 +-- clankers-actuator ............. (actuator-core, bevy, bevy_rapier3d)
 +-- clankers-urdf ................. (core, urdf-rs, bevy_urdf)
 +-- clankers-gym .................. (core, env, noise)
 +-- clankers-domain-rand .......... (core, noise, bevy_rapier3d)
 +-- clankers-policy ............... (core, ort)
 +-- clankers-render ............... (core, bevy)
 +-- clankers-teleop ............... (core, bevy)
 |
clankers-app (binary -- depends on clankers-sim)
clankers-test-utils (dev-dependency for shared fixtures)
```

### Dependency Matrix

| Crate              | core | noise | env | act-core | actuator | urdf | gym | domain-rand | policy | render | teleop |
|--------------------|:----:|:-----:|:---:|:--------:|:--------:|:----:|:---:|:-----------:|:------:|:------:|:------:|
| clankers-core       | --   |       |     |          |          |      |     |             |        |        |        |
| clankers-noise      |      | --    |     |          |          |      |     |             |        |        |        |
| clankers-env        | X    | X     | --  |          |          |      |     |             |        |        |        |
| clankers-actuator-core |   |       |     | --       |          |      |     |             |        |        |        |
| clankers-actuator   |      |       |     | X        | --       |      |     |             |        |        |        |
| clankers-urdf       | X    |       |     |          |          | --   |     |             |        |        |        |
| clankers-gym        | X    | X     | X   |          |          |      | --  |             |        |        |        |
| clankers-domain-rand| X    | X     |     |          |          |      |     | --          |        |        |        |
| clankers-policy     | X    |       |     |          |          |      |     |             | --     |        |        |
| clankers-render     | X    |       |     |          |          |      |     |             |        | --     |        |
| clankers-teleop     | X    |       |     |          |          |      |     |             |        |        | --     |
| clankers-sim        | X    | X     | X   | X        | X        | X    | X   | X           | X      | X      | X      |

---

## 3. Implementation Phases

### Phase 1: Foundation

**Crates:** `clankers-core`, `clankers-noise`

**Rationale:** These two crates have zero internal dependencies. `clankers-core` defines the vocabulary that every other crate imports: types (`Action`, `Observation`, `StepResult`), traits (`RLEnvironment`, `Sensor`), configuration (`SimConfig`, `SimTime`), system ordering (`ClankersSet`), and error types. `clankers-noise` provides the `NoiseModel` enum used by sensors, domain randomization, and action perturbation. Everything else in the workspace depends on one or both of these.

| Crate | Internal Deps | Key Deliverables |
|-------|---------------|------------------|
| clankers-core | none | `Action`, `Observation`, `StepResult`, `SimConfig`, `SimTime`, `ClankersSet`, `RobotHandle`, `ObjectHandle`, error types, `ClankersCorePlugin` |
| clankers-noise | none | `NoiseModel` enum (Gaussian, Uniform, Bias, Drift, Chain), `apply()`, deterministic seeding |

### Phase 2: Environment and Actuation

**Crates:** `clankers-env`, `clankers-actuator-core`, `clankers-actuator`

**Rationale:** With core types and noise models in place, the environment crate can implement state management, sensor collection, and episode lifecycle. The actuator crates are split by design: `actuator-core` contains pure motor math (testable without Bevy), and `actuator` wraps it in a Bevy plugin. These three crates together define how the simulation reads state and applies actions.

| Crate | Internal Deps | Key Deliverables |
|-------|---------------|------------------|
| clankers-env | core, noise | State management, sensor systems (joint, IMU, camera, contact), `ObservationBuffer`, episode lifecycle |
| clankers-actuator-core | none | `MotorModel` (DC, ideal), gear ratio, transmission, friction (Coulomb, viscous), PID controller |
| clankers-actuator | actuator-core | `ActuatorPlugin`, `apply_actions` system, joint state sync |

### Phase 3: Robot Loading

**Crates:** `clankers-urdf`

**Rationale:** Before training infrastructure can be built, the system needs to load robot descriptions. URDF parsing depends on `clankers-core` types for `RobotHandle` and joint/link representations. This is a standalone phase because URDF loading is complex (mesh loading, joint configuration, collision geometry) and benefits from focused attention.

| Crate | Internal Deps | Key Deliverables |
|-------|---------------|------------------|
| clankers-urdf | core | URDF parsing (via `urdf-rs`), mesh loading, robot entity spawning, joint configuration, `RobotModel` |

### Phase 4: Training Infrastructure

**Crates:** `clankers-gym`, `clankers-domain-rand`

**Rationale:** With environment, actuation, and robot loading complete, the training pipeline can be assembled. `clankers-gym` implements the TCP server and Gymnasium-compatible protocol that Python training scripts connect to. `clankers-domain-rand` implements physics parameter randomization for sim-to-real transfer. Both depend on core and noise; gym additionally depends on env.

| Crate | Internal Deps | Key Deliverables |
|-------|---------------|------------------|
| clankers-gym | core, env, noise | TCP server, length-prefixed JSON protocol (v1.0.0), `reset`/`step` handlers, reward/termination wiring, VecEnv batch interface |
| clankers-domain-rand | core, noise | Mass/friction randomization, external force perturbation, per-environment randomization, `DomainRandPlugin` |

### Phase 5: Inference and Visualization

**Crates:** `clankers-policy`, `clankers-render`, `clankers-teleop`

**Rationale:** These crates enable deployment and debugging. They depend only on `clankers-core` (and Bevy for render/teleop), so they could theoretically be built earlier, but they are not needed until the training pipeline is functional. Building them in Phase 5 means they can be tested against a working simulation.

| Crate | Internal Deps | Key Deliverables |
|-------|---------------|------------------|
| clankers-policy | core | ONNX model loading (via `ort`), `policy_inference_system`, action generation from observations, feature flags for `cpu`/`cuda`/`tensorrt` |
| clankers-render | core | Headless rendering, GPU-to-CPU image transfer, camera observation pipeline, resolution configuration |
| clankers-teleop | core | Manual control trait, keyboard/gamepad input mapping, debug visualization |

### Phase 6: Integration

**Crates:** `clankers-sim`, `clankers-app`

**Rationale:** The top-level plugin wires every crate together into a single `ClankersPlugin` that users add to their Bevy app. It depends on all other crates and cannot be built until they are complete. `clankers-app` is the example binary that demonstrates the full stack.

| Crate | Internal Deps | Key Deliverables |
|-------|---------------|------------------|
| clankers-sim | all crates | `ClankersPlugin` (training/inference/visualization modes), system ordering enforcement, plugin composition, scene loading |
| clankers-app | clankers-sim | Example application, CLI interface (`--mode`, `--scene`, `--port`, `--policy`), scene TOML loading |

### Phase Diagram

```
Phase 1          Phase 2              Phase 3       Phase 4             Phase 5              Phase 6
----------       ---------------      ---------     ---------------     ----------------     -----------
core ------+---> env -----------+---> urdf ----+--> gym -----------+--> policy --------+---> sim
           |                    |              |                   |                    |
noise -----+---> actuator-core -+              +--> domain-rand ---+--> render ---------+---> app
                 actuator ------+                                      teleop ----------+
```

---

## 4. Per-Crate Summary Table

| Crate | Phase | Key Types / Traits | External Dependencies | Complexity |
|-------|:-----:|--------------------|-----------------------|:----------:|
| clankers-core | 1 | `Action`, `Observation`, `StepResult`, `SimConfig`, `SimTime`, `ClankersSet`, `RobotHandle`, `ObjectHandle`, `RLEnvironment` trait, error types | bevy (app + ecs), serde, thiserror, rand, rand_chacha, toml | M |
| clankers-noise | 1 | `NoiseModel` (Gaussian, Uniform, Bias, Drift, Chain), `apply()` | rand, rand_distr | S |
| clankers-env | 2 | `ObservationBuffer`, `SensorPlugin`, joint/IMU/camera/contact sensors, episode state | bevy, bevy_rapier3d, clankers-core, clankers-noise | L |
| clankers-actuator-core | 2 | `MotorModel`, `GearRatio`, `Transmission`, `FrictionModel`, PID | (none or minimal) | S |
| clankers-actuator | 2 | `ActuatorPlugin`, `apply_actions`, `joint_state_sync_system` | bevy, bevy_rapier3d, clankers-actuator-core | M |
| clankers-urdf | 3 | `RobotModel`, URDF parser adapter, mesh loader, entity spawner | urdf-rs, bevy_urdf, clankers-core | M |
| clankers-gym | 4 | TCP server, protocol messages, `GymPlugin`, batch step/reset, reward/termination wiring | serde_json, clankers-core, clankers-env, clankers-noise | L |
| clankers-domain-rand | 4 | `DomainRandPlugin`, `RandomizationConfig`, mass/friction/force randomizers | bevy_rapier3d, clankers-core, clankers-noise | M |
| clankers-policy | 5 | `PolicyPlugin`, `OnnxPolicy`, `policy_inference_system` | ort, clankers-core | M |
| clankers-render | 5 | `RenderPlugin`, headless camera, GPU-to-CPU pipeline | bevy (render), clankers-core | M |
| clankers-teleop | 5 | `TeleopPlugin`, `ManualController` trait, input mapping | bevy (input), clankers-core | S |
| clankers-sim | 6 | `ClankersPlugin`, mode constructors, plugin composition | all internal crates | L |
| clankers-app | 6 | CLI args, scene loading, example setup | clankers-sim, clap | S |
| clankers-test-utils | -- | Robot fixtures, sensor mocks, test harnesses | various | S |

**Complexity key:** S = small (< 500 lines, straightforward), M = medium (500-2000 lines, moderate design decisions), L = large (2000+ lines, significant design surface)

---

## 5. Testing Strategy

### Unit Tests

Every module contains in-module unit tests using `#[cfg(test)]`. Tests live alongside the code they test. Pure-math crates (`clankers-noise`, `clankers-actuator-core`) have the highest unit test density because they are easy to test in isolation.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_noise_has_correct_mean() {
        // ...
    }
}
```

### Integration Tests

Each crate has a `tests/` directory for integration tests that exercise the public API. Crates with Bevy plugins (`clankers-env`, `clankers-actuator`, `clankers-sim`) use minimal Bevy app setups to test system behavior.

```
crates/clankers-core/tests/
    config_roundtrip.rs
    action_validation.rs
crates/clankers-noise/tests/
    noise_determinism.rs
    chain_composition.rs
```

### Test Framework

- **rstest** for `#[fixture]` and `#[case]` parameterized tests.
- **cargo-tarpaulin** for coverage reports.
- **clankers-test-utils** crate provides shared fixtures:
  - Pre-built `RobotModel` instances (UR5, simple 2-link arm).
  - Sensor mock data generators.
  - Minimal Bevy `App` builders for plugin testing.
  - Deterministic RNG seeds for reproducibility.

### Workspace-Level End-to-End Tests

Located in `tests/` at the workspace root (or in the `clankers-app` crate). These tests exercise the full pipeline:

1. Load a scene from TOML.
2. Spawn a robot from URDF.
3. Run N physics steps with a fixed action sequence.
4. Verify observations match expected values (deterministic).
5. Verify reward computation.
6. Verify episode termination/truncation.

### Determinism Verification

Every test that involves RNG must use a fixed seed and assert exact floating-point outputs. The seed hierarchy is: workspace seed -> environment seed -> sensor seed -> noise seed. See `DETERMINISM.md` for the full contract.

---

## 6. CI/CD Strategy

### GitHub Actions Workflow

A single workflow file (`.github/workflows/ci.yml`) runs on every push and pull request targeting `main`.

#### Jobs

| Job | Command | Purpose |
|-----|---------|---------|
| **fmt** | `cargo fmt --all -- --check` | Enforce consistent formatting |
| **clippy** | `cargo clippy --workspace --all-targets -- -D warnings` | Catch lint violations (pedantic + nursery enabled in workspace) |
| **test** | `cargo test --workspace` | Run all unit and integration tests |
| **deny** | `cargo deny check` | Audit dependencies for licenses, advisories, and bans |
| **doc** | `cargo doc --workspace --no-deps` | Verify documentation builds without warnings |
| **coverage** | `cargo tarpaulin --workspace --out xml` | Generate coverage report |

#### Matrix

- **OS:** Ubuntu latest (primary), Windows latest (secondary)
- **Rust:** stable, MSRV (1.88)

#### Caching

- Cache `~/.cargo/registry`, `~/.cargo/git`, and `target/` keyed on `Cargo.lock` hash.
- Use `actions/cache` with a fallback to the previous week's cache.

### Dependency Auditing

`cargo-deny` configuration (`deny.toml`) at workspace root:

- **Licenses:** Allow MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Zlib. Deny copyleft.
- **Advisories:** Deny any crate with known security advisories.
- **Bans:** No duplicate versions of critical crates (bevy, rapier3d).

### Release Process

1. Version bump in workspace `Cargo.toml`.
2. Tag with `v{version}`.
3. CI builds release binaries for Linux, Windows, macOS.
4. Publish library crates to crates.io in dependency order.

---

## 7. Quality Gates

Each phase must satisfy every gate before the next phase begins. There are no exceptions.

### Per-Phase Gates

| Gate | Criterion | Verification |
|------|-----------|--------------|
| **Build** | `cargo build --workspace` succeeds with zero errors | CI |
| **Lint** | `cargo clippy --workspace --all-targets -- -D warnings` reports zero warnings | CI |
| **Format** | `cargo fmt --all -- --check` passes | CI |
| **Tests** | `cargo test --workspace` -- all tests pass | CI |
| **Safety** | No `unsafe` code anywhere in the workspace | `#[forbid(unsafe_code)]` in workspace lints |
| **Docs** | `cargo doc --workspace --no-deps` builds without warnings | CI |
| **Dependencies** | `cargo deny check` passes | CI |

### Phase Transition Checklist

Before starting Phase N+1:

- [ ] All crates in Phase N compile.
- [ ] All clippy warnings in Phase N are resolved.
- [ ] All tests in Phase N pass.
- [ ] Public API documentation exists for all exported items.
- [ ] Integration tests cover the primary use cases.
- [ ] No TODO or FIXME comments remain in Phase N code (tracked issues are acceptable).

### Code Review Standards

- Every PR requires at least one review.
- PRs must pass all CI checks before merge.
- No force-pushes to `main`.
- Squash-merge to keep history clean.

---

## Appendix A: External Dependency Inventory

| Dependency | Version | Used By | Purpose |
|------------|---------|---------|---------|
| bevy | 0.17.3 | core, env, actuator, urdf, render, teleop, sim | ECS runtime, rendering, asset loading |
| bevy_rapier3d | 0.32 | env, actuator, domain-rand, sim | Physics engine integration |
| serde | 1.x | core, gym | Serialization/deserialization |
| serde_json | 1.x | gym | JSON protocol encoding |
| thiserror | 2.x | core | Error type derivation |
| rand | 0.8 | core, noise | Random number generation |
| rand_chacha | 0.3 | core | Deterministic RNG |
| rand_distr | 0.4 | noise | Statistical distributions |
| toml | 0.8 | core | Configuration file parsing |
| urdf-rs | latest | urdf | URDF file parsing |
| bevy_urdf | latest | urdf | URDF-to-Bevy entity conversion |
| ort | latest | policy | ONNX Runtime bindings |
| clap | latest | app | CLI argument parsing |
| rstest | latest | test-utils, all (dev) | Test fixtures and parameterization |

## Appendix B: File Layout Convention

```
crates/clankers-{name}/
    Cargo.toml
    src/
        lib.rs          # Public API re-exports, module declarations
        types.rs        # Data structures
        traits.rs       # Trait definitions (if applicable)
        plugin.rs       # Bevy plugin (if applicable)
        systems.rs      # Bevy systems (if applicable)
        error.rs        # Error types (if applicable)
    tests/
        integration_*.rs
```

Each `lib.rs` exports a `prelude` module containing the most commonly used items.

## Appendix C: Workspace Lint Configuration

The workspace enforces strict linting via `Cargo.toml`:

```toml
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

This configuration denies all default clippy lints, warns on pedantic and nursery lints, and allows a small set of false-positive-prone rules. The `unsafe_code = "forbid"` setting prevents any use of `unsafe` across the entire workspace.
