//! User-journey reference (G5).
//!
//! CODE_QUALITY_REVIEW § "Gap 5: Documentation Is Spotty for User
//! Journeys". The repo has API docs and design notes but no canonical
//! end-to-end reference for the six major workflows the system supports.
//!
//! This module is the journey index. Each variant of [`UserJourney`]
//! carries doc-comment prose describing the workflow plus links to the
//! concrete entry points (CLI commands, library calls, examples). The
//! enum exists so the journey list is discoverable by `cargo doc` and
//! rust-analyzer — no `*.md` files needed.
//!
//! The journey docs intentionally live alongside the rest of the code:
//! when the implementation changes, the journey doc that references it
//! is one rust-analyzer click away, and any rename trips the
//! `#[deny(rustdoc::broken_intra_doc_links)]` lint when we eventually
//! add it.

// ---------------------------------------------------------------------------
// UserJourney
// ---------------------------------------------------------------------------

/// The six canonical user journeys the clankers stack supports.
///
/// Each variant's rustdoc is the journey reference — the level of
/// detail aims to match a short Stack Overflow answer: enough to point
/// the reader at the right APIs / commands without trying to substitute
/// for in-depth design docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UserJourney {
    /// **Author a new task.**
    ///
    /// Authoring a task = wiring up a scene, an observation contract,
    /// an action contract, a reward, and termination rules. The
    /// canonical shape is [`crate::env_spec::EnvSpec`] — every field
    /// either references registered IDs or carries a small inline
    /// struct.
    ///
    /// **Steps:**
    ///
    /// 1. Pick an `EnvId` (`namespace/name`). Use the `"user"`
    ///    namespace for downstream-owned tasks; `"clankers"` is
    ///    reserved for in-tree tasks shipped with the workspace.
    /// 2. Build a [`crate::env_spec::SceneSpec`] referencing a URDF
    ///    file or a registered scene factory id. Object lists declare
    ///    static / dynamic bodies the task expects.
    /// 3. Declare the `ActionContract` + `ObservationContract` with
    ///    the appropriate [`crate::types::ActionSpace`] /
    ///    [`crate::types::ObservationSpace`] variant.
    /// 4. Pick a reset distribution from
    ///    [`crate::env_spec::ResetDistribution`]; for new envs default
    ///    to `Fixed` and add randomisation later.
    /// 5. Pick a [`crate::env_spec::RewardProvider`] — `Rust` for
    ///    in-tree, deterministic rewards; `Python` for ergonomic
    ///    user-side rewards (lower throughput).
    /// 6. Validate end-to-end with
    ///    [`crate::validators::validate_observation_against_space`]
    ///    inside a smoke test that takes one step and checks the
    ///    observation matches the declared shape.
    ///
    /// **Entry point:** `clankers_core::env_spec`.
    /// **Sample task:** `crates/clankers-sim/src/scenarios/cartpole.rs`.
    AuthorTask,

    /// **Train a policy against an existing task.**
    ///
    /// Training currently runs Python-side via `clankers` package's
    /// Gymnasium-compatible env wrapper backed by the Rust gym server.
    ///
    /// **Steps:**
    ///
    /// 1. Pick a task id and confirm the env loads via
    ///    `python -m clankers.cli list-envs` (G3 follow-up).
    /// 2. Decide on parallelism — `GymVecEnv::new_sequential` for
    ///    debugging, `new_parallel` for throughput. The Python wrapper
    ///    picks one based on the `--num-envs` flag.
    /// 3. Pin the master seed in the run manifest. The clankers stack
    ///    derives per-env seeds via `SeedHierarchy` so reruns with the
    ///    same `master_seed` reproduce.
    /// 4. Train via your library of choice (`stable-baselines3`,
    ///    `rl-games`, custom PyTorch loop). Wrap with
    ///    `clankers.evaluation.evaluate_policy` for the final scoring
    ///    pass.
    /// 5. Stamp a [`crate::manifest::RunManifest`] sidecar next to the
    ///    saved policy weights. Reviewers / future-you need it.
    ///
    /// **Entry point:** Python `clankers.gymnasium_env.GymnasiumEnv`.
    /// **Sample script:** `python/examples/train_cartpole.py` (when
    /// landed — currently the script exists per-task).
    TrainPolicy,

    /// **Record a trace / dataset.**
    ///
    /// Traces are MCAP-formatted with a JSON sidecar
    /// ([`crate::manifest::RunManifest`]). Production use case is
    /// either (a) building synthetic-data datasets for imitation /
    /// world-model training, or (b) capturing failure cases for
    /// regression tests.
    ///
    /// **Steps:**
    ///
    /// 1. Decide on the recorder pipeline: in-process synchronous via
    ///    `clankers-record`, or out-of-process via the gym server's
    ///    `--record-out` flag (which uses the async writer with a
    ///    bounded queue + drop counter).
    /// 2. Pin a `frame_rate` and `kind_filter` to keep the MCAP small.
    ///    Camera frames are heavy — only record them when downstream
    ///    use requires it.
    /// 3. Monitor `RecorderHealth.error_count` mid-run; non-zero
    ///    means the worker thread observed a write failure and the
    ///    trace is incomplete.
    /// 4. Validate the trace post-hoc with
    ///    [`crate::validators::validate_trace_step`] applied to each
    ///    step decoded from the MCAP. CI runs this on every committed
    ///    sample trace.
    ///
    /// **Entry point:** `clankers_record::async_writer::AsyncRecorder`.
    /// **Health resource:** `clankers_record::async_writer::RecorderHealth`.
    RecordTrace,

    /// **Export an ONNX policy.**
    ///
    /// ONNX export is the bridge for moving a trained Python policy
    /// into the in-process Rust evaluation path or onto a deployed
    /// robot.
    ///
    /// **Steps:**
    ///
    /// 1. From the Python training side, call your library's ONNX
    ///    exporter (e.g. `torch.onnx.export`) on the policy network
    ///    only — value heads, exploration noise, and observation
    ///    preprocessing belong in the env wrapper, not the exported
    ///    graph.
    /// 2. Pin a stable input name (`"obs"`) and output name
    ///    (`"action"`) so the in-process consumer doesn't need
    ///    bespoke per-policy plumbing.
    /// 3. Load via `clankers_policy::onnx::OnnxPolicy::load`, attach
    ///    to a `GymEnv` via `OnnxPolicy::as_applicator`.
    /// 4. Stamp the ONNX file's SHA-256 into the consuming
    ///    [`crate::manifest::RunManifest`] so post-hoc analyses know
    ///    which policy file produced which trace.
    ///
    /// **Entry point:** `clankers_policy::onnx`.
    ExportOnnx,

    /// **Speak the wire protocol from a non-Python client.**
    ///
    /// The gym server speaks length-prefixed JSON + binary frames over
    /// TCP. Any language can connect; the Python client is the
    /// reference implementation.
    ///
    /// **Steps:**
    ///
    /// 1. Match the protocol version negotiated in the `Hello` frame.
    ///    Current Python client floor is `1.2.0`; the server accepts
    ///    `>= 1.0.0` for backward compatibility but feature-gates
    ///    capabilities like binary batches via the negotiated
    ///    [`Capabilities`](https://docs.rs/clankers-gym) struct.
    /// 2. Frame layout: 4-byte LE length + JSON envelope. Binary
    ///    follow-ups (observation buffers, image batches) carry a
    ///    [`tensor_frame::TensorFrameHeader`](https://docs.rs/clankers-gym)
    ///    so any client decodes them without bespoke wire knowledge.
    /// 3. Honour the per-connection read / write timeouts the server
    ///    applies — long-running clients should send periodic
    ///    keep-alive `ping` frames.
    /// 4. Validate every received action / observation against the
    ///    declared space using the contracts above before applying it
    ///    to your local env / policy.
    ///
    /// **Entry point:** Rust server `clankers_gym::server::GymServer`,
    /// Python client `clankers.client.GymClient`.
    SpeakProtocol,

    /// **Understand the workspace architecture.**
    ///
    /// The repo is split into focused crates under
    /// `crates/`, with `apps/clankers-app` as the canonical binary
    /// entry point and `examples/` as the runnable showcase. The
    /// dependency graph is intentionally shallow: `clankers-core` →
    /// `clankers-physics` / `clankers-actuator` / `clankers-urdf` →
    /// `clankers-env` / `clankers-gym` / `clankers-record` →
    /// `clankers-sim` (scenarios) → `clankers-app` (CLI) /
    /// `examples/`.
    ///
    /// **Layer roles:**
    ///
    /// - `clankers-core` — types, traits, contracts. No Bevy systems,
    ///   no Rapier types. Everything else depends on this.
    /// - `clankers-physics` — Rapier-specific implementation behind
    ///   the engine-neutral [`crate::env_spec`] / readback /
    ///   buffer APIs.
    /// - `clankers-gym` — protocol + TCP server + vec-env runners.
    /// - `clankers-record` — MCAP writers + sync / async pipelines.
    /// - `clankers-sim` — scenarios + scene-build glue.
    /// - `clankers-app` — `clankers` CLI binary (bench / record /
    ///   inspect / validate / compare).
    ///
    /// **Why this shape:** each crate corresponds to one user
    /// journey above; crossing the boundary means converting
    /// between typed contract surfaces so silent drift is caught at
    /// the boundary.
    UnderstandArchitecture,
}

impl UserJourney {
    /// All journeys, in canonical doc order.
    pub const ALL: [UserJourney; 6] = [
        Self::AuthorTask,
        Self::TrainPolicy,
        Self::RecordTrace,
        Self::ExportOnnx,
        Self::SpeakProtocol,
        Self::UnderstandArchitecture,
    ];

    /// Short slug usable as a CLI flag or doc anchor.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::AuthorTask => "author-task",
            Self::TrainPolicy => "train-policy",
            Self::RecordTrace => "record-trace",
            Self::ExportOnnx => "export-onnx",
            Self::SpeakProtocol => "speak-protocol",
            Self::UnderstandArchitecture => "architecture",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn slugs_are_unique() {
        let slugs: HashSet<&str> = UserJourney::ALL.iter().map(|j| j.slug()).collect();
        assert_eq!(slugs.len(), UserJourney::ALL.len());
    }

    #[test]
    fn all_variants_covered() {
        assert_eq!(UserJourney::ALL.len(), 6);
    }
}
