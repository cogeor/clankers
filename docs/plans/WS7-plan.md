# Workstream 7 — Performance Architecture (Rayon VecEnv, Binary Protocol, Dense Runtimes, Async Recorder)

Status: planned. Estimated implementation: 4 PRs.
Source report: `notes/clankers_codebase_quality_report_2026-05-25.md`,
"Performance Findings" subsections #1 (VecEnv Is Sequential),
#2 (Observation Collection Allocates), #3 (Protocol Uses JSON For Hot
Training Path), #5 (Hot-Path HashMaps), #6 (Recorder Writes Every Frame
Synchronously).
Depends on: W1 (`JointLayout`, `ObservationSchema`, `ActionSchema`,
`RecorderSchema`), W2 (layout-bound `MotorOverrides` `Vec` ordering),
W3 (`ObservationView<'a>`, fallible `Action::as_continuous`),
W4 (`EncodedObservation` enum and 4-byte LE framing baseline).
Unblocks: W5 PR4 (`bench` subcommand tree).

## 1. Goal

Turn the sequential, JSON-only Clankers runtime into one with measurable
parallel throughput by adding a Rayon-backed `ParallelVecEnvRunner`,
extending W4's `EncodedObservation` enum with binary batch frames for the
training hot path, compiling setup-time `HashMap` lookups into dense
`Vec<…Runtime>` indexed by layout position, wrapping the synchronous MCAP
recorder in a bounded-channel async writer with dropped-frame metrics, and
gating all of the above behind committed CSV baselines that fail CI on
>15% regression.

## 2. Why this workstream, why this order

W7 sits last among the **core technical** workstreams (before only the
test/CI hardening of W8) because every PR consumes contracts that the
earlier workstreams put in place:

- **Consumes W1** — the binary batch frame format embeds an
  `ObservationSchema` version and an `ActionSchema` dim; the dense
  runtime layout is keyed by `JointLayout` slot index. Without W1 the
  binary frame would have no negotiated shape to encode and the dense
  vectors would have no canonical order.
- **Consumes W2** — `MotorOverrides` is already
  `Vec<(JointHandle, MotorOverrideParams)>` in layout order after W2's
  PR1 (see `MEMORY.md`: "every joint must have an override"). W7 PR3
  generalises that pattern to `BodyHandles` and `JointHandles` and
  rebinds `crates/clankers-physics/src/rapier/systems.rs` hot loops to
  index by layout slot rather than `Entity` hash lookup. Doing this
  before W2 would have created a second source of truth for joint
  ordering.
- **Consumes W3** — the binary frame avoids the per-env `Observation`
  clone in `crates/clankers-gym/src/vec_env.rs:140-145` by writing
  directly from `VecObsBuffer::row(env_idx) -> ObservationView<'_>`
  (added in W3 PR1). Without W3's zero-copy view the encoder would
  re-introduce the very allocation the binary frame is trying to
  eliminate.
- **Consumes W4** — `EncodedObservation` (the `FlatF32` / `RawU8Image` /
  `Dict` enum) was promoted to a transport-level type in W4 PR1. W7 PR2
  extends it with `BatchF32 { num_envs, dim, payload }` and
  `BatchRawU8Image { num_envs, w, h, c, layout, payload }`. The JSON
  control plane (Init / Reset / Step / Close framing) stays exactly as
  W4 left it; only the obs/action payload slots flip to binary.
- **Unblocks W5 PR4** — the `clankers bench` command tree in W5's PR4
  was deliberately left as scaffolding (subcommand stubs returning
  `unimplemented!()`); W7 PR4 fills the four benchmark bodies
  (`bench vec`, `bench protocol`, `bench record`, `bench mpc`) and
  commits the baseline CSV those subcommands compare against.

Doing W7 earlier would force every later workstream to chase a moving
target: `ParallelVecEnvRunner` cannot be deterministic without
`JointLayout`-ordered observations; the binary frame cannot serialise
without a typed `ObservationSchema`; the dense runtime cannot be indexed
without W2's slot order; and the async recorder cannot share a single
`RecorderSchema` version with the loader without W6 finalising it. W7 is
also the **only** workstream that touches the cargo benchmark harness and
CI regression gate, so consolidating those changes into a single
workstream avoids three earlier PRs each modifying `benches/` and the CI
yaml independently.

CLAUDE.md mandates `cargo test -j 24` / `cargo build -j 24` (machine has
32 cores; 8 cores of headroom must remain free). Every cargo command in
this plan — including `cargo bench` — explicitly passes `-j 24`. Rayon's
default thread pool also caps at logical-core count, so PR4 sets
`RAYON_NUM_THREADS=24` in the bench harness to keep benchmarks
reproducible regardless of host CPU count.

## 3. Out of scope

The following look related but are deliberately deferred from W7:

- **MPC solver warm-start.** The quality report's MPC subsection notes
  Clarabel is rebuilt per solve. Investigating warm-start / workspace
  reuse is a separate workstream (filed in
  `notes/clankers_upstream_contributions_report_2026-05-25.md`). W7 only
  adds `bench mpc` so the eventual warm-start can be regression-tested.
- **Python-side binary client rewrite.** W7 ships the Rust server side of
  the binary frame plus a Python decoder stub in `python/clankers/
  client.py` that consumes the new frame variants. A full Python
  `VecEnvClient` rewrite to skip `.tolist()` / `json.dumps()` is left to
  a follow-up (referenced under "Python data pipelines copy heavily",
  Performance Findings #8).
- **Recorder file-format change.** The async recorder wraps the same
  `mcap::Writer` from W6 in a bounded channel. The on-disk format, topic
  constants, and `RecorderSchema` version remain exactly as W6 finalised
  them.
- **GPU readback / render double-buffering.** Performance Findings #7
  (render readback latency) is out of scope; rendering pipeline changes
  belong to a separate viz workstream.
- **Exclusive observation system removal.** Performance Findings #4
  (`observe_system(world: &mut World)` is exclusive and prevents Bevy
  parallel scheduling) is a separate architectural change; W7 keeps the
  exclusive observe path and parallelises across envs above it.
- **VecEnv dynamic env shape changes.** `ParallelVecEnvRunner` assumes
  every env shares the same `ObservationSchema` and `ActionSchema` —
  heterogeneous env batches are not supported and not in scope.
- **Sequential `VecEnvRunner` deletion.** `VecEnvRunner` at
  `crates/clankers-gym/src/vec_env.rs:11` stays for back-compat. A
  runtime config / feature flag selects parallel; defaulting to
  parallel is a follow-up after the determinism evidence has shipped.

## 4. Files to change

Line references are from `master` at the time of writing this plan and
match the evidence pre-gathered in
`.delegate/work/20260525-185618-workstream-plans/07/PLAN.md`.

### NEW

| Path | Purpose |
|---|---|
| `crates/clankers-env/src/parallel_runner.rs` | `ParallelVecEnvRunner` (Rayon `par_iter_mut` over `Vec<Box<dyn VecEnvInstance + Send>>`, deterministic per-env seed by index, ordered result collection). |
| `crates/clankers-gym/src/binary_frame.rs` | Little-endian binary frame writer/reader: `BatchF32`, `BatchRawU8Image` payload layout, `BinaryFrameHeader { version: u32, kind: u8, num_envs: u32, dim: u32, … }`. |
| `crates/clankers-record/src/async_writer.rs` | `AsyncRecorder` wrapping `mcap::Writer` behind a bounded `crossbeam_channel::bounded(capacity)` channel; writer thread + dropped-frame counter resource. |
| `crates/clankers-physics/src/rapier/runtime.rs` | Dense `BodyRuntime { entity: Entity, handle: RigidBodyHandle, layout_slot: usize }`, `JointRuntime { entity: Entity, handle: ImpulseJointHandle, layout_slot: usize, motor: MotorOverrideParams }`, and `RobotGroup::compile_runtime(&layout) -> Vec<JointRuntime>` builder. |
| `benches/baseline.csv` | Committed CSV with columns `bench,num_envs,steps_per_sec,p50_us,p95_us,p99_us` for `vec`, `protocol`, `record`, `mpc` at fixed seeds. Regenerated by `cargo bench --bench vec -j 24`. |
| `crates/clankers-env/tests/parallel_determinism.rs` | Integration test `parallel_vec_env_seed_assignment_is_deterministic`. |
| `crates/clankers-gym/tests/binary_protocol_roundtrip.rs` | Integration tests `batch_f32_roundtrip_byte_equal` and `batch_raw_u8_roundtrip_byte_equal`. |
| `crates/clankers-record/tests/async_backpressure.rs` | Integration test `async_recorder_drops_frames_under_backpressure`. |
| `crates/clankers-physics/tests/dense_runtime.rs` | Integration test `dense_runtime_matches_hashmap_lookup`. |
| `scripts/compare_baseline.py` | CSV comparator (current vs `benches/baseline.csv`), exits non-zero on `>tolerance` regression. |
| `benches/vec.rs` | Criterion bench crate target wrapping the same `bench vec` body so `cargo bench --bench vec` and `clankers bench vec` produce the same numbers. |

### MODIFY

| Path:line | Change |
|---|---|
| `crates/clankers-gym/src/vec_env.rs:11` | Re-export `ParallelVecEnvRunner` alongside `VecEnvRunner`. |
| `crates/clankers-gym/src/vec_env.rs:54` | Change `runner: VecEnvRunner` to `runner: Box<dyn VecRunnerLike>` (PR1 introduces the trait; sequential and parallel both impl it). |
| `crates/clankers-gym/src/vec_env.rs:67` | Add `+ Send` bound on the `dyn VecEnvInstance` trait object. Document the bound (no globally-shared Bevy resources). |
| `crates/clankers-gym/src/vec_env.rs:72` | `VecEnvRunner::new(envs, config)` becomes `runner_for(envs, config)` factory that dispatches on `config.parallel`. |
| `crates/clankers-gym/src/vec_env.rs:152` | `step_all` delegates to the trait; the for-loop body lives only in the sequential impl. |
| `crates/clankers-gym/src/vec_env.rs:140-145` | `collect_step_results` iterates `VecObsBuffer::row(i)` (W3) instead of `self.runner.get_obs(i)`. |
| `crates/clankers-gym/src/protocol.rs` (post-W4 line range) | Extend the W4 `EncodedObservation` enum with `BatchF32 { num_envs: u32, dim: u32, #[serde(skip)] payload: Vec<u8> }` and `BatchRawU8Image { num_envs: u32, width: u32, height: u32, channels: u8, layout: ImageLayout, #[serde(skip)] payload: Vec<u8> }`. Bump `PROTOCOL_VERSION` to `1.2.0`. |
| `crates/clankers-gym/src/server.rs` (post-W4 lines) | In the Init handshake, negotiate a `binary_batch: bool` capability alongside W4's `binary_obs: bool` and `image_on_reset: bool`. If client requests binary batch and server supports it, the batch-reset and batch-step encoders switch to `binary_frame::encode_batch_f32` / `encode_batch_raw_u8`. JSON path remains the default for legacy clients. |
| `crates/clankers-record/src/recorder.rs:22` | Add `crate::async_writer::AsyncRecorder` re-export. |
| `crates/clankers-record/src/recorder.rs:93` | `Recorder::writer` becomes `Option<RecorderBackend>` where `RecorderBackend` is `Sync(McapWriter<BufWriter<File>>)` or `Async(AsyncRecorder)`. Default stays `Sync` for back-compat. |
| `crates/clankers-record/src/recorder.rs:159` | `write_json` dispatches on backend: `Sync` keeps the existing direct write; `Async` `try_send`s into the bounded channel and increments `DroppedFrames` resource on `TrySendError::Full`. |
| `crates/clankers-record/src/recorder.rs:372`, `:417`, `:443`, `:469`, `:523` | Per-frame system writers route through `write_json` (no per-system change); the async dispatch happens once at the backend boundary. |
| `crates/clankers-physics/src/rapier/systems.rs` | Convert hot-path `HashMap<Entity, RigidBodyHandle>` and `HashMap<Entity, ImpulseJointHandle>` lookups to `Vec<BodyRuntime>` and `Vec<JointRuntime>` indexing by layout slot. Keep the `HashMap`s as setup-only name → slot tables exposed via `BodyHandles::slot_for(entity)`. |
| `crates/clankers-env/src/buffer.rs` (around `:137`) | Add `VecObsBuffer::row(env_idx: usize) -> ObservationView<'_>` (W3 prerequisite already adds the per-buffer version; this is the vec analogue used by the binary encoder). |
| `apps/clankers-app/src/commands/bench.rs` (created in W5 PR4 as stubs) | Implement `bench vec`, `bench protocol`, `bench record`, `bench mpc` bodies. Each prints a human table and, with `--json`, a single-line JSON document. Each accepts `--csv <path>` for the baseline comparison. |
| `.github/workflows/ci.yml` | Add a `cargo bench --bench vec -j 24 -- --output-format json` step that pipes into `scripts/compare_baseline.py benches/current.csv benches/baseline.csv --tolerance 0.15`. (CI yaml edit lives in the implementation PR; this plan only specifies the step.) |
| `crates/clankers-mpc/src/lib.rs` | No code change in W7; the `bench mpc` body calls into the existing solver entry point and records p50/p95/p99 distributions. Documented as "consumes W7's bench harness, does not modify MPC". |

### DELETE

- None. Sequential `VecEnvRunner` survives W7. The synchronous recorder
  backend survives W7. HashMap `BodyHandles` / `JointHandles` survive W7
  as setup-time name→slot lookup tables. W7 is purely additive plus the
  hot-path rewiring.

## 5. Checklist items

Each item is ≤300 LOC and lands as part of one of the four PRs in
section 9. Items are listed in the PR order.

### PR1 — `ParallelVecEnvRunner`

- [ ] Define a `pub trait VecRunnerLike: Send` in
      `crates/clankers-env/src/vec_runner.rs` with methods `step_all`,
      `reset_all`, `get_obs`, `num_envs`. Implement it for the existing
      sequential `VecEnvRunner`.
- [ ] Add `Send` bound on `Box<dyn VecEnvInstance>` at
      `crates/clankers-gym/src/vec_env.rs:67`. Document under
      `// SAFETY:`-style comment that env instances must not share
      globally-mutable Bevy resources (asset server, render device).
      Compile-time check via a `static_assertions::assert_impl_all!`
      macro on the canonical example env type (cartpole).
- [ ] Implement `crates/clankers-env/src/parallel_runner.rs` with
      `ParallelVecEnvRunner { envs: Vec<Mutex<Box<dyn VecEnvInstance +
      Send>>>, config: VecEnvConfig }` (mutex needed for `par_iter_mut`
      ergonomics; contention is zero because each env is touched by
      exactly one worker). `step_all` does
      `self.envs.par_iter_mut().enumerate().map(|(i, env)| env.lock().step(&actions[i])).collect()`.
- [ ] Implement deterministic per-env seed: `seed(i) = base_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64)` (a splitmix64-style derivation). Document the formula.
- [ ] Implement `impl VecRunnerLike for ParallelVecEnvRunner` so
      `GymVecEnv` is agnostic.
- [ ] Add factory `pub fn runner_for(envs, config) -> Box<dyn VecRunnerLike>`
      that returns `ParallelVecEnvRunner` when `config.parallel == true`,
      else `VecEnvRunner`. Default for `VecEnvConfig::new()` stays
      `parallel: false` for back-compat.
- [ ] Write `crates/clankers-env/tests/parallel_determinism.rs` containing
      `parallel_vec_env_seed_assignment_is_deterministic`
      (N=16 cartpole envs, seeds [0..16], 100 steps, run 3 times, assert
      every collected observation `Vec<f32>` byte-equal across all three
      runs).
- [ ] Add a `[dev-dependencies]` `rayon = "1.10"` entry to
      `crates/clankers-env/Cargo.toml`. Use `rayon` (not `rayon-core`)
      so the thread-pool config is centralised.

### PR2 — Binary protocol frame

- [ ] Extend W4's `EncodedObservation` enum (post-W4 line range) with
      `BatchF32 { num_envs: u32, dim: u32, #[serde(skip)] payload: Vec<u8> }`
      and `BatchRawU8Image { num_envs, width, height, channels, layout, #[serde(skip)] payload: Vec<u8> }`.
      Payloads ride on a separate length-prefixed binary frame
      immediately after the JSON envelope (matches W4's existing
      `RawU8Image` pattern).
- [ ] Implement `crates/clankers-gym/src/binary_frame.rs` with
      `BinaryFrameHeader { version: u32, kind: u8, num_envs: u32, dim: u32, _reserved: u32 }`
      (24 bytes, all little-endian) and free functions
      `encode_batch_f32(envs: &[ObservationView]) -> Vec<u8>`,
      `decode_batch_f32(bytes: &[u8]) -> Result<(BinaryFrameHeader, &[f32]), BinaryFrameError>`.
      Use `bytemuck::cast_slice::<f32, u8>` for zero-copy on
      little-endian hosts; document that big-endian hosts pay an `f32`
      byte-swap pass (Clankers currently targets x86_64 and aarch64,
      both LE).
- [ ] Implement the same pair for `BatchRawU8Image` (payload is just the
      concatenated `Vec<u8>`s; no endianness concern).
- [ ] Bump `PROTOCOL_VERSION` to `1.2.0` and add a CHANGELOG entry.
      Negotiate `binary_batch: bool` capability in the Init handshake;
      server downgrades to JSON when client omits the capability.
- [ ] Server-side rewiring at the post-W4 batch handler: if
      `session.binary_batch` is true and the
      observation is `Continuous`, call
      `binary_frame::encode_batch_f32(&views)`; otherwise call the
      existing JSON path.
- [ ] Write `crates/clankers-gym/tests/binary_protocol_roundtrip.rs`
      containing `batch_f32_roundtrip_byte_equal` (random
      `Vec<Vec<f32>>` of `num_envs=8, dim=17`, encode then decode, assert
      `bytemuck::cast_slice` round-trips byte-equal) and
      `batch_raw_u8_roundtrip_byte_equal` (random
      `Vec<Vec<u8>>` of `num_envs=4, w=64, h=64, c=3`, encode then
      decode, assert byte-equal).
- [ ] Document the wire format in the
      `crates/clankers-gym/src/protocol.rs` module doc — extend the W4
      hex example with a `BatchF32` frame annotation.

### PR3 — Dense `*Runtime` structs

- [ ] Create `crates/clankers-physics/src/rapier/runtime.rs` defining
      `pub struct BodyRuntime { pub entity: Entity, pub handle:
      RigidBodyHandle, pub layout_slot: usize }`, `pub struct
      JointRuntime { pub entity: Entity, pub handle: ImpulseJointHandle,
      pub layout_slot: usize, pub motor: MotorOverrideParams }`, and
      `pub struct MotorRuntime { pub handle: ImpulseJointHandle, pub
      params: MotorOverrideParams }` aliased through `JointRuntime` for
      hot-loop ergonomics.
- [ ] Add `RobotGroup::compile_runtime(&self, layout: &JointLayout) ->
      Vec<JointRuntime>` in `crates/clankers-sim/src/builder.rs`.
      Walks layout in slot order, resolves each joint name through the
      existing setup-time `HashMap<String, Entity>`, panics with a
      structured `LayoutCompileError` if any layout joint is missing
      (this is the W2-validated `validate_motor_coverage` invariant
      lifted into a typed compile step).
- [ ] Migrate the hot-path lookups in
      `crates/clankers-physics/src/rapier/systems.rs` from
      `BodyHandles: HashMap<Entity, RigidBodyHandle>` / `JointHandles:
      HashMap<Entity, ImpulseJointHandle>` to indexed `Vec<BodyRuntime>` /
      `Vec<JointRuntime>` reads. Keep the `HashMap`s as setup-time
      name-lookup resources exposed via
      `BodyHandles::slot_for(entity: Entity) -> Option<usize>`.
- [ ] `MotorOverrides` is already `Vec<(JointHandle, MotorOverrideParams)>`
      from W2 PR1; align the field name and slot order with
      `Vec<JointRuntime>` so the same `for joint_runtime in &joints`
      loop drives both motor application and joint-state read-back.
- [ ] Write `crates/clankers-physics/tests/dense_runtime.rs` containing
      `dense_runtime_matches_hashmap_lookup` (same arm scene built two
      ways: once with HashMap lookups, once with `Vec<JointRuntime>`;
      step 100×; assert every body's world-space pose is byte-equal at
      every step).
- [ ] Add a feature flag `dense-runtime` (default on) so the migration
      can be reverted in a single release window if a downstream
      regression appears.

### PR4 — Async recorder + bench CLI + baseline CSV + CI gate

- [ ] Implement `crates/clankers-record/src/async_writer.rs` with
      `AsyncRecorder { tx: Sender<RecorderMessage>, dropped: Arc<AtomicU64> }`
      and a worker thread that consumes `RecorderMessage::{Frame {
      channel_id, payload, header }, Flush, Close}` from a
      `crossbeam_channel::bounded(256)` queue. Bounded capacity
      configurable via `RecordingConfig::async_buffer_capacity`
      (default 256 frames).
- [ ] Add `DroppedFrames(Arc<AtomicU64>)` Bevy resource exposed via
      `clankers info --record-stats`. Increment on every
      `try_send` returning `TrySendError::Full`.
- [ ] Wire `crates/clankers-record/src/recorder.rs:93,159` through the
      new backend enum; sync path remains the default for back-compat;
      `RecordingConfig::async_mode: bool` (default false) flips to async.
- [ ] Write `crates/clankers-record/tests/async_backpressure.rs`
      containing `async_recorder_drops_frames_under_backpressure`
      (slow-filesystem stub via `std::io::sink()` wrapped in a `Sleep`
      adapter; push 10_000 frames at 1 kHz; assert dropped-frame counter
      > 0 AND recorder doesn't deadlock AND final `Close` flushes
      gracefully).
- [ ] Implement the four `bench` subcommand bodies in
      `apps/clankers-app/src/commands/bench.rs` (the scaffolding stub
      created in W5 PR4). Each accepts `--envs N1,N2,...`, `--json`,
      `--csv <path>`, `--seed`. Output schema:
      `{"bench":"vec","num_envs":8,"steps_per_sec":12345.6,
      "p50_us":80.1,"p95_us":120.4,"p99_us":180.7}`.
- [ ] Add `benches/vec.rs` Criterion target that delegates to the same
      core helper used by `clankers bench vec`. Wire
      `[[bench]] name = "vec"` in the appropriate crate's `Cargo.toml`.
- [ ] Generate and commit `benches/baseline.csv` by running
      `cargo bench --bench vec -j 24 -- --output-format json |
      scripts/json_to_csv.py > benches/baseline.csv` on the reference
      developer machine. Document the regeneration command in
      `benches/README.md`.
- [ ] Add `scripts/compare_baseline.py` (Python 3.11+, no third-party
      deps — uses stdlib `csv` + `argparse`) implementing
      `compare(current_csv, baseline_csv, tolerance: float) -> int`.
      Return 0 on all rows within tolerance; 1 on any
      `(baseline - current) / baseline > tolerance` (i.e. > 15% slower).
- [ ] Add the CI step to `.github/workflows/ci.yml` running
      `cargo bench --bench vec -j 24 -- --output-format json >
      benches/current.json` followed by the comparator invocation. The
      step is `continue-on-error: false` so a regression blocks merge.

## 6. Tests required before implementation (test-first)

Test placement follows
`.delegate/work/20260525-185618-workstream-plans/TASK.md` § "Test
placement conventions". Every test below is written and committed (red)
in the same PR as the production code it covers, immediately before the
implementation commit.

### `crates/clankers-env/tests/parallel_determinism.rs`

- `parallel_vec_env_seed_assignment_is_deterministic` — construct a
  `ParallelVecEnvRunner` with N=16 cartpole envs, base seed = `0xDEADBEEF`,
  derived per-env seeds via the documented splitmix64 step. Run
  `reset_all(Some(base_seed))` then 100 `step_all(&actions)` calls with
  a zero-action vector. Collect the final `VecObsBuffer` flat row for
  every env. Repeat the entire run **three times** with the same base
  seed. Assert every collected `Vec<f32>` is bit-equal across all
  three runs and across all 16 envs.
- `parallel_vec_env_changes_base_seed_changes_output` — same fixture
  with `base_seed = 0xDEADBEEF` vs `base_seed = 0xC0FFEE`; assert at
  least one observation differs (sanity check that seeding is wired).
- `parallel_vec_env_preserves_result_ordering` — env at slot 3 returns a
  deterministic `Observation::Continuous(vec![3.0; 4])` synthetic value
  on step; collected batch must have `observations[3]` at index 3 (not
  shuffled by Rayon work-stealing).

### `crates/clankers-gym/tests/binary_protocol_roundtrip.rs`

- `batch_f32_roundtrip_byte_equal` — generate `num_envs=8`, `dim=17`,
  random `f32` payload with a fixed seed. Call
  `binary_frame::encode_batch_f32(&views)` then
  `binary_frame::decode_batch_f32(&bytes)`. Assert decoded header equals
  expected and decoded slice byte-equals the original concatenated
  `Vec<f32>` (via `bytemuck::cast_slice::<f32, u8>`).
- `batch_raw_u8_roundtrip_byte_equal` — same shape contract for
  `num_envs=4, w=64, h=64, c=3, layout=Hwc`. Random `u8` payload with a
  fixed seed. Encode then decode, assert header equals expected and
  payload byte-equals original.
- `binary_frame_header_size_is_24_bytes` — `assert_eq!
  (std::mem::size_of::<BinaryFrameHeader>(), 24)` — pins the wire size.
- `binary_frame_rejects_version_mismatch` — manually construct a frame
  with `version: 0xFFFF_FFFF`; decode returns
  `Err(BinaryFrameError::UnsupportedVersion)`.

### `crates/clankers-record/tests/async_backpressure.rs`

- `async_recorder_drops_frames_under_backpressure` — build an
  `AsyncRecorder` with `capacity=4` writing into a `SleepingSink` that
  blocks 50 ms per write. Push 100 frames as fast as possible. After
  pushing, call `recorder.close()`. Assert: `dropped_frames > 0`, no
  panic, no deadlock (wrap whole test in `std::thread::spawn` with a
  5-second join timeout).
- `async_recorder_zero_drops_when_buffer_sufficient` — same fixture with
  `capacity=1024` and 100 frames; assert `dropped_frames == 0`.
- `async_recorder_close_flushes_pending_frames` — push 10 frames with a
  fast sink, immediately `close()`, then read the sink contents; assert
  all 10 frames were written.

### `crates/clankers-physics/tests/dense_runtime.rs`

- `dense_runtime_matches_hashmap_lookup` — construct the same arm scene
  twice: scene A uses the legacy `BodyHandles: HashMap` path, scene B
  uses `Vec<BodyRuntime>` indexed by layout slot. Step both 100 times
  with the same action sequence and same seed. After every step, assert
  every body's `Isometry3` (translation + rotation as bit-equal `f32`)
  matches between the two scenes.
- `compile_runtime_rejects_missing_layout_joint` — build a layout
  containing a joint name absent from `RobotGroup`; assert
  `compile_runtime` returns `Err(LayoutCompileError::MissingJoint {
  name: "phantom_joint" })`.
- `compile_runtime_orders_by_layout_slot` — layout has 4 joints in order
  `["a", "b", "c", "d"]`; `RobotGroup` was built with insertion order
  `["d", "a", "c", "b"]`. Assert `compile_runtime(&layout)` returns a
  `Vec<JointRuntime>` whose slot-0 entry has name `"a"`, slot-1 has
  `"b"`, etc.

### Inline `#[cfg(test)] mod tests` in `crates/clankers-env/src/parallel_runner.rs`

- `seed_derivation_is_deterministic` — `seed(0xDEAD, 0)` always equals
  the same constant; `seed(0xDEAD, 1) != seed(0xDEAD, 0)`; `seed(0, i)`
  is non-zero for `i in 0..16`.

All tests above must compile and **fail** when committed in the
test-first sub-step of each PR; the same PR's subsequent production
commit turns them green.

## 7. Success criteria

Every criterion is checkable with one concrete command. All cargo
invocations use `-j 24` per CLAUDE.md ("machine has 32 cores — leave
headroom for the OS"):

- `cargo test -j 24 -p clankers-env --test parallel_determinism` exits
  0; reports 3 passed.
- `cargo test -j 24 -p clankers-gym --test binary_protocol_roundtrip`
  exits 0; reports 4 passed.
- `cargo test -j 24 -p clankers-record --test async_backpressure` exits
  0; reports 3 passed.
- `cargo test -j 24 -p clankers-physics --test dense_runtime` exits 0;
  reports 3 passed.
- `cargo test -j 24 --workspace` exits 0 (no regression from the dense-
  runtime migration or async recorder rewiring).
- `clankers bench vec --envs 1,2,4,8 --json -j 24` writes a JSON array;
  `jq '.[] | select(.num_envs == 8) | .steps_per_sec' | head -1` divided
  by `jq '.[] | select(.num_envs == 1) | .steps_per_sec' | head -1` is
  **≥ 3.0** — the **≥3× wall-clock speedup at 8 envs vs sequential**
  gate from LOOPS.yaml loop 7.
- `clankers bench protocol --json -j 24` reports `binary_steps_per_sec /
  json_steps_per_sec` **≥ 2.0** on 16-dim observation, 8-dim action
  payloads — the **≥2× binary-vs-JSON throughput** gate from LOOPS.yaml
  loop 7.
- `cargo bench --bench vec -j 24 -- --output-format json >
  benches/current.json` exits 0 (Criterion runs without panic).
- `scripts/compare_baseline.py benches/current.csv benches/baseline.csv
  --tolerance 0.15` exits 0 (no row >15% slower than the committed
  baseline). Non-zero exit blocks the CI step that follows.
- `clankers info --record-stats` prints a `dropped_frames` row sourced
  from the `DroppedFrames` Bevy resource (proves the metric is wired
  end-to-end).
- `cargo clippy -j 24 --workspace --all-targets --tests --benches -- -D
  warnings` exits 0 (no new clippy warnings from any of the four PRs).
- `RAYON_NUM_THREADS=24 cargo bench --bench vec -j 24` exits 0 — proves
  the bench harness honours the project parallelism ceiling.

## 8. Risks & mitigations

1. **Rayon over Bevy app instances may have global-state contention.**
   Bevy ships several singletons (asset server, render device, task
   pool) that envs would silently share. A naive
   `par_iter_mut().for_each(|env| env.step())` against envs that all
   load assets through one global `AssetServer` will serialise on the
   asset-server mutex and look like no parallel speedup at all.
   **Mitigation:** require every `VecEnvInstance` to own a fully
   independent `bevy::app::App`. Document the rule in the
   `VecEnvInstance: Send` trait doc comment. Add a
   `static_assertions::assert_impl_all!(CartpoleVecEnv: Send)` line in
   the canonical example env so the bound is enforced at compile time.
   The `parallel_vec_env_seed_assignment_is_deterministic` test
   exercises 16 independent cartpole envs precisely so a regression to
   shared-state behaviour shows up as cross-env data corruption.

2. **Binary protocol fragmentation across clients in flight.** Existing
   Python clients (and any third-party integrations) speak the W4 JSON-
   only protocol. Flipping to binary by default mid-release would
   silently break them.
   **Mitigation:** explicit `protocol_version: u32` in the Init
   handshake (already present from W4, bumped to `1.2.0` by W7 PR2).
   Add `binary_batch: bool` capability flag. Server downgrades to JSON
   when the client omits the capability. The CHANGELOG entry calls out
   "binary batch is opt-in for `1.2.0`; will become default in `2.0.0`".
   `clankers protocol smoke --binary` and `--json` both pass on a
   `1.2.0` server.

3. **Dense-runtime migration risks silent ordering bugs.** Converting
   `HashMap<Entity, …>` to `Vec<…Runtime>` indexed by layout slot is
   easy to get wrong: an off-by-one in the layout-to-slot mapping
   would corrupt every per-joint quantity (motor targets, sensor
   reads, recorder rows) in a way that simulation may continue without
   crashing.
   **Mitigation:** the `dense_runtime_matches_hashmap_lookup` test
   compares byte-equal world-space pose between HashMap and dense
   builds across 100 steps; any slot mismatch produces a diverging
   `Isometry3`. Ship dense runtime behind the `dense-runtime` feature
   flag (default on) so a downstream regression can be reverted with
   one cargo feature toggle for one release window.

4. **Async recorder buffer overflow under pathological load.** Image
   recording at 4K@60Hz pushes ~1.5 GiB/s through the channel; a 256-
   frame default would back up in 4 seconds on a slow disk.
   **Mitigation:** bounded channel default 256 frames with explicit
   `RecordingConfig::async_buffer_capacity` override; dropped-frame
   counter is a `pub` `Arc<AtomicU64>` surfaced via `clankers info
   --record-stats` and printed at the end of every `clankers record`
   run. Add a CLI flag `clankers record --max-dropped-frames N` that
   fails the recording with non-zero exit if the counter ever exceeds
   N (CI integration sets `--max-dropped-frames 0` to catch silent
   drops in tests).

5. **`cargo bench` numbers vary across hosts and across runs.** A 15%
   regression gate is tight; CPU thermals, scheduler jitter, and
   background processes can produce >15% variance even with no code
   change.
   **Mitigation:** `RAYON_NUM_THREADS=24` (matches `-j 24`) caps
   parallelism. Bench harness runs each cell 10× with Criterion's
   default outlier rejection. Baseline CSV is regenerated by the
   maintainer on a quiet machine and committed. CI runs on the same
   GitHub runner class so absolute numbers shift only when the runner
   class changes. The compare script tolerates `--tolerance 0.15`
   (15%) which empirically covers Criterion-reported noise without
   masking real regressions; a tighter tolerance can be considered
   once we have a month of CI history.

6. **`Mutex` inside `ParallelVecEnvRunner` looks like a parallelism
   anti-pattern.** Reviewers will reasonably ask why each env is
   `Mutex<Box<dyn VecEnvInstance + Send>>`.
   **Mitigation:** document inline that `par_iter_mut` requires
   `&mut Vec<…>` while the natural Rayon idiom for "one worker, one
   env" wants per-element `&mut`. The `Mutex` carries zero contention
   because every env is locked by exactly one worker for its entire
   step, so the cost is one uncontended atomic per step (~1 ns). An
   alternative `slice::par_iter_mut::<Box<dyn VecEnvInstance>>` API
   would avoid the mutex but require the trait bound `T: 'static +
   Send + Sync`, which is a stronger guarantee than what envs can
   reasonably provide. The 3× speedup criterion verifies empirically
   that the mutex is not a bottleneck.

## 9. PR breakdown

Exactly **4 commits**, matching LOOPS.yaml loop 7's
`expected_implementation_prs: 4`.

### PR1 — `feat(env): add ParallelVecEnvRunner with deterministic seeding`

Scope: introduce `VecRunnerLike` trait, sequential impl, parallel impl,
`Send` bound, factory. No protocol changes, no recorder changes, no
physics changes.

Files (diff estimate):
- `crates/clankers-env/src/vec_runner.rs` (~+80 LOC: `VecRunnerLike`
  trait, impl for `VecEnvRunner`).
- `crates/clankers-env/src/parallel_runner.rs` (~+180 LOC new).
- `crates/clankers-env/Cargo.toml` (+2 LOC: `rayon` dep).
- `crates/clankers-gym/src/vec_env.rs` (~+30 LOC, ~−20 LOC: `Send`
  bound, factory call, trait-object field).
- `crates/clankers-env/tests/parallel_determinism.rs` (~+120 LOC new).

Total ≈ 410 LOC across 5 files. Sub-commits: (a) tests-first commit
(red), (b) trait + sequential impl (green for sequential side), (c)
parallel impl + factory (green for parallel determinism test).

**Checklist items covered:** PR1 § 5 bullets 1–7 (8 items).

### PR2 — `feat(gym): binary batch frame for high-throughput training`

Scope: extend `EncodedObservation` with batch variants, implement
`binary_frame.rs`, wire server-side encoder, version handshake. No env
runner changes, no recorder changes, no physics changes.

Files (diff estimate):
- `crates/clankers-gym/src/protocol.rs` (~+60 LOC: two new enum
  variants, `PROTOCOL_VERSION` bump, doc-example extension).
- `crates/clankers-gym/src/binary_frame.rs` (~+200 LOC new: header,
  encode/decode for f32 and u8 batches, error enum).
- `crates/clankers-gym/src/server.rs` (~+40 LOC: capability negotiation
  + dispatch in batch handlers).
- `crates/clankers-gym/Cargo.toml` (+2 LOC: `bytemuck` dep).
- `crates/clankers-gym/tests/binary_protocol_roundtrip.rs` (~+150 LOC
  new).
- `crates/clankers-gym/CHANGELOG.md` (+10 LOC: 1.2.0 entry).

Total ≈ 460 LOC across 6 files. Sub-commits: (a) tests-first commit
(red), (b) binary_frame module (green for roundtrip tests), (c) protocol
extension + server wiring + version bump.

**Checklist items covered:** PR2 § 5 bullets 1–7 (7 items).

### PR3 — `perf(physics): compile setup HashMaps to dense runtime vectors`

Scope: introduce `BodyRuntime` / `JointRuntime`, `RobotGroup::
compile_runtime`, migrate `rapier/systems.rs` hot loops. No runner
changes, no protocol changes, no recorder changes.

Files (diff estimate):
- `crates/clankers-physics/src/rapier/runtime.rs` (~+120 LOC new).
- `crates/clankers-sim/src/builder.rs` (~+80 LOC: `compile_runtime`
  helper + `LayoutCompileError`).
- `crates/clankers-physics/src/rapier/systems.rs` (~+200 LOC, ~−150
  LOC: replace HashMap lookups with indexed Vec reads in every hot loop;
  keep HashMap as setup-time `slot_for(entity)` lookup).
- `crates/clankers-physics/Cargo.toml` (+2 LOC: `dense-runtime` feature
  flag, default on).
- `crates/clankers-physics/tests/dense_runtime.rs` (~+200 LOC new).

Total ≈ 600 LOC across 5 files. Sub-commits: (a) tests-first commit
(red, comparing HashMap vs dense paths on the same scene), (b) runtime
module + compile_runtime helper (green for compile + ordering tests),
(c) systems.rs hot-loop migration (green for byte-equal pose test).

**Checklist items covered:** PR3 § 5 bullets 1–6 (6 items).

### PR4 — `feat(record,cli): async recorder, bench subcommand bodies, CI baseline gate`

Scope: async recorder with bounded channel + dropped-frame metric;
populate `clankers bench vec/protocol/record/mpc` bodies; commit baseline
CSV; add CI regression gate. Depends on PR1–PR3 being merged because
the bench bodies measure their performance contributions.

Files (diff estimate):
- `crates/clankers-record/src/async_writer.rs` (~+200 LOC new).
- `crates/clankers-record/src/recorder.rs` (~+50 LOC, ~−10 LOC: backend
  enum, async dispatch in `write_json`).
- `crates/clankers-record/Cargo.toml` (+2 LOC: `crossbeam-channel`
  dep).
- `crates/clankers-record/tests/async_backpressure.rs` (~+150 LOC new).
- `apps/clankers-app/src/commands/bench.rs` (~+250 LOC: implement four
  subcommand bodies on top of W5 PR4's stubs).
- `apps/clankers-app/src/commands/info.rs` (~+15 LOC:
  `--record-stats` flag printing `DroppedFrames`).
- `benches/vec.rs` (~+60 LOC new Criterion target).
- `benches/baseline.csv` (~+40 LOC new committed CSV).
- `benches/README.md` (~+20 LOC new — regeneration command, expected
  variance window).
- `scripts/compare_baseline.py` (~+80 LOC new).
- `.github/workflows/ci.yml` (~+12 LOC: bench step + comparator
  invocation).

Total ≈ 880 LOC across 11 files. Sub-commits: (a) tests-first commit
(red — backpressure test, bench subcommand smoke tests asserting
`bench vec --envs 1 --json` produces parseable JSON), (b) async writer
module (green for backpressure), (c) bench subcommand bodies + Criterion
target, (d) baseline CSV + comparator + CI step.

**Checklist items covered:** PR4 § 5 bullets 1–9 (9 items).

After PR4: every command in section 7 holds on the merge commit. CI
fails any future PR whose `cargo bench --bench vec -j 24` is >15% slower
than `benches/baseline.csv` on the same runner class.
