# WS5 — `clankers` CLI as Quality Gate

Workstream 5 of 8. Source: Workstream Catalogue in
`.delegate/work/20260525-185618-workstream-plans/TASK.md`.
Authoritative reference: `notes/clankers_codebase_quality_report_2026-05-25.md`,
section **"Proposed Rust CLI"** (lines 508–659). This workstream does not tie
to a single numbered finding; it implements the report's CLI design
requirements in full.

## 1. Goal

Promote `apps/clankers-app` from a four-subcommand demo
(`headless`/`serve`/`viz`/`info`) into the project's primary entry point and
quality gate, exposing `info` / `validate` / `inspect` / `run` / `serve` /
`record` / `replay` / `bench` with uniform `--json`, `--seed`, and `--scenario`
flags, and defining a new `crates/clankers-sim/src/scenarios/` module that
hosts first-class scenario builders (`arm_pick`, `cartpole`) consumed by both
the CLI and (later, in W8) the example bins.

## 2. Why this workstream, why this order

This workstream depends on the two foundational workstreams and unblocks the
two trailing ones:

- **Depends on W1** (`clankers-core` contracts). `clankers validate` consumes
  `JointLayout`, `ObservationSchema`, `ActionSchema`, and `RecorderSchema`
  through their `validate_against(&other) -> Result<(), SchemaMismatch>` API
  defined in WS1 PR1. `clankers inspect urdf --json` prints
  `JointLayout.hash()`. Without W1's types the CLI has no contract to
  validate against — today's `actuated_joints()` returns a `HashMap` whose
  iteration order is nondeterministic.
- **Depends on W2** (layout-bound sensors and actions). `clankers validate
  --scenario <name> --strict` surfaces
  `RobotGroup::validate_motor_coverage(&layout)` (introduced in WS2 PR1)
  which checks that every joint — arm AND gripper — has a `MotorOverride`,
  the convention recorded in `MEMORY.md` ("MotorOverrides — ALL Joints Must
  Be Overridden"). Without W2's validator, `validate` has nothing to call.
- **Depends on W4 (partial)** for `serve --protocol binary`. PR2 below
  consumes `EncodedObservation` from WS4 PR1 (`crates/clankers-gym/src/encoding.rs`).
  If W4 slips, PR2 ships with `--protocol json` only and the binary mode is
  delivered in a follow-up PR2b.
- **Unblocks W7** — `clankers bench` is the CSV/JSON output sink that W7's
  `ParallelVecEnvRunner` and binary-protocol benchmark code write to. W7
  PR4 (async recorder + bench CLI) extends the `bench` subcommand tree
  that W5 PR4 introduces.
- **Unblocks W8** — W8 PR1/PR2 lift robot scene setup out of
  `examples/src/bin/*.rs` into `crates/clankers-sim/src/scenarios/` and
  shrink each example to a ≤100-line wrapper over the scenarios W5
  declares. W8 PR3's new CI step (`clankers validate --scenario <each>`)
  invokes the validate command W5 ships in PR1.

The order — W5 after W1+W2, before W7+W8 — is forced: the CLI cannot
validate contracts that do not exist (W1), cannot surface motor-coverage
errors without the validator (W2), and is itself the gating surface that
W7's CSV baselines and W8's CI invocations depend on.

The "Proposed Rust CLI" section of the quality report (lines 508–659)
spells out the eight mandatory commands (`info`, `validate`, `bench`,
`serve`, `run`, `replay`, `record`, `inspect`); W5 implements all but
`export` (deferred — `export` is a Phase 5 follow-up not tied to any of
the eight workstreams).

## 3. Out of scope

The following look adjacent but are deliberately deferred:

- **No performance work.** `clankers bench` subcommands are wired as
  stubs in PR4 that call into W7 internals; the actual `ParallelVecEnvRunner`,
  binary batch frame, dense `Vec<…Runtime>` rewrites, and async recorder
  are W7 PR1–PR4. W5 ships the CLI surface that W7 fills in.
- **No example-bin migration.** The 22 binaries under `examples/src/bin/`
  continue to ship alongside CLI scenarios for one release. Lifting their
  scene setup into `crates/clankers-sim/src/scenarios/` and shrinking each
  bin to ≤100 lines is W8 PR1+PR2. W5 only declares the scenarios module
  and ships two reference scenarios (`arm_pick`, `cartpole`).

### 3.1 First-class scenario catalog

W5 declares the full set of first-class scenarios (so `clankers run
--scenario <name>` recognises every name from PR2 onward), but ships only
**two** as compiled reference implementations (`arm_pick`, `cartpole`).
The remaining four are registered with stubbed `build()` bodies that
return `Err(ScenarioError::NotImplementedYet { migrated_in: "W8" })`.
W8 PR1 fills `arm_*`; W8 PR2 fills the rest.

| Scenario name      | W5 status      | Bin(s) replaced (lifted in W8) |
|--------------------|----------------|--------------------------------|
| `arm_pick`         | reference impl | `examples/src/bin/arm_pick_gym.rs` (281 lines), `arm_pick_record.rs` (526), `arm_pick_replay.rs` (876) |
| `cartpole`         | reference impl | `examples/src/bin/cartpole_gym.rs` (156), `cartpole_vec_gym.rs` (147), `cartpole_policy_viz.rs` (340), `cartpole_vec_benchmark.rs` (174) |
| `pendulum`         | stub (W8 PR2)  | `examples/src/bin/pendulum_headless.rs` (113), `pendulum_viz.rs` (393) |
| `quadruped_trot`   | stub (W8 PR2)  | `examples/src/bin/quadruped_mpc.rs` (299), `quadruped_mpc_viz.rs` (965), `quadruped_mpc_bench.rs` (491) |
| `multi_robot`      | stub (W8 PR2)  | `examples/src/bin/multi_robot.rs` (188), `multi_robot_viz.rs` (489) |
| `arm_ik`           | stub (W8 PR1)  | `examples/src/bin/arm_ik.rs` (95), `arm_ik_viz.rs` (646) |

Total: 6 scenarios named (within the gate's 4–6 range); 14 of the 22
example bins mapped. The remaining 8 bins (`arm_gym.rs`, `arm_with_policy.rs`,
`arm_manipulation.rs`, `arm_bench.rs`, `arm_policy_viz.rs`, `arm_pick_record.rs`
viz overlays, `domain_rand.rs`, `arm_ik`'s pure-IK demo) are W8 micro-scenarios
or viz overlays not constitutive of a new first-class scenario.
- **No CI changes.** Adding the `clankers validate --scenario <each>` step
  to `.github/workflows/ci.yml`, swapping `cargo test --workspace
  --exclude clankers-examples` for `cargo test --workspace`, and enforcing
  the ≤100-line cap on example bins are all W8 PR3.
- **No synthetic-compiler changes.** Declaring `action_semantics` in the
  synthetic compiler's trace schema is W8 PR3 (`python/clankers_synthetic/compiler.py`).
- **No `viz` rendering features.** `clankers viz` may grow `--scenario`
  selection (in PR2) but no new rendering, no new shaders, no policy
  inspector changes. The Cosmos pipeline (`MEMORY.md`: "Cosmos Pipeline
  (Sim-to-Real)") is untouched.
- **No `export` subcommand.** MCAP→NPZ/Zarr/Arrow conversion (quality
  report line 643–650) is a separate Phase 5 deliverable, not part of W5.
- **No public Python CLI wrapper.** `clankers` is a Rust binary. A `python
  -m clankers` shim is out of scope; Python users continue to use
  `clankers_gym` directly.
- **No `protocol` introspection (`clankers protocol smoke`).** Quality
  report line 150 mentions this; deferred to a follow-up because it
  duplicates `clankers inspect protocol` in PR1.

## 4. Files to change

### NEW

| Path | Purpose |
|------|---------|
| `crates/clankers-sim/src/scenarios/mod.rs` | Scenarios module root. Defines `ScenarioConfig` struct (seed, max_steps, headless, record_path), `ScenarioBuilder` trait (`fn build(&self, app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle`), `ScenarioRegistry` (string-keyed `HashMap<&'static str, Box<dyn ScenarioBuilder>>`), and `register_builtin(&mut registry)` that wires `arm_pick` and `cartpole`. |
| `crates/clankers-sim/src/scenarios/arm_pick.rs` | First reference scenario. `pub fn build(app: &mut App, cfg: &ScenarioConfig)`. Mirrors the minimum-viable scene setup currently in `examples/src/bin/arm_pick_gym.rs` (lines for URDF load + table + cube + camera) — restricted to the subset that does NOT depend on viz. |
| `crates/clankers-sim/src/scenarios/cartpole.rs` | Second reference scenario. Mirrors `examples/src/bin/cartpole_gym.rs` headless path. |
| `apps/clankers-app/src/commands/mod.rs` | Submodule root; re-exports one `run` fn per command file. |
| `apps/clankers-app/src/commands/info.rs` | `info` command implementation (expanded). |
| `apps/clankers-app/src/commands/validate.rs` | `validate` command implementation (`--urdf`, `--scene`, `--policy`, `--recording-schema`, `--strict`, `--json`, `--scenario`). |
| `apps/clankers-app/src/commands/inspect.rs` | `inspect mcap|onnx|urdf|scene` subtree. |
| `apps/clankers-app/src/commands/run.rs` | `run --scenario` command. |
| `apps/clankers-app/src/commands/serve.rs` | `serve --scenario --protocol json|binary --num-envs --record --seed --max-steps --obs image|flat|dict`. |
| `apps/clankers-app/src/commands/record.rs` | `record --scenario --output --topics --camera --frame-decimation`. |
| `apps/clankers-app/src/commands/replay.rs` | `replay --input --camera --policy --export`. |
| `apps/clankers-app/src/commands/bench.rs` | `bench headless|vec|protocol|render|record|mpc|policy` subtree with `--json`, `--csv`, `--baseline`. |
| `apps/clankers-app/tests/cli_validate.rs` | Integration test: `validate_corrupted_urdf_returns_nonzero_exit`, `validate_good_urdf_returns_zero_exit`. |
| `apps/clankers-app/tests/cli_info_json.rs` | Integration test: `info_json_includes_version_and_scenarios`. |
| `apps/clankers-app/tests/cli_run_scenario.rs` | Integration test: `run_arm_pick_scenario_completes_one_episode`. |
| `apps/clankers-app/tests/cli_inspect.rs` | Integration test: `inspect_urdf_prints_joint_layout`. |
| `apps/clankers-app/tests/cli_serve_protocol.rs` | Integration test: `serve_binary_protocol_round_trips_observation`. |
| `apps/clankers-app/tests/cli_bench.rs` | Integration test: `bench_headless_emits_csv_columns`. |
| `apps/clankers-app/tests/fixtures/corrupted.urdf` | Malformed-XML URDF for the validate smoke test. |
| `apps/clankers-app/tests/fixtures/minimal.urdf` | One-link, one-joint URDF for the good-path validate test. |
| `crates/clankers-sim/tests/scenarios_registry.rs` | Integration test: `arm_pick_scenario_builds_without_panic`, `cartpole_scenario_builds_without_panic`, `registry_lists_builtin_scenarios`. |
| `apps/clankers-app/CHANGELOG.md` | Per-PR `Added` entries (CLI surface evolution). |

### MODIFY

| Path | Lines / sites | Change |
|------|---------------|--------|
| `apps/clankers-app/src/main.rs` | `:10` (use `clap`), `:29–74` (`Commands` enum), `:104` (`run_headless`), `:143` (`run_serve`), `:199` (`run_viz`), `:241` (`run_info`), `:261–282` (`main`) | Extend the `Commands` enum from 4 variants (`Headless`, `Serve`, `Viz`, `Info`) to ~10 (`Info`, `Validate`, `Inspect`, `Run`, `Serve`, `Record`, `Replay`, `Bench`, `Viz`, plus deprecated `Headless` alias). Replace the four monolithic `run_*` functions with `mod commands;` and per-command dispatch in `main`. The existing `run_headless` becomes `commands::run::execute` invoked when no `--scenario` is supplied (legacy default). `run_serve` becomes `commands::serve::execute`. `run_viz` is preserved and gets an optional `--scenario` arg. `run_info` is replaced by `commands::info::execute` with `--json`. |
| `apps/clankers-app/Cargo.toml` | `:10–20` (`[dependencies]`) | Add `clankers-sim.workspace = true` (already present; verify scenarios re-export), `clankers-urdf.workspace = true`, `clankers-record.workspace = true`, `clankers-onnx.workspace = true` (for `inspect onnx`), `serde.workspace = true`, `serde_json.workspace = true` (for `--json`), `csv.workspace = true` (for `bench --csv`), `anyhow.workspace = true` (for command-level error chains), `assert_cmd = "2"` (dev-dep) and `tempfile = "3"` (dev-dep) for the CLI integration tests. |
| `crates/clankers-sim/src/lib.rs` | `:18–19` (`pub mod builder; pub mod stats;`), `:33–34` (re-exports) | Add `pub mod scenarios;` and `pub use scenarios::{ScenarioBuilder, ScenarioConfig, ScenarioRegistry};`. |

### DELETE

| Path | Reason |
|------|--------|
| None in W5. The existing `Headless`/`Serve`/`Viz`/`Info` subcommands are reshaped, not deleted. Example bins under `examples/src/bin/*.rs` are deleted by W8, not W5. |

### Verbatim references

Today's CLI surface in `apps/clankers-app/src/main.rs`:

```rust
// :29–74
enum Commands {
    Headless { episodes, max_steps, seed },
    Serve    { address, joints, max_steps },
    Viz      { joints, max_steps },
    Info,
}
```

is replaced (PR1+PR2+PR3+PR4 cumulatively) by:

```rust
enum Commands {
    Info     { #[arg(long)] json: bool },
    Validate { /* --urdf --scene --policy --recording-schema --strict --json --scenario */ },
    Inspect  { #[command(subcommand)] target: InspectTarget /* Mcap|Onnx|Urdf|Scene */ },
    Run      { /* --scenario --episodes --max-steps --seed --policy --record */ },
    Serve    { /* --scenario --address --num-envs --protocol --record --seed */ },
    Record   { /* --scenario --output --topics --camera --frame-decimation */ },
    Replay   { /* --input --camera --policy --export */ },
    Bench    { #[command(subcommand)] suite: BenchSuite /* Headless|Vec|Protocol|Render|Record|Mpc|Policy */ },
    Viz      { #[arg(long)] scenario: Option<String>, joints: usize, max_steps: u32 },
    #[command(hide = true)]
    Headless { /* deprecated alias for `run --scenario default --episodes N` */ },
}
```

## 5. Checklist items

Each item is atomic, ≤300 LOC, and lands as part of one of the four PRs in
section 9. Ordering within a PR reflects implementation order.

### PR1 — 5a (read-only)

- [ ] Define `ScenarioConfig { seed: Option<u64>, max_steps: u32, headless: bool, record_path: Option<PathBuf> }` and the `ScenarioBuilder` trait in `crates/clankers-sim/src/scenarios/mod.rs` (skeleton only; no builtins yet).
- [ ] Add `ScenarioRegistry` with `register(&mut self, name: &'static str, builder: Box<dyn ScenarioBuilder>)` and `get(&self, name: &str) -> Option<&dyn ScenarioBuilder>`; expose `list_builtin() -> Vec<&'static str>` returning an empty `vec![]` for PR1 (filled in PR2).
- [ ] Re-export the scenarios module from `crates/clankers-sim/src/lib.rs`.
- [ ] Carve `apps/clankers-app/src/commands/` submodule layout: create `mod.rs` + one stub file per planned command (each stub returns `unimplemented!("WS5 PRn")` for non-PR1 commands so the binary keeps compiling).
- [ ] Implement `commands::info::execute(json: bool)`: prints version, enabled cargo features (`env!("CARGO_PKG_FEATURES")`-style), Bevy/Rapier/ORT versions (read from `Cargo.lock` at build time via a generated module or `built` crate), supported protocol versions (from `clankers_gym::protocol::PROTOCOL_VERSION`), build profile, and `ScenarioRegistry::list_builtin()`. `--json` emits a single object with the same keys.
- [ ] Implement `commands::validate::execute(args)`: takes `--urdf <path>`, `--scene <path>`, `--policy <path>`, `--recording-schema <path>`, `--scenario <name>`, `--strict`, `--json`. Returns nonzero exit on any failure. Calls `clankers_urdf::parse(&path)` → `JointLayout::from_robot(&model)` → `motor_coverage::validate(&layout, &scene)` (the W2 helper). When `--json` is set, emits `{"status": "ok"|"error", "errors": [...]}`.
- [ ] Implement `commands::inspect::execute(target, json)` with subcommands `mcap`, `onnx`, `urdf`, `scene`. `inspect urdf <path>` prints `{joints: [...], joint_layout: {hash, count, names_in_order}}`. `inspect mcap <path>` prints topic list + message counts via `clankers_record`. `inspect onnx <path>` prints input/output tensor shapes + dtypes via `clankers_onnx`. `inspect scene <path>` prints parsed `SceneConfig`.
- [ ] Add fixtures: `apps/clankers-app/tests/fixtures/corrupted.urdf` (malformed XML — missing closing tag) and `apps/clankers-app/tests/fixtures/minimal.urdf` (one fixed link, one revolute joint).
- [ ] Add integration tests `validate_corrupted_urdf_returns_nonzero_exit` and `validate_good_urdf_returns_zero_exit` in `apps/clankers-app/tests/cli_validate.rs` using `assert_cmd`.
- [ ] Add integration test `info_json_includes_version_and_scenarios` in `apps/clankers-app/tests/cli_info_json.rs`.
- [ ] Add integration test `inspect_urdf_prints_joint_layout` in `apps/clankers-app/tests/cli_inspect.rs` (asserts `--json | jq '.joint_layout.hash'` returns the same hash twice across two invocations).
- [ ] Wire the new `Commands` variants `Info`, `Validate`, `Inspect` into the `match` in `apps/clankers-app/src/main.rs:264`. The existing `Headless`, `Serve`, `Viz`, `Info` variants remain (PR2/PR3 reshape them).
- [ ] CHANGELOG entry for PR1.

### PR2 — 5b (run + serve, scenarios)

- [ ] Implement `crates/clankers-sim/src/scenarios/arm_pick.rs` (~150 LOC). Mirrors the headless subset of `examples/src/bin/arm_pick_gym.rs` (current 281 lines): URDF load, table, cube, single camera, episode config. NO viz, NO policy hook (those come via `cfg.headless == false` in W8).
- [ ] Implement `crates/clankers-sim/src/scenarios/cartpole.rs` (~80 LOC). Mirrors `examples/src/bin/cartpole_gym.rs` (current 156 lines) headless path.
- [ ] Wire `register_builtin(&mut registry)` to register both scenarios.
- [ ] Add integration tests `arm_pick_scenario_builds_without_panic`, `cartpole_scenario_builds_without_panic`, `registry_lists_builtin_scenarios` in `crates/clankers-sim/tests/scenarios_registry.rs`.
- [ ] Implement `commands::run::execute(args)`: `--scenario <name>` (required), `--episodes N`, `--max-steps N`, `--seed`, `--policy <path>`, `--record <path>`, `--json` (emits per-episode stats as JSONL). Default behaviour without `--scenario` runs the original empty headless app (back-compat with today's `headless` subcommand).
- [ ] Implement `commands::serve::execute(args)`: `--scenario <name>`, `--address`, `--num-envs N`, `--protocol json|binary`, `--record <path>`, `--seed`, `--max-steps`, `--obs image|flat|dict`. For `--protocol binary` consume `clankers_gym::encoding::EncodedObservation` (WS4 deliverable). When WS4 has not landed, `--protocol binary` returns a clear error referencing the WS4 dependency.
- [ ] Add integration test `run_arm_pick_scenario_completes_one_episode` in `apps/clankers-app/tests/cli_run_scenario.rs`.
- [ ] Add integration test `serve_binary_protocol_round_trips_observation` in `apps/clankers-app/tests/cli_serve_protocol.rs` (gated `#[cfg(feature = "ws4-binary")]` if W4 not yet merged).
- [ ] Add `--scenario` flag to `Viz` variant for forward compatibility with W8 (does not change rendering).
- [ ] Wire `Run`, `Serve`, `Viz` (with optional scenario) variants into `main.rs` dispatch.
- [ ] CHANGELOG entry for PR2.

### PR3 — 5b (record + replay)

- [ ] Implement `commands::record::execute(args)`: `--scenario <name>`, `--output <path.mcap>`, `--topics topic1,topic2`, `--camera label`, `--frame-decimation N`, `--metadata`. Wraps `clankers_record::Recorder` and surfaces bounded-channel backpressure (dropped-frame count printed to stderr).
- [ ] Implement `commands::replay::execute(args)`: `--input <path.mcap>`, `--camera label`, `--policy <path.onnx>` (optional, for compare), `--export <dir>` (optional, dumps frames as PNG). When `--policy` is given, computes per-step action L2 between policy and recorded action and prints the summary.
- [ ] Add integration test `record_writes_mcap_file_with_camera_topic` in `apps/clankers-app/tests/cli_record.rs` (small, 5-step capture, asserts file exists and `inspect mcap` lists `/camera/<label>`).
- [ ] Add integration test `replay_iterates_recorded_episode_steps` in `apps/clankers-app/tests/cli_replay.rs`.
- [ ] Wire `Record`, `Replay` variants into `main.rs` dispatch.
- [ ] Mark `Headless` variant `#[command(hide = true)]` and route it through `commands::run::execute` with `scenario = "default"`. Print a one-line deprecation notice on use.
- [ ] CHANGELOG entry for PR3.

### PR4 — 5c (bench)

- [ ] Define `BenchSuite` enum (`Headless`, `Vec`, `Protocol`, `Render`, `Record`, `Mpc`, `Policy`) under `commands::bench`. Each variant carries its own args (e.g. `Vec { envs: Vec<usize>, steps: u32 }`).
- [ ] Implement `commands::bench::execute(suite, json, csv, baseline)`. Output paths: stdout human-readable table by default; `--json` to stdout; `--csv <path>` writes CSV with columns `suite,case,wall_ms,steps_per_sec,p50_ms,p99_ms`; `--baseline <path.json>` compares vs committed baseline and exits 1 on >15% regression.
- [ ] Implement `bench headless` and `bench vec` directly (these call into W7's `ParallelVecEnvRunner` once W7 PR1 has merged; until then, sequential fallback emits warning and runs sequentially).
- [ ] Implement `bench protocol` (round-trips N steps through `clankers_gym::server` + in-process client; measures `obs/action` framing throughput).
- [ ] Stub `bench render`, `bench record`, `bench mpc`, `bench policy` as "not implemented in this PR" exit-2 placeholders (filled in W7 PR4 for `record` and follow-up work for the rest).
- [ ] Commit baseline file `apps/clankers-app/benches/baseline.json` with one entry per implemented suite, recorded on the merging developer's machine and tagged with hostname + commit SHA.
- [ ] Add integration test `bench_headless_emits_csv_columns` in `apps/clankers-app/tests/cli_bench.rs` (runs `bench headless --csv <tmp> --steps 10` and asserts header row).
- [ ] Add integration test `bench_baseline_comparison_fails_on_regression` (constructs a synthetic baseline where the committed value is 100× faster than the test run; asserts exit code 1).
- [ ] Wire `Bench` variant into `main.rs` dispatch.
- [ ] CHANGELOG entry for PR4 noting CSV baseline format and the >15% regression gate.

## 6. Tests required before implementation

Test-first: every test below is authored and committed in the same PR as
the checklist item it covers, **before** the implementation is written.
Each row names file, test, and assertion shape. All Rust integration tests
use `assert_cmd` to invoke the compiled `clankers` binary; Python is not
involved in this workstream.

| Test | Path | PR | Assertion shape |
|------|------|----|-----------------|
| `validate_corrupted_urdf_returns_nonzero_exit` | `apps/clankers-app/tests/cli_validate.rs` | PR1 | `Command::cargo_bin("clankers").args(["validate", "--urdf", "tests/fixtures/corrupted.urdf"]).assert().failure().code(predicate::ne(0))`. The fixture URDF has a missing `</robot>` tag. |
| `validate_good_urdf_returns_zero_exit` | `apps/clankers-app/tests/cli_validate.rs` | PR1 | Same invocation against `tests/fixtures/minimal.urdf` exits 0 and stdout contains `"status": "ok"` (when `--json` supplied) or `Validation OK` otherwise. |
| `info_json_includes_version_and_scenarios` | `apps/clankers-app/tests/cli_info_json.rs` | PR1 | `clankers info --json` parses as JSON via `serde_json::from_slice`, has keys `version`, `protocol_version`, `scenarios` (array, may be empty in PR1, populated in PR2). |
| `inspect_urdf_prints_joint_layout` | `apps/clankers-app/tests/cli_inspect.rs` | PR1 | Run `clankers inspect urdf tests/fixtures/minimal.urdf --json` twice. Parse both outputs. Assert `output1.joint_layout.hash == output2.joint_layout.hash` (deterministic) AND the hash field is a non-empty hex string. |
| `arm_pick_scenario_builds_without_panic` | `crates/clankers-sim/tests/scenarios_registry.rs` | PR2 | `let mut app = App::new(); arm_pick::build(&mut app, &ScenarioConfig::default());` returns without panic. `app.finish(); app.cleanup(); app.update();` runs one frame without panic. |
| `cartpole_scenario_builds_without_panic` | `crates/clankers-sim/tests/scenarios_registry.rs` | PR2 | Same shape as above. |
| `registry_lists_builtin_scenarios` | `crates/clankers-sim/tests/scenarios_registry.rs` | PR2 | `let mut r = ScenarioRegistry::default(); register_builtin(&mut r); assert_eq!(r.list_builtin().sort(), vec!["arm_pick", "cartpole"].sort());`. |
| `run_arm_pick_scenario_completes_one_episode` | `apps/clankers-app/tests/cli_run_scenario.rs` | PR2 | `clankers run --scenario arm_pick --episodes 1 --max-steps 50 --seed 0` exits 0 and stdout JSONL last line has `"episode": 1, "steps": 50` (or fewer if episode auto-terminates). |
| `serve_binary_protocol_round_trips_observation` | `apps/clankers-app/tests/cli_serve_protocol.rs` | PR2 (gated on W4) | Start `clankers serve --scenario cartpole --protocol binary --address 127.0.0.1:0` in a background thread; in-process client sends Init+Reset; assert returned `EncodedObservation::FlatF32` deserialises to the expected dim. Skipped if W4 not merged. |
| `record_writes_mcap_file_with_camera_topic` | `apps/clankers-app/tests/cli_record.rs` | PR3 | `clankers record --scenario arm_pick --output <tmpdir>/run.mcap --camera front --max-steps 5` exits 0; `clankers inspect mcap <tmpdir>/run.mcap --json | jq '.topics[]'` includes `/camera/front`. |
| `replay_iterates_recorded_episode_steps` | `apps/clankers-app/tests/cli_replay.rs` | PR3 | After the record test, `clankers replay --input <tmpdir>/run.mcap --json` emits 5 step lines. |
| `bench_headless_emits_csv_columns` | `apps/clankers-app/tests/cli_bench.rs` | PR4 | `clankers bench headless --csv <tmp>/out.csv --steps 10` exits 0; first line of CSV is `suite,case,wall_ms,steps_per_sec,p50_ms,p99_ms`; data rows ≥ 1. |
| `bench_baseline_comparison_fails_on_regression` | `apps/clankers-app/tests/cli_bench.rs` | PR4 | Write a synthetic `baseline.json` with `steps_per_sec: 1_000_000_000`. Run `clankers bench headless --baseline <tmp>/baseline.json`. Assert exit code 1 and stderr contains `regression`. |

Fixture creation is a PR1 checklist item; the fixture file paths above
are declared as new files in section 4.

## 7. Success criteria

Each criterion is checkable with a concrete command. CLAUDE.md mandates
`-j 24` on every cargo invocation; the project machine has 32 cores and
must leave headroom for the OS.

- `cargo test -j 24 -p clankers-app` exits 0 (all CLI integration tests
  above pass; uses `assert_cmd` which builds the binary as part of test
  setup).
- `cargo test -j 24 -p clankers-sim --test scenarios_registry` exits 0.
- `cargo clippy -j 24 -p clankers-app --all-targets -- -D warnings` exits
  0 (no clippy regression from the new command modules).
- `cargo build -j 24 -p clankers-app --release` succeeds and produces a
  `clankers` binary.
- New-user acceptance sequence (every command exits 0):
  ```
  ./target/release/clankers info
  ./target/release/clankers validate --scenario arm_pick
  ./target/release/clankers run --scenario arm_pick --episodes 1 --max-steps 50 --seed 0
  ```
- `clankers --help` lists at least 10 subcommands (`info`, `validate`,
  `inspect`, `run`, `serve`, `record`, `replay`, `bench`, `viz`, plus the
  hidden `headless` alias).
- `clankers validate --scenario arm_pick --strict --json | jq '.status'`
  prints `"ok"`.
- `clankers inspect urdf tests/fixtures/minimal.urdf --json | jq -r '.joint_layout.hash'`
  returns a non-empty hex string; two invocations return the same string
  (the WS1 determinism guarantee surfaced at the CLI).
- `clankers info --json | jq -r '.scenarios | length'` returns `2` after
  PR2 (`arm_pick` and `cartpole`).
- `clankers bench headless --csv /tmp/clankers-bench.csv --steps 100`
  exits 0 and `head -1 /tmp/clankers-bench.csv` matches
  `suite,case,wall_ms,steps_per_sec,p50_ms,p99_ms`.
- `wc -l apps/clankers-app/src/main.rs` returns ≤120 lines (the four
  monolithic `run_*` functions at lines 104/143/199/241 are gone, replaced
  by per-command modules).
- `grep -n 'unimplemented!' apps/clankers-app/src/commands/` returns zero
  hits after PR4 (no remaining placeholder stubs; the `bench render|mpc|
  policy` "not implemented" arms return a proper exit code, not panic).
- `cargo doc -j 24 -p clankers-app --no-deps` succeeds.

## 8. Risks & mitigations

- **Risk:** 22 example bins still ship alongside CLI scenarios for one
  release window, doubling maintenance surface while users migrate.
  **Mitigation:** W8 is the explicit cleanup workstream and is sequenced
  immediately after W5. W5 only adds new surface; W8 PR1+PR2 lift the
  scene setup into `scenarios/` and shrink each bin to ≤100 lines. The
  CHANGELOG for PR1 explicitly notes the parallel existence and points at
  W8 for migration.

- **Risk:** `serve --protocol binary` depends on W4's `EncodedObservation`.
  If W4 slips past W5, PR2 cannot ship binary mode.
  **Mitigation:** Gate the binary branch behind a Cargo feature
  `ws4-binary` on `clankers-app`. If W4 has not merged at PR2 review time,
  ship PR2 with `--protocol json` working and `--protocol binary` returning
  a clear "not yet available — pending W4" error. The binary mode follows
  in a small PR2b. The `serve_binary_protocol_round_trips_observation`
  test is gated on the same feature flag.

- **Risk:** The `bench` subtree is large (7 subcommands), making PR4 risky
  to review.
  **Mitigation:** PR4 lands `bench headless`, `bench vec`, and
  `bench protocol` fully (these have direct in-process implementations).
  `bench render`, `bench record`, `bench mpc`, `bench policy` ship as
  proper exit-2 "not implemented in this PR" placeholders that emit a
  one-line message and a pointer to the issue tracker — they do NOT
  panic. The four placeholder subcommands are implemented in W7 PR4 and
  follow-up work.

- **Risk:** `scenarios::arm_pick::build` may duplicate logic from
  `examples/src/bin/arm_pick_gym.rs`, leading to drift if the example
  changes before W8 deletes it.
  **Mitigation:** Add a CI doctest (in W8) that constructs an app via
  `scenarios::arm_pick::build` AND via the example bin's
  `setup_scene` function (extracted in W8) and asserts the resulting
  entity count + component shape match. For W5 the new scenario is
  authoritative; the example bin is a thin demo that may diverge until
  W8 deletes it.

- **Risk:** `clankers validate --scenario arm_pick` exercises the WS2
  `validate_motor_coverage` helper, which depends on the `MotorOverrides`
  convention from `MEMORY.md` ("ALL Joints Must Be Overridden"). If a
  scenario forgets a gripper joint override the CLI fails — useful, but
  may surprise contributors who copy-paste from older example bins.
  **Mitigation:** The `validate` error message prints the missing joint
  name and a one-line pointer to the `MEMORY.md` convention plus the
  canonical pattern (`examples/src/bin/arm_ik_viz.rs` ::
  `arm_motor_override_system`). PR1 includes a doc comment on
  `commands::validate::execute` with the same pointers.

- **Risk:** The new `apps/clankers-app/Cargo.toml` dev-dependency on
  `assert_cmd` and `tempfile` pulls in a non-trivial dependency tree
  (cargo metadata graph grows by ~30 crates).
  **Mitigation:** Both crates are already mature/cached in the workspace
  build cache; the cost is one-time on first build and the integration
  tests need a way to invoke the compiled binary deterministically. The
  alternative (in-process `Cli::parse_from(...)` + capturing stdout) is
  more brittle and does not test the actual `main()` entry path.

## 9. PR breakdown

Exactly **4 commits** (matches `LOOPS.yaml` `expected_implementation_prs: 4`).
Each batch is reviewable in isolation.

### PR1 — 5a (read-only CLI: info, validate, inspect)

Touches Rust only. Approximately **600 LOC diff** (300 new commands, 100
test, 100 fixtures + commands wiring, 100 module skeleton in
`clankers-sim/src/scenarios/mod.rs`).

- `crates/clankers-sim/src/scenarios/mod.rs` (skeleton, ~80 LOC).
- `crates/clankers-sim/src/lib.rs` re-export (~3 LOC).
- `apps/clankers-app/src/commands/{mod,info,validate,inspect}.rs` (~350 LOC).
- `apps/clankers-app/src/main.rs` refactor: extract dispatch, add new
  variants (~80 LOC net; original `run_*` functions trimmed).
- `apps/clankers-app/Cargo.toml` dep additions (~10 LOC).
- `apps/clankers-app/tests/fixtures/{corrupted,minimal}.urdf` (~30 LOC).
- `apps/clankers-app/tests/cli_{validate,info_json,inspect}.rs` (~120 LOC).
- `apps/clankers-app/CHANGELOG.md` entry (~10 LOC).

Commit message:
`feat(cli): add info --json, validate, and inspect read-only commands`

### PR2 — 5b run/serve (scenarios + run + serve)

Touches Rust only. Approximately **700 LOC diff**.

- `crates/clankers-sim/src/scenarios/arm_pick.rs` (~150 LOC).
- `crates/clankers-sim/src/scenarios/cartpole.rs` (~80 LOC).
- `crates/clankers-sim/src/scenarios/mod.rs` `register_builtin` wiring (~20 LOC).
- `apps/clankers-app/src/commands/{run,serve}.rs` (~250 LOC).
- `apps/clankers-app/src/main.rs` `Run`, `Serve`, `Viz` (with `--scenario`)
  wiring (~30 LOC).
- `crates/clankers-sim/tests/scenarios_registry.rs` (~80 LOC).
- `apps/clankers-app/tests/cli_run_scenario.rs` (~50 LOC).
- `apps/clankers-app/tests/cli_serve_protocol.rs` (~50 LOC, feature-gated).
- `apps/clankers-app/CHANGELOG.md` entry (~10 LOC).

Commit message:
`feat(cli,sim): add scenarios module with arm_pick+cartpole; add run and serve --scenario`

### PR3 — 5b record/replay

Touches Rust only. Approximately **400 LOC diff**.

- `apps/clankers-app/src/commands/{record,replay}.rs` (~200 LOC).
- `apps/clankers-app/src/main.rs` `Record`, `Replay`, hidden `Headless`
  wiring (~30 LOC).
- `apps/clankers-app/tests/cli_record.rs` (~70 LOC).
- `apps/clankers-app/tests/cli_replay.rs` (~70 LOC).
- `apps/clankers-app/CHANGELOG.md` entry (~10 LOC).

Commit message:
`feat(cli): add record and replay commands; deprecate headless alias`

### PR4 — 5c bench

Touches Rust only. Approximately **600 LOC diff**. Depends on W7 PR1
(for `bench vec`); ships sequential fallback if W7 has not landed.

- `apps/clankers-app/src/commands/bench.rs` (~400 LOC including
  `BenchSuite` enum, CSV writer, baseline comparator).
- `apps/clankers-app/src/main.rs` `Bench` wiring (~10 LOC).
- `apps/clankers-app/benches/baseline.json` (~50 LOC).
- `apps/clankers-app/tests/cli_bench.rs` (~120 LOC).
- `apps/clankers-app/CHANGELOG.md` entry (~20 LOC).

Commit message:
`feat(cli): add bench subcommand tree with --json/--csv and baseline comparison`
