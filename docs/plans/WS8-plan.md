# Workstream 8 — Examples/library separation + CI tightening + synthetic `action_semantics`

## 1. Goal

Move robot-specific scene, control, and rendering setup out of the 22
`examples/src/bin/*.rs` binaries into reusable `clankers-sim::scenarios`
builders (and `clankers-viz::overlays` for the egui chrome), tighten CI to
compile and lint every target (including `clankers-examples`) and to run
`clankers validate --scenario` for every first-class scenario, and fix the
synthetic compiler's action-semantics contract — quality report findings
#4 and #6 — by declaring an `action_semantics` field on the trace schema
and rejecting `move_relative` when no FK source is available.

## 2. Why this workstream, why this order

This is the final workstream because it consumes everything the prior
seven shipped:

- **Depends on W1** — `JointLayout`, `ObservationSchema`, `ActionSchema`
  + `ActionSemantics` enum are the typed contracts every
  `scenarios::*::build(app, cfg)` returns. Without W1, scenarios fall
  back to ad-hoc `Vec<f32>` agreements between bin and runtime — the
  very thing the report's finding #4 calls out.
- **Depends on W2** — `RobotGroup::validate_motor_coverage(&layout)`
  is what `clankers validate --scenario` actually invokes per scenario.
  Without W2, the new CI step has nothing concrete to assert.
- **Depends on W5** — W5 created `crates/clankers-sim/src/scenarios/`
  and the `clankers run --scenario <name>` consumer; W8 migrates **all**
  22 example bins onto that API and adds the per-scenario CI gate that
  W5's design anticipates.
- **Unblocks nothing** — W8 is the last workstream. Its commit gate is
  what flips the report's overall grade from B- toward A by closing
  findings #4 and #6 and by removing the structural debt
  ("Large example binaries accumulate canonical logic without
  compile/test pressure", report `:158`).

Ordering W8 last is forced by the dependency chain. Running W8 in
parallel with W5 risks the scenario API churning twice; running it
before W2 means the `clankers validate` CI step has no validator to
call.

## 3. Out of scope

The following look related but are explicitly **not** done in W8:

- **No new first-class scenarios.** W5 owns the scenario API surface and
  the canonical set (`arm_pick`, `cartpole`, `pendulum`,
  `quadruped_trot`, `multi_robot`, `domain_rand`). W8 lifts the existing
  bin logic into the W5 module; if a bin needs a scenario W5 did not
  declare (e.g. `arm_ik` standalone), W8 adds it as a private helper,
  not as a `clankers run --scenario` target.
- **No performance work.** Parallel vec-env, binary batch protocol,
  dense `Vec<…Runtime>` hot paths, and the bench CSV gate are all W7
  (`docs/plans/WS7-plan.md`). W8 may not introduce or rename any
  benchmark surface.
- **No protocol or schema changes.** Wire framing, `EncodedObservation`,
  image-on-reset, and `RecorderSchema` are W4 / W6. W8 consumes them
  read-only.
- **No new MPC tuning.** Quadruped MPC gains, gait scheduler, and
  `MotorOverrides` conventions are frozen by `MEMORY.md`; W8 preserves
  them verbatim when lifting `quadruped_mpc*.rs` into
  `scenarios::quadruped_trot`.
- **LOC threshold decision (acknowledged compromise).** The report's
  ideal is ≤100 LOC per example bin. W8 adopts a **two-tier ceiling**:
  - **≤100 LOC for headless bins** (no `bevy_egui`, no
    `bevy_panorbit_camera`).
  - **≤150 LOC for visualisation bins** that own non-trivial egui
    overlays and policy/replay UI.
  - **Documented allowlist** for three exceptional bins that own
    substantial end-to-end UI (`arm_pick_replay.rs` ≤200,
    `quadruped_mpc_viz.rs` ≤200, `arm_ik_viz.rs` ≤150) — each justified
    by name in the xtask allowlist; reviewers may push back on any
    individual entry.

  Rationale: the biggest viz binaries own egui timeline scrubbers,
  policy-overlay rendering, and panorbit camera plumbing that legitimately
  does not belong in a headless `scenarios::*` builder. A scenario
  builder + a `clankers-viz::overlays::*` overlay split lands most viz
  bins at ~150 LOC, with the three exceptions documented above. Picking
  one global threshold either bloats `scenarios` with viz code (if we
  pick 100) or under-constrains headless bins (if we pick 150).

## 4. Files to change

### NEW

| Action | Path | Purpose |
|--------|------|---------|
| NEW | `crates/clankers-sim/src/scenarios/arm_pick.rs` | Shared `build(app, cfg)` for the pick-and-place scene (table, red cube, finger colliders, motor-override seeding). Consumed by `arm_pick_gym.rs`, `arm_pick_record.rs`, `arm_pick_replay.rs`, `arm_policy_viz.rs`, `arm_manipulation.rs`. |
| NEW | `crates/clankers-sim/src/scenarios/arm_ik.rs` | IK-only arm scene (no gripper objects). Consumed by `arm_ik.rs` and `arm_ik_viz.rs`. |
| NEW | `crates/clankers-sim/src/scenarios/arm_bench.rs` | Headless arm scene tuned for `arm_bench.rs` throughput measurements. |
| NEW | `crates/clankers-sim/src/scenarios/quadruped_trot.rs` | MPC-ready quadruped scene with `MpcLoopState` factory; consumed by `quadruped_mpc.rs`, `quadruped_mpc_viz.rs`, `quadruped_mpc_bench.rs`. |
| NEW | `crates/clankers-sim/src/scenarios/cartpole.rs` | Headless cartpole; consumed by `cartpole_gym.rs`, `cartpole_vec_gym.rs`, `cartpole_policy_viz.rs`, `cartpole_vec_benchmark.rs`. |
| NEW | `crates/clankers-sim/src/scenarios/multi_robot.rs` | Multi-robot scene; consumed by `multi_robot.rs`, `multi_robot_viz.rs`. |
| NEW | `crates/clankers-sim/src/scenarios/pendulum.rs` | Pendulum scene; consumed by `pendulum_headless.rs`, `pendulum_viz.rs`. |
| NEW | `crates/clankers-sim/src/scenarios/domain_rand.rs` | Domain-randomised scene; consumed by `domain_rand.rs`. |
| NEW | `crates/clankers-viz/src/overlays/mod.rs` | Module entry. |
| NEW | `crates/clankers-viz/src/overlays/arm_ik_overlay.rs` | Lifted egui UI from `arm_ik_viz.rs`. |
| NEW | `crates/clankers-viz/src/overlays/arm_policy_overlay.rs` | Lifted egui from `arm_policy_viz.rs`. |
| NEW | `crates/clankers-viz/src/overlays/arm_pick_replay_overlay.rs` | Timeline scrubber + frame thumbnails lifted from `arm_pick_replay.rs`. |
| NEW | `crates/clankers-viz/src/overlays/cartpole_policy_overlay.rs` | Lifted from `cartpole_policy_viz.rs`. |
| NEW | `crates/clankers-viz/src/overlays/quadruped_mpc_overlay.rs` | Telemetry + foot-force HUD lifted from `quadruped_mpc_viz.rs`. |
| NEW | `crates/clankers-viz/src/overlays/multi_robot_overlay.rs` | Lifted from `multi_robot_viz.rs`. |
| NEW | `crates/clankers-viz/src/overlays/pendulum_overlay.rs` | Lifted from `pendulum_viz.rs`. |
| NEW | `xtask/src/line_count.rs` | Walks `examples/src/bin/`, enforces the two-tier ceiling + allowlist; run as `cargo xtask check-bin-size`. |
| NEW | `xtask/tests/bin_line_count.rs` | `every_example_bin_under_threshold` integration test. |
| NEW | `crates/clankers-sim/tests/scenario_smoke.rs` | `each_first_class_scenario_builds` — iterates `scenarios::REGISTRY`, calls each `build(...)`, asserts no panic. |
| NEW | `python/clankers_synthetic/action_adapter.py` | `NormalizedPositionAdapter`, `AbsoluteJointPositionAdapter`, `JointVelocityAdapter`, `TorqueAdapter` (the four `ActionSemantics` enum arms from W1). |
| NEW | `python/tests/test_synthetic_action_semantics.py` | Three tests (see section 6). |

### MODIFY

| Action | Path | Lines / sites | Change |
|--------|------|---------------|--------|
| MODIFY | `.github/workflows/ci.yml` | `:44` (`cargo clippy --workspace --all-targets`), `:45` (`cargo test --workspace --exclude clankers-examples`), `:66–70` (Python `--ignore` flags) | Replace exclusion with `cargo check --workspace --all-targets -j 24` then `cargo test --workspace -j 24`. Extend the clippy line to `cargo clippy --workspace --all-targets --tests --benches -j 24 -- -D warnings`. Drop the two remaining `--ignore` flags (`test_dl_pipeline_e2e.py`, `test_trajectory_dataset.py`) and gate slow tests behind a pytest `slow` mark instead. Add new step `cargo xtask check-bin-size -j 24`. Add a per-scenario matrix step `cargo run -p clankers-app -- validate --scenario ${{ matrix.scenario }} --strict`. |
| MODIFY | `examples/src/bin/arm_ik.rs` | full file (95 LOC today) | Reduce to ≤100 LOC: build via `scenarios::arm_ik::build(...)`, keep only argument parsing + main loop. |
| MODIFY | `examples/src/bin/arm_gym.rs` | full file (116 LOC) | Reduce to ≤100 LOC. |
| MODIFY | `examples/src/bin/arm_with_policy.rs` | full file (142 LOC) | Reduce to ≤100 LOC. |
| MODIFY | `examples/src/bin/arm_bench.rs` | full file (170 LOC) | Reduce to ≤100 LOC using `scenarios::arm_bench`. |
| MODIFY | `examples/src/bin/arm_manipulation.rs` | full file (291 LOC) | Reduce to ≤100 LOC. |
| MODIFY | `examples/src/bin/arm_pick_gym.rs` | full file (281 LOC; see `:39–80` `PickApplicator`, `:130–195` scene setup) | Reduce to ≤100 LOC via `scenarios::arm_pick::build(...)`; `PickApplicator` moves to the scenario module. |
| MODIFY | `examples/src/bin/arm_pick_record.rs` | full file (526 LOC) | Reduce to ≤150 LOC. |
| MODIFY | `examples/src/bin/arm_pick_replay.rs` | full file (876 LOC) | Reduce to ≤200 LOC (documented exception). |
| MODIFY | `examples/src/bin/arm_policy_viz.rs` | full file (430 LOC) | Reduce to ≤150 LOC via `scenarios::arm_pick` + `overlays::arm_policy_overlay`. |
| MODIFY | `examples/src/bin/arm_ik_viz.rs` | full file (646 LOC) | Reduce to ≤150 LOC (documented exception). |
| MODIFY | `examples/src/bin/cartpole_gym.rs` | full file (156 LOC) | Reduce to ≤100 LOC. |
| MODIFY | `examples/src/bin/cartpole_vec_gym.rs` | full file (147 LOC) | Reduce to ≤100 LOC. |
| MODIFY | `examples/src/bin/cartpole_vec_benchmark.rs` | full file (174 LOC) | Reduce to ≤100 LOC. |
| MODIFY | `examples/src/bin/cartpole_policy_viz.rs` | full file (340 LOC) | Reduce to ≤150 LOC via `overlays::cartpole_policy_overlay`. |
| MODIFY | `examples/src/bin/quadruped_mpc.rs` | full file (299 LOC; see `:25` `main`, `:54–68` `MpcLoopState` builder) | Reduce to ≤100 LOC via `scenarios::quadruped_trot`. |
| MODIFY | `examples/src/bin/quadruped_mpc_viz.rs` | full file (965 LOC) | Reduce to ≤200 LOC (documented exception) via `scenarios::quadruped_trot` + `overlays::quadruped_mpc_overlay`. |
| MODIFY | `examples/src/bin/quadruped_mpc_bench.rs` | full file (491 LOC) | Reduce to ≤150 LOC. |
| MODIFY | `examples/src/bin/multi_robot.rs` | full file (188 LOC) | Reduce to ≤100 LOC. |
| MODIFY | `examples/src/bin/multi_robot_viz.rs` | full file (489 LOC) | Reduce to ≤150 LOC. |
| MODIFY | `examples/src/bin/pendulum_headless.rs` | full file (113 LOC) | Reduce to ≤100 LOC. |
| MODIFY | `examples/src/bin/pendulum_viz.rs` | full file (393 LOC) | Reduce to ≤150 LOC. |
| MODIFY | `examples/src/bin/domain_rand.rs` | full file (229 LOC) | Reduce to ≤100 LOC. |
| MODIFY | `crates/clankers-sim/src/lib.rs` | `pub mod scenarios` re-export | Surface the new submodules and `scenarios::REGISTRY` static (`&[(name, build_fn)]`) consumed by the smoke test and by `clankers run --scenario`. |
| MODIFY | `crates/clankers-viz/src/lib.rs` | top of file | `pub mod overlays;` re-export. |
| MODIFY | `python/clankers_synthetic/compiler.py` | `:68` (`normalize_action`), `:133` (`elif skill.name == "move_relative"`), `:250–267` (`_exec_move_relative`) | Route `normalize_action` through a per-instance `action_adapter` dispatcher selected from the negotiated `ActionSemantics` (default `NormalizedPositionAdapter`). `_exec_move_relative` raises `MoveRelativeWithoutFkError` when neither the env exposes a `forward_kinematics` callable nor the last `info` dict contains an `end_effector` body pose; the old behaviour (delta from origin) is reachable only via the explicit `--legacy-move-relative` opt-in surfaced by the compiler CLI. |
| MODIFY | `python/clankers_synthetic/specs.py` | `class ExecutionTrace` and `class TraceStep` definitions | Add required field `action_semantics: Literal["NormalizedPosition", "AbsoluteJointPosition", "JointVelocity", "Torque"]`. Pydantic `ValidationError` on absence (no default). |
| MODIFY | `python/clankers_synthetic/scripts/run_arm_pick.py` (and siblings) | call-site of `SkillCompiler(...)` | Pass `action_semantics="NormalizedPosition"` (or the negotiated value from the gym handshake) to the compiler constructor. |

### DELETE

| Action | Path | Reason |
|--------|------|--------|
| DELETE | `examples/src/arm_setup.rs` (after migration) | Contents subsumed by `crates/clankers-sim/src/scenarios/arm_pick.rs` and `arm_ik.rs`. Re-exported as deprecated alias for one release before final delete in a follow-up. |
| DELETE | `examples/src/quadruped_setup.rs` (after migration) | Same — contents move into `scenarios::quadruped_trot`. Deprecated alias kept for one release. |
| DELETE | `examples/src/mpc_control.rs` (after migration) | `MpcLoopState` + helpers move into `scenarios::quadruped_trot::mpc`. Deprecated alias kept for one release. |

### Verbatim quotes

The CI lines that change (`.github/workflows/ci.yml`):

```yaml
# :44 today
- run: cargo clippy --workspace --all-targets -- -D warnings
# :45 today
- run: cargo test -j 24 --workspace --exclude clankers-examples
# :66-70 today
python -m pytest tests/ -x -q --timeout=60 \
  --ignore=tests/test_dl_pipeline_e2e.py \
  --ignore=tests/test_trajectory_dataset.py \
  --ignore=tests/test_mcap_loader.py \
  --ignore=tests/test_mcap_augmentor.py
```

The synthetic compiler sites that change (`python/clankers_synthetic/compiler.py`):

```python
# :68 today
def normalize_action(self, joint_targets: np.ndarray) -> np.ndarray:
    """Convert absolute joint positions to normalized [-1, 1] actions."""
    return (joint_targets - self._joint_centers) / self._joint_half_ranges

# :250-267 today
def _exec_move_relative(self, skill, env, obs, joints, all_steps):
    """Execute move_relative: move EE by delta in world frame."""
    delta = np.array(skill.params.get("delta", [0, 0, 0]), dtype=float)
    ...
    if self.ik_solver is not None:
        target = np.zeros(3) + delta            # <— bug: from origin, not current EE
        ik_result = self.ik_solver.solve(target, joints)
        target_joints = ik_result.joint_angles
```

## 5. Checklist items

Each item is atomic, ≤300 LOC, and lands as part of one of the three PRs
in section 9.

PR1 — arm scenarios (lift arm-family bins):

- [ ] Create `crates/clankers-sim/src/scenarios/arm_ik.rs` with
      `pub fn build(app: &mut App, cfg: ArmIkConfig)`; export `ArmIkConfig`.
- [ ] Shrink `examples/src/bin/arm_ik.rs` from 95 → ≤100 LOC; bin wraps
      `scenarios::arm_ik::build`.
- [ ] Create `crates/clankers-sim/src/scenarios/arm_pick.rs`; move
      `PickApplicator` (currently `examples/src/bin/arm_pick_gym.rs:39–80`)
      and the table/cube/finger-collider setup (currently `:130–195`)
      into the module.
- [ ] Shrink `examples/src/bin/arm_pick_gym.rs` from 281 → ≤100 LOC.
- [ ] Shrink `examples/src/bin/arm_pick_record.rs` from 526 → ≤150 LOC;
      record-side logic delegated to `clankers-record` helper.
- [ ] Shrink `examples/src/bin/arm_pick_replay.rs` from 876 → ≤200 LOC
      (documented allowlist exception); the timeline scrubber UI moves
      to `crates/clankers-viz/src/overlays/arm_pick_replay_overlay.rs`.
- [ ] Create `crates/clankers-viz/src/overlays/arm_ik_overlay.rs`;
      shrink `examples/src/bin/arm_ik_viz.rs` from 646 → ≤150 LOC.
- [ ] Create `crates/clankers-viz/src/overlays/arm_policy_overlay.rs`;
      shrink `examples/src/bin/arm_policy_viz.rs` from 430 → ≤150 LOC.
- [ ] Shrink `examples/src/bin/arm_gym.rs` from 116 → ≤100 LOC.
- [ ] Shrink `examples/src/bin/arm_with_policy.rs` from 142 → ≤100 LOC.
- [ ] Shrink `examples/src/bin/arm_manipulation.rs` from 291 → ≤100 LOC.
- [ ] Create `crates/clankers-sim/src/scenarios/arm_bench.rs`; shrink
      `examples/src/bin/arm_bench.rs` from 170 → ≤100 LOC.
- [ ] Add `crates/clankers-sim/tests/scenario_smoke.rs` with
      `each_arm_scenario_builds` covering `arm_ik`, `arm_pick`,
      `arm_bench`.
- [ ] Add `xtask/src/line_count.rs` + `xtask/tests/bin_line_count.rs`
      with the two-tier ceiling and the three-name allowlist.

PR2 — quadruped + cartpole + multi-robot + pendulum + domain-rand
scenarios:

- [ ] Create `crates/clankers-sim/src/scenarios/quadruped_trot.rs`;
      lift the `MpcLoopState` builder (currently
      `examples/src/bin/quadruped_mpc.rs:54–68`) and the foot-contact
      detection loop (`:80–198`) into the scenario.
- [ ] Shrink `examples/src/bin/quadruped_mpc.rs` from 299 → ≤100 LOC.
- [ ] Create `crates/clankers-viz/src/overlays/quadruped_mpc_overlay.rs`
      (foot-force HUD + telemetry); shrink
      `examples/src/bin/quadruped_mpc_viz.rs` from 965 → ≤200 LOC
      (documented allowlist exception — biggest single lift).
- [ ] Shrink `examples/src/bin/quadruped_mpc_bench.rs` from 491 → ≤150
      LOC.
- [ ] Create `crates/clankers-sim/src/scenarios/cartpole.rs`; shrink
      `cartpole_gym.rs` 156 → ≤100, `cartpole_vec_gym.rs` 147 → ≤100,
      `cartpole_vec_benchmark.rs` 174 → ≤100.
- [ ] Create `crates/clankers-viz/src/overlays/cartpole_policy_overlay.rs`;
      shrink `examples/src/bin/cartpole_policy_viz.rs` from 340 → ≤150
      LOC.
- [ ] Create `crates/clankers-sim/src/scenarios/multi_robot.rs`; shrink
      `multi_robot.rs` 188 → ≤100.
- [ ] Create `crates/clankers-viz/src/overlays/multi_robot_overlay.rs`;
      shrink `multi_robot_viz.rs` 489 → ≤150 LOC.
- [ ] Create `crates/clankers-sim/src/scenarios/pendulum.rs`; shrink
      `pendulum_headless.rs` 113 → ≤100.
- [ ] Create `crates/clankers-viz/src/overlays/pendulum_overlay.rs`;
      shrink `pendulum_viz.rs` 393 → ≤150.
- [ ] Create `crates/clankers-sim/src/scenarios/domain_rand.rs`; shrink
      `examples/src/bin/domain_rand.rs` 229 → ≤100.
- [ ] Extend `crates/clankers-sim/tests/scenario_smoke.rs` —
      `each_first_class_scenario_builds` iterates the full
      `scenarios::REGISTRY` (now 8+ entries).
- [ ] Tighten `xtask/tests/bin_line_count.rs` to expect all 22 bins
      under their tier ceiling (or in the allowlist).

PR3 — CI tightening + synthetic compiler `action_semantics`:

- [ ] Edit `.github/workflows/ci.yml:45`: replace
      `cargo test -j 24 --workspace --exclude clankers-examples` with
      `cargo check --workspace --all-targets -j 24` followed by
      `cargo test --workspace -j 24`.
- [ ] Edit `.github/workflows/ci.yml:44`: replace
      `cargo clippy --workspace --all-targets -- -D warnings` with
      `cargo clippy --workspace --all-targets --tests --benches -j 24
      -- -D warnings`.
- [ ] Add a new CI step `cargo xtask check-bin-size -j 24` after the
      clippy step; fails CI if any bin breaches its tier ceiling and
      is not in the allowlist.
- [ ] Add a per-scenario matrix step that runs
      `cargo run --release -p clankers-app -- validate --scenario
      ${{ matrix.scenario }} --strict` for every name in
      `scenarios::REGISTRY` (initially: `arm_pick`, `arm_ik`,
      `cartpole`, `quadruped_trot`, `multi_robot`, `pendulum`,
      `domain_rand`).
- [ ] Edit `.github/workflows/ci.yml:66–70`: drop
      `--ignore=tests/test_dl_pipeline_e2e.py` and
      `--ignore=tests/test_trajectory_dataset.py`; mark those tests
      with `@pytest.mark.slow` and add a second CI step
      `pytest -m 'not slow'` for PRs plus a nightly job
      `pytest -m slow`. The two `test_mcap_*` ignores stay in W8
      because W6 owns un-ignoring them.
- [ ] Add `python/clankers_synthetic/action_adapter.py` with the four
      adapter classes; each exposes `to_env_action(joint_targets) ->
      np.ndarray` and `from_env_action(action) -> np.ndarray`.
- [ ] Add `action_semantics` field to `ExecutionTrace` and `TraceStep`
      in `python/clankers_synthetic/specs.py` — required (no default),
      typed as `Literal["NormalizedPosition", "AbsoluteJointPosition",
      "JointVelocity", "Torque"]`.
- [ ] Modify `python/clankers_synthetic/compiler.py:68` to dispatch
      through `self._adapter.to_env_action(...)` instead of returning a
      bare normalised vector.
- [ ] Modify `python/clankers_synthetic/compiler.py:250` `_exec_move_relative`
      to raise `MoveRelativeWithoutFkError` when no FK source is
      available; honour `--legacy-move-relative` opt-in flag.
- [ ] Add `MoveRelativeWithoutFkError` to a new
      `python/clankers_synthetic/errors.py`.
- [ ] Add `python/tests/test_synthetic_action_semantics.py` with the
      three tests named in section 6.

## 6. Tests required before implementation

The tests below are authored and committed **before** the implementation
checklist items they cover. All four file paths follow `TASK.md`'s test
placement conventions.

| Test | Path | Assertion shape |
|------|------|-----------------|
| `every_example_bin_under_threshold` | `xtask/tests/bin_line_count.rs` | For each `examples/src/bin/*.rs`, count lines (ignoring trailing whitespace). Assert: if filename ends `_viz.rs` → `lines <= 150`; otherwise `lines <= 100`. Allowlist (constant `const ALLOWED_EXCEPTIONS: &[(&str, usize)]`) raises the ceiling for `arm_pick_replay.rs` (200), `quadruped_mpc_viz.rs` (200), `arm_ik_viz.rs` (150). Any bin not in the allowlist that exceeds its tier ceiling fails the test. Test prints the offending bin's line count for easy diagnosis. |
| `each_first_class_scenario_builds` | `crates/clankers-sim/tests/scenario_smoke.rs` | Iterate `scenarios::REGISTRY` (a `pub const &[(&str, fn(&mut App, ScenarioConfig))]`). For each entry, construct a fresh `App`, call `build(&mut app, ScenarioConfig::default())`, advance the app for 10 frames, assert no panic and `app.world().resource::<EpisodeStats>().total_steps >= 10`. |
| `trace_without_action_semantics_is_rejected` | `python/tests/test_synthetic_action_semantics.py` | Construct a JSON dict for an `ExecutionTrace` that omits the `action_semantics` field. Call `ExecutionTrace.model_validate(d)`. Assert `pydantic.ValidationError` raised, message contains `"action_semantics"`. |
| `move_relative_without_fk_is_rejected` | `python/tests/test_synthetic_action_semantics.py` | Construct a `SkillCompiler` with `ik_solver=None` and a mock env that returns an `info` dict without an `"end_effector"` body pose. Compile a one-skill `CanonicalPlan` whose only skill is `move_relative({"delta": [0.05, 0, 0]})`. Assert `MoveRelativeWithoutFkError` raised. |
| `normalized_position_adapter_clamps_to_range` | `python/tests/test_synthetic_action_semantics.py` | Construct `NormalizedPositionAdapter(joint_centers=[0.0]*6, joint_half_ranges=[1.0]*6)`; call `adapter.to_env_action(np.array([2.0]*6))`. Assert result `np.allclose(out, np.ones(6))` (clamped to `[-1, 1]`). Also assert `adapter.semantics == "NormalizedPosition"`. |

Fixture creation: none new. The xtask test reads the workspace's
`examples/src/bin/` directory directly; the Python tests use in-memory
mock envs.

## 7. Success criteria

Each criterion is checkable with a concrete command. CLAUDE.md mandates
`-j 24` for every cargo invocation:

- `cargo check --workspace --all-targets -j 24` exits 0 (i.e.
  `clankers-examples` compiles in CI again — closes report finding #6
  for the Rust side).
- `cargo test --workspace -j 24` exits 0 with `clankers-examples`
  included (no `--exclude` flag on the command line).
- `cargo clippy --workspace --all-targets --tests --benches -j 24 --
  -D warnings` exits 0.
- `cargo xtask check-bin-size -j 24` exits 0. Equivalent shell sanity
  check: `awk 'NR==FNR{ok[$2]=$1; next} {l=ok[FILENAME]; if(l>200)
  print FILENAME, l}' <(wc -l examples/src/bin/*.rs)` shows nothing
  outside the allowlist.
- `cargo run --release -p clankers-app -- validate --scenario arm_pick
  --strict` exits 0; repeat for `arm_ik`, `cartpole`,
  `quadruped_trot`, `multi_robot`, `pendulum`, `domain_rand` — all
  exit 0.
- `cargo test -j 24 -p clankers-sim --test scenario_smoke` reports
  every scenario in `REGISTRY` building successfully.
- `pytest -q python/tests/test_synthetic_action_semantics.py` exits 0
  with three tests passing.
- `pytest -q python/tests/ -m 'not slow'` exits 0 with **no
  `--ignore` flags on the command line** (the two W8-owned ignores
  removed; the two W6-owned `test_mcap_*` ignores tracked separately).
- `grep -n 'normalize_action' python/clankers_synthetic/compiler.py`
  shows the method exists but its body delegates to
  `self._adapter.to_env_action(...)`.

## 8. Risks & mitigations

- **Risk:** LOC-threshold debate. Reviewers may push back on either
  the strict ≤100 (finding the viz exceptions ugly) or the pragmatic
  ≤150 (finding it permissive).
  **Mitigation:** Section 3 documents the two-tier choice and the
  three-name allowlist verbatim. The xtask test encodes the same
  thresholds — a reviewer who wants to change them edits one constant
  in `xtask/src/line_count.rs` and one entry in section 3. No bin
  reaches its ceiling by accident: every bin in the allowlist is
  named.

- **Risk:** Mass example refactor risk — touching 22 binaries in
  parallel makes review hard, and a single regression in scene setup
  (motor-override seeding, finger-collider geometry,
  `num_solver_iterations = 50` per MEMORY.md) silently destabilises a
  long-running benchmark.
  **Mitigation:** PR1 (arm family) and PR2 (everything else) split by
  robot kind, letting reviewers focus per family. Scenarios are pure
  builders, so `git diff` between the original bin and the bin +
  scenario shows the move is mechanical. The
  `each_first_class_scenario_builds` smoke test catches setup-time
  panics; the `quadruped_mpc_bench` headless rerun (already in CI via
  `cargo test --workspace`) catches MPC tuning drift. MEMORY.md's
  `num_solver_iterations = 50` and the `MotorOverrides` "every joint
  including grippers" convention are preserved verbatim — the
  scenario builders embed them as constants and the
  `validate_motor_coverage` check (W2) asserts no joint is missed.

- **Risk:** CI runtime regression. Adding `cargo check --all-targets`,
  clippy `--tests --benches`, the xtask line-count step, and a
  per-scenario validate matrix could push the `check` job from ~10
  minutes to ~25.
  **Mitigation:** Parallelise the per-scenario validate as a GitHub
  Actions matrix (each scenario gets its own runner); cache Bevy +
  Rapier compilation via the existing `Swatinem/rust-cache@v2` step;
  the xtask line-count step is a `wc`-equivalent and adds <1 second.
  Track CI wall-clock in the first PR3 commit; if total time
  regresses >50%, split the workflow into `check`, `examples-bins`,
  `scenario-validate` jobs (parallel by default in Actions).

- **Risk:** Synthetic `move_relative` rejection breaks existing
  traces and tutorial notebooks that rely on the
  "delta-from-origin" misbehaviour.
  **Mitigation:** Two-stage rollout. PR3 ships a deprecation warning
  (`DeprecationWarning: move_relative without FK source — silently
  treated as delta from world origin; this will become an error in
  v0.X+1`); the `--legacy-move-relative` CLI flag keeps the old
  behaviour available. A follow-up workstream (post-W8) flips the
  flag default and removes the warning. The release notes for PR3
  list every notebook/script that uses `move_relative` against a
  no-FK env so users have a concrete migration list.

- **Risk:** W5's `clankers-sim::scenarios` API may need extension to
  accommodate quadruped/multi-robot complexity (e.g. scenarios that
  want to return `MpcLoopState` alongside the `App`, or scenarios
  that need to register multiple robots). W8 cannot rewrite the W5
  API mid-stream.
  **Mitigation:** PR1 lands first and exercises the simplest case
  (arm scenarios). Any required extension to `ScenarioConfig`
  surfaces during PR1 review; if the API needs to change, the change
  lands as a W5 follow-up commit (not under W8). PR2 starts only
  after the API shape is settled. This keeps W8 scoped to
  consumption, not redesign.

## 9. PR breakdown

Exactly **3 commits** (matches `LOOPS.yaml` `expected_implementation_prs: 3`).

### PR1 — Arm scenarios (lift arm-family bins; xtask LOC ceiling)

Approximately **1500 LOC diff** (mostly moves and deletes; net negative).

- New `crates/clankers-sim/src/scenarios/{arm_ik,arm_pick,arm_bench}.rs` (~600 LOC,
  most lifted verbatim from existing `examples/src/arm_setup.rs` and
  bin scene-setup blocks).
- New `crates/clankers-viz/src/overlays/{arm_ik_overlay,arm_policy_overlay,arm_pick_replay_overlay}.rs` (~400 LOC, lifted egui).
- Shrink 10 arm bins: `arm_ik.rs`, `arm_gym.rs`, `arm_pick_gym.rs`,
  `arm_pick_record.rs`, `arm_pick_replay.rs`, `arm_policy_viz.rs`,
  `arm_with_policy.rs`, `arm_manipulation.rs`, `arm_bench.rs`,
  `arm_ik_viz.rs` (~−2500 LOC net).
- New `xtask/src/line_count.rs` (~80 LOC) + `xtask/tests/bin_line_count.rs` (~60 LOC).
- New `crates/clankers-sim/tests/scenario_smoke.rs` arm portion (~40 LOC).
- Deprecation alias `pub use` in `examples/src/arm_setup.rs` (~5 LOC).

Commit message:
`refactor(examples): lift arm-family scenes into clankers-sim::scenarios + enforce LOC ceiling`

### PR2 — Quadruped/cartpole/multi-robot/pendulum/domain-rand scenarios

Approximately **2000 LOC diff** (the quadruped lift alone is ~900 LOC).

- New `crates/clankers-sim/src/scenarios/{quadruped_trot,cartpole,multi_robot,pendulum,domain_rand}.rs` (~900 LOC).
- New `crates/clankers-viz/src/overlays/{quadruped_mpc_overlay,cartpole_policy_overlay,multi_robot_overlay,pendulum_overlay}.rs` (~600 LOC).
- Shrink 12 bins: `quadruped_mpc.rs`, `quadruped_mpc_viz.rs`,
  `quadruped_mpc_bench.rs`, `cartpole_gym.rs`, `cartpole_vec_gym.rs`,
  `cartpole_policy_viz.rs`, `cartpole_vec_benchmark.rs`,
  `multi_robot.rs`, `multi_robot_viz.rs`, `pendulum_headless.rs`,
  `pendulum_viz.rs`, `domain_rand.rs` (~−2800 LOC net).
- Extend `scenario_smoke.rs` to cover all `REGISTRY` entries (~30 LOC).
- Tighten `bin_line_count.rs` to the full 22-bin set (~5 LOC).
- Deprecation aliases in `examples/src/{quadruped_setup,mpc_control}.rs`
  (~10 LOC).

Commit message:
`refactor(examples): lift quadruped/cartpole/multi-robot/pendulum scenes into clankers-sim::scenarios`

### PR3 — CI tightening + synthetic `action_semantics` adapter

Approximately **400 LOC diff** (CI + Python schema + new adapter module).

- `.github/workflows/ci.yml`: rewrite the Rust `check` job to drop
  `--exclude clankers-examples`, add `cargo check --workspace
  --all-targets`, extend clippy to `--tests --benches`, add the xtask
  bin-size step and the per-scenario validate matrix; rewrite the
  Python job to drop the two W8-owned `--ignore` flags and add a
  `pytest -m slow` nightly job (~80 LOC YAML).
- `python/clankers_synthetic/action_adapter.py` new (~150 LOC: four
  adapter classes + dispatch helper).
- `python/clankers_synthetic/errors.py` new (~20 LOC:
  `MoveRelativeWithoutFkError`).
- `python/clankers_synthetic/specs.py`: add `action_semantics` field
  to `ExecutionTrace` and `TraceStep` (~10 LOC).
- `python/clankers_synthetic/compiler.py:68,250` modifications + new
  `_adapter` attribute and constructor arg (~50 LOC).
- Updated `python/clankers_synthetic/scripts/run_*.py` call sites
  passing `action_semantics=` (~20 LOC).
- `python/tests/test_synthetic_action_semantics.py` new (~80 LOC: the
  three tests from section 6).

Commit message:
`feat(ci,synthetic): include clankers-examples in CI; declare action_semantics in synthetic traces`
