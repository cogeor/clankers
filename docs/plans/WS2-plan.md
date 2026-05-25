# Workstream 2 — Layout-Bound Sensors and Actions

Status: planned. Estimated implementation: 2 PRs.
Source report: `notes/clankers_codebase_quality_report_2026-05-25.md` (finding #1).
Depends on: W1 (`JointLayout` contract). Unblocks: W5 (CLI validation
surfaces motor coverage check), W7 (dense-`Vec` runtime hot path).

## 1. Goal

Bind every joint sensor, action applicator, and motor override to the
ordered `JointLayout` produced by W1, eliminating Bevy query-order
nondeterminism at every call site and turning "all joints overridden"
into a setup-time invariant the CLI can fail on.

## 2. Why this workstream, why this order

W1 introduces `JointLayout` (ordered joint names + entity ids + types +
limits) as the canonical contract for "which joint is index `k`". W2 is
the first consumer that actually replaces the today-implicit ordering
with that contract. The dependency edge is hard: a `JointStateSensor`
that takes `Arc<JointLayout>` cannot exist before the type exists.

- **Upstream blocker.** W1 must merge first. Without `JointLayout`,
  sensors keep iterating Bevy queries in archetype/insertion order and
  the report finding #1 reproduces: "A trained policy can bind output
  index 0 to one joint during training and a different joint during
  replay/deployment" (quality report finding #1).
- **Downstream unblockers.**
  - **W5 — `clankers validate`.** Surfacing
    `RobotGroup::validate_motor_coverage(&layout)` as a CLI failure mode
    requires the validator function to exist on the library side; W2
    delivers that function. The CLI gate in `clankers validate
    --scenario arm_pick --strict` becomes meaningful only once the
    library can answer "is every joint covered?".
  - **W7 — dense-`Vec` runtime.** The performance plan converts
    hot-path `HashMap<Entity, …>` lookups into `Vec<…Runtime>` indexed
    by layout slot. `MotorOverrides::Vec<(JointHandle, params)>` in
    layout order (this workstream) is the prerequisite shape; W7 then
    cache-tiles over it without further redesign.
- **Why now, not later.** Examples already encode the "every joint must
  have an override" rule as a comment in
  `crates/clankers-physics/src/rapier/systems.rs` (the `MotorOverrides`
  doc) and as a project memory entry (see `MEMORY.md`,
  "MotorOverrides — ALL Joints Must Be Overridden"). Leaving this as
  prose costs us "robot flailing wildly" failures every time a new
  example forgets a gripper finger. Encoding it as a validator is a
  small, contained change that pays off across every future scenario.

## 3. Out of scope

- **Schema work.** `JointLayout` itself, `ObservationSchema`,
  `ActionSchema`, `FrameSchema`, `RecorderSchema` — all belong to W1.
  W2 imports `JointLayout` and assumes it is `Clone + Send + Sync +
  Hash + Eq` (W1's contract).
- **New sensors.** No additions to the sensor catalogue. W2 only changes
  the constructor signature and read path of the joint-related sensors
  (`JointStateSensor`, `JointCommandSensor`, `JointTorqueSensor`,
  `RobotJointStateSensor`, `RobotJointCommandSensor`,
  `RobotJointTorqueSensor`). IMU/contact/lidar/raycast/EE sensors are
  untouched.
- **Protocol / transport changes.** Image-on-reset, binary frames,
  `EncodedObservation` enum — all W4 territory. The wire format
  produced by a layout-bound sensor is byte-identical to today's
  output; only the production order is now deterministic.
- **`Observation` / `Action` enum redesign.** Stripping
  `as_slice`/`as_mut_slice`/`into_vec` panics is W3.
- **Example-bin refactor.** W2 updates call sites mechanically to take
  a layout argument. Shrinking the bins to ≤100 lines and lifting
  scenario setup into `clankers-sim/src/scenarios/` is W8.
- **CI tightening.** Re-including `clankers-examples` in
  `cargo test --workspace` is W8.
- **Python.** The Python `JointEncoder` is already deterministic
  (quality report, finding #1 "Why this was not caught"); W2 makes the
  Rust runtime match it. No Python files change in this workstream.

## 4. Files to change

All `path:line` references taken from grep evidence in
`.delegate/work/20260525-185618-workstream-plans/02/PLAN.md`.

### NEW

- `crates/clankers-physics/src/rapier/systems.rs` — new
  `MotorOverrideParams::vec_from_layout(&JointLayout, |&JointName|
  -> Option<Params>)` constructor helper (private to crate, used by
  validator and `From<HashMap>`).
- `crates/clankers-physics/tests/motor_coverage.rs` — new integration
  test file containing `motor_override_missing_joint_is_rejected`.
- `crates/clankers-test-utils/tests/layout_conformance.rs` — new
  integration test file containing
  `same_urdf_same_sensor_vector_order` and
  `action_index_maps_to_layout_entity`.

### MODIFY

#### Sensor crate — `crates/clankers-env/src/sensors.rs`

The `JointStateSensor` struct (currently lines 29–63) replaces
`n_joints: usize` with `layout: Arc<JointLayout>` plus a cached
`Vec<Entity>` snapshot taken at construction. `JointCommandSensor` and
`JointTorqueSensor` follow the same pattern. The
`Robot*` variants additionally retain their `RobotId` filter and verify
that each layout entity carries the matching `RobotId` component.

Read path: instead of `world.query::<&JointState>().iter(world)`, the
new path is `for entity in &self.entities { world.get::<JointState>
(entity) … }`. Missing components fill `NaN` rather than skip silently,
so a misconfigured layout produces a visible obs vector signature.

Tests in the same file that call `JointStateSensor::new(2)` etc.
(unit tests at lines 879, 892, 899, 905, 917, 927, 939, 962, 980,
988, 995, 1001, 1014, 1030, 1037, 1356, 1374, 1386, 1397, 1404, 1408,
1412) are migrated to build a `JointLayout` via a new
`clankers_test_utils::fixtures::layout_for_test_world(world)` helper.

#### Physics crate — `crates/clankers-physics/src/rapier/systems.rs`

`MotorOverrides` (currently lines 50–54) gains a layout-ordered storage:

```text
pub struct MotorOverrides {
    pub joints: Vec<(JointHandle, MotorOverrideParams)>,
    // Kept for one release for `From<HashMap>` migration convenience.
    pub legacy_map: Option<HashMap<Entity, MotorOverrideParams>>,
}
```

A `From<HashMap<Entity, MotorOverrideParams>>` impl converts the legacy
input, sorting by the order in which entities appear in the supplied
layout (sort key is an `Arc<JointLayout>` carried alongside, set by a
new `MotorOverrides::with_layout(layout)` builder). The
`rapier_step_system` reads (currently line 104) switch from
`HashMap::get(&entity)` to a single forward pass over the `Vec`, which
also enables the W7 dense-runtime PR.

#### Sim crate — `crates/clankers-sim/src/builder.rs`

Add `impl RobotGroup` method:

```text
pub fn validate_motor_coverage(
    &self,
    layout: &JointLayout,
    overrides: &MotorOverrides,
) -> Result<(), MissingJoints>;
```

`MissingJoints { layout_joint_names: Vec<String>,
override_joint_count: usize }` reports the diff. The method is
infallible-by-construction when `MotorOverrides::with_layout(layout)`
is used, and acts as a defensive double-check for the
`From<HashMap>` migration path. `SceneBuilder::build()` calls it if a
`MotorOverrides` resource is present and panics with a clear message in
debug builds, returns a `Result` from a new `try_build` in release.

#### Public-API re-export — `crates/clankers-env/src/lib.rs`

Re-export `JointLayout` from `clankers-core` so the layout type is
reachable from the same prelude as the sensors that consume it.

#### Call-site migration (PR2 only)

`JointStateSensor::new` and `RobotJointStateSensor::new` call sites — 13
plain + 3 robot variants:

- `apps/clankers-app/src/main.rs:164`
- `examples/src/arm_setup.rs:187`
- `examples/src/quadruped_setup.rs:451`
- `examples/tests/mpc_walk.rs:417`
- `examples/src/bin/arm_policy_viz.rs:382`
- `examples/src/bin/arm_with_policy.rs:56`
- `examples/src/bin/cartpole_gym.rs:97`
- `examples/src/bin/cartpole_vec_benchmark.rs:73`
- `examples/src/bin/cartpole_policy_viz.rs:286`
- `examples/src/bin/cartpole_vec_gym.rs:70`
- `examples/src/bin/multi_robot_viz.rs:436`
- `examples/src/bin/pendulum_viz.rs:291`
- `examples/src/bin/pendulum_headless.rs:43`
- `examples/src/bin/pendulum_headless.rs:96`
- `examples/src/bin/multi_robot.rs:136`
- `examples/src/bin/multi_robot.rs:146`
- `examples/src/bin/multi_robot.rs:155`

`ActionApplicator` impls — 6 sites where the trait is `impl`'d plus the
trait def:

- `crates/clankers-core/src/traits.rs:89` (trait def: add
  `layout(&self) -> &JointLayout`)
- `apps/clankers-app/src/main.rs:83`
- `examples/src/bin/arm_gym.rs:24`
- `examples/src/bin/arm_pick_gym.rs:39`
- `examples/src/bin/cartpole_gym.rs:28`
- `examples/src/bin/cartpole_vec_benchmark.rs:27`
- `examples/src/bin/cartpole_vec_gym.rs:26`

`MotorOverrides::default()` insertion sites (each gets a
`validate_motor_coverage` call paired with it):

- `examples/src/arm_setup.rs:339`
- `examples/tests/mpc_walk.rs:411`
- `examples/tests/arm_startup.rs:26`
- `examples/src/bin/quadruped_mpc_viz.rs:886`

### DELETE (PR2 only)

- `JointStateSensor::new(n_joints: usize)` (the deprecated
  `usize`-only ctor introduced in PR1).
- `RobotJointStateSensor::new(RobotId, n: usize)` (same).
- `MotorOverrides::legacy_map` field + `From<HashMap>` impl (kept for
  one release; deleted at the end of PR2 once every example migrates).

## 5. Checklist items

Each item is atomic and sized to fit in ≤300 LOC of diff.

- [ ] **PR1-1.** Add `JointStateSensor::new(layout: Arc<JointLayout>)`
      alongside existing `new(n_joints)` and mark the old ctor
      `#[deprecated(since = "0.X", note = "use new(layout)")]`.
      Update the `Sensor::read` impl to walk the cached
      `Vec<Entity>` from layout. Inline unit tests updated.
- [ ] **PR1-2.** Mirror PR1-1 for `JointCommandSensor` and
      `JointTorqueSensor`. Same deprecation pattern.
- [ ] **PR1-3.** Add layout-bound constructors and read paths to
      `RobotJointStateSensor`, `RobotJointCommandSensor`,
      `RobotJointTorqueSensor`. The `RobotId` filter becomes an
      assertion against the layout entity's component.
- [ ] **PR1-4.** Add `MotorOverrides::with_layout(Arc<JointLayout>) ->
      Self` builder + `Vec<(JointHandle, MotorOverrideParams)>`
      storage. Keep `joints: HashMap<…>` legacy field behind
      `legacy_map: Option<…>` for the migration window. Implement
      `From<HashMap<Entity, MotorOverrideParams>>` that produces a
      `MotorOverrides` requiring `with_layout(...)` before use.
- [ ] **PR1-5.** Add `RobotGroup::validate_motor_coverage(&JointLayout,
      &MotorOverrides) -> Result<(), MissingJoints>` in
      `crates/clankers-sim/src/builder.rs`. Add a `MissingJoints`
      error type with `Display` listing the missing joint names.
- [ ] **PR1-6.** Add `ActionApplicator::layout(&self) -> &JointLayout`
      method to the trait at `crates/clankers-core/src/traits.rs:89`.
      Default impl panics with `unimplemented!()` so existing impls
      still compile; the deprecation lint warns to migrate.
- [ ] **PR1-7.** Write the three conformance tests (see section 6).
      They must fail on `main` and pass after the implementing commits.
- [ ] **PR1-8.** Update `crates/clankers-env/src/lib.rs` to re-export
      `JointLayout` from `clankers-core` so the prelude carries it.
- [ ] **PR2-1.** Migrate all 13 plain `JointStateSensor::new` call
      sites and 3 `RobotJointStateSensor::new` call sites to the
      layout-bound ctor.
- [ ] **PR2-2.** Migrate all 6 `ActionApplicator` impls to return a
      real `&JointLayout` from `layout()` (no more `unimplemented!`).
- [ ] **PR2-3.** Add a `validate_motor_coverage` call at each of the 4
      `MotorOverrides::default()` insertion sites in examples and
      example tests. Each call site fails the example startup if any
      joint is missing — per the `MEMORY.md` rule that every arm AND
      gripper joint must be overridden.
- [ ] **PR2-4.** Delete the deprecated `new(n_joints)` ctors on all 6
      joint-sensor types, the `legacy_map` field on `MotorOverrides`,
      and the default `unimplemented!()` on `ActionApplicator::layout`.
- [ ] **PR2-5.** Wire the `clankers validate --scenario <name>
      --strict` path (placeholder until W5 lands the CLI itself: the
      `validate_motor_coverage` symbol is now public and ready to
      consume).

## 6. Tests required before implementation

These tests must be written, committed, and **failing** before the
implementation in section 5 lands. Test-first per `TASK.md`.

### `same_urdf_same_sensor_vector_order`

- **File:** `crates/clankers-test-utils/tests/layout_conformance.rs`
  (new file; declares its creation).
- **Assertion shape:**
  1. Parse `examples/assets/arm.urdf` twice; build a `JointLayout`
     each time.
  2. Build a Bevy `World`, spawn the same arm, register a
     `JointStateSensor::new(layout_a.clone())` and a
     `JointStateSensor::new(layout_b.clone())` from the two layouts.
  3. Step the world 5 times with non-trivial joint motion.
  4. Call `.read()` on each sensor at every step; collect both into
     `Vec<Vec<f32>>`.
  5. Assert `obs_from_a == obs_from_b` (byte-equal vectors) for all 5
     steps, regardless of entity insertion order.

### `action_index_maps_to_layout_entity`

- **File:** `crates/clankers-test-utils/tests/layout_conformance.rs`
  (same file as above).
- **Assertion shape:**
  1. Build a `JointLayout` from a fixture URDF; record the
     entity-name pairs by index.
  2. Construct a continuous `Action` of size `layout.len()` with one
     `1.0` in slot `k` and `0.0` elsewhere.
  3. Run a stub `ActionApplicator` that mutates `JointCommand.value`
     using the new layout-bound indexing.
  4. Assert that the entity whose `JointCommand.value == 1.0` is
     exactly `layout.entities()[k]` — for every `k` in
     `0..layout.len()`.

### `motor_override_missing_joint_is_rejected`

- **File:** `crates/clankers-physics/tests/motor_coverage.rs` (new
  integration test file).
- **Assertion shape:**
  1. Build a `JointLayout` for the 6-DOF arm + 2-DOF gripper (8
     joints total).
  2. Construct a `MotorOverrides` with only 7 entries (one finger
     missing).
  3. Call `RobotGroup::validate_motor_coverage(&layout, &overrides)`
     and assert `Err(MissingJoints { layout_joint_names, .. })` where
     `layout_joint_names` contains the missing finger name verbatim.
  4. Construct a `MotorOverrides` covering all 8 joints, assert
     `Ok(())`.
  5. Snapshot the `Display` message to ensure the error names every
     missing joint by name — operator-facing message quality matters
     for the CLI consumer (W5).

## 7. Success criteria

Every criterion is checkable with a concrete command. All `cargo`
commands use `-j 24` per `CLAUDE.md` ("machine has 32 cores, leave
headroom").

- `cargo test -j 24 -p clankers-env --test layout_conformance` exits 0
  and runs both `same_urdf_same_sensor_vector_order` and
  `action_index_maps_to_layout_entity`.
- `cargo test -j 24 -p clankers-physics --test motor_coverage` exits 0
  and runs `motor_override_missing_joint_is_rejected`.
- `cargo test -j 24 -p clankers-env` passes (all inline unit tests in
  `sensors.rs` migrated to the layout API).
- `cargo test -j 24 -p clankers-sim` passes (covers
  `validate_motor_coverage` happy path).
- `cargo build -j 24 --workspace --all-targets` passes after PR2
  (every example bin compiles against the new sensor API).
- `cargo clippy -j 24 --workspace --all-targets --tests --benches --
  -D warnings -D deprecated` passes after PR2 (zero remaining uses of
  the deprecated `new(usize)` ctors).
- `clankers validate --scenario arm_pick --strict` exits nonzero when
  any joint is missing a motor override. Tested by deleting one entry
  from `examples/src/arm_setup.rs::initial_motor_overrides` in a
  scratch branch and confirming the CLI fails. (Requires W5 PR1 to
  ship the CLI subcommand; W2 ships the library function it calls.)
- `grep -rn 'JointStateSensor::new(' crates/ apps/ examples/` returns
  zero hits where the argument is a bare `usize` literal or `usize`
  binding (all call sites now pass an `Arc<JointLayout>`).
- `grep -rn 'MotorOverrides::default()' crates/ apps/ examples/`
  returns zero hits that are **not** immediately followed (within 10
  lines) by a `validate_motor_coverage` call. CI can enforce this with
  a small shell check or ripgrep negative-lookahead.

## 8. Risks & mitigations

- **Risk: Bevy query iteration order vs layout order may diverge over
  an episode** — entities can be despawned/respawned (e.g. on episode
  reset), and the stored `Vec<Entity>` snapshot from layout could
  become stale.
  - **Mitigation.** Sensors hold `Vec<Entity>` snapshots taken from
    `JointLayout::entities()` at construction. `JointLayout` itself
    is rebuilt during episode reset (W1 ships this with
    `RobotGroup::reset_layout()`), and sensors are recreated with the
    new layout in the same Bevy startup system. A debug-build
    assertion at the top of `Sensor::read` checks that every cached
    entity is still alive (`world.get_entity(e).is_some()`) and panics
    with a layout-rebuild hint otherwise.

- **Risk: `MotorOverrides` HashMap removal breaks user-extension code
  in example bins and downstream consumers** — the public API change
  is large (4 hot example bins, plus any out-of-tree user code we
  cannot grep).
  - **Mitigation.** Keep the `From<HashMap<Entity,
    MotorOverrideParams>>` convenience impl alive for one release
    (PR1 ships it, PR2 deletes it after migration). Document the
    migration in the PR1 changelog entry. Out-of-tree consumers get
    one release window to migrate; in-tree migration is mechanical
    grep-and-replace.

- **Risk: Adding `Arc<JointLayout>` to all sensor constructors creates
  a large API diff across the workspace** — 16+ call sites, several
  in test files, makes PR review heavy.
  - **Mitigation.** Two-PR split. PR1 adds the new API alongside the
    old one with `#[deprecated]`; tests and ctors coexist. PR2 is
    purely mechanical migration + deletion, easy to scan. CI runs
    `cargo clippy -- -D deprecated` only after PR2 lands, so PR1
    doesn't immediately break the build.

- **Risk: `validate_motor_coverage` becomes a `panic!` rather than a
  recoverable error, breaking embedded/long-running scenarios that
  load robots dynamically.**
  - **Mitigation.** `SceneBuilder::build()` keeps the panic for the
    common case (programming error: caller forgot a joint).
    `SceneBuilder::try_build() -> Result<SpawnedScene, BuildError>`
    propagates the `MissingJoints` so the CLI and any future plugin
    loader can degrade gracefully. The `clankers validate` command
    will call `try_build`.

- **Risk: New `RobotId`-filtering tests in `RobotJointStateSensor`
  surface a real bug where two robots' layouts collide** — the W1
  design must namespace layout entities per `RobotId`.
  - **Mitigation.** The conformance tests in section 6 exercise the
    single-robot path; PR2 adds a smoke test
    `two_robots_disjoint_sensor_vectors` (using
    `examples/src/bin/multi_robot.rs:136,146,155` as fixture) before
    deleting the deprecated `new(RobotId, usize)` ctor. If the smoke
    test fails, escalate to W1 to namespace layouts by `RobotId`.

## 9. PR breakdown

Exactly **2 commits** — matches the catalogue's "W2: 2 PRs".

### PR1 — `feat(env,physics,sim): layout-bound sensor and motor APIs (deprecate count-only ctors)`

Scope:

- Adds the new layout-bound constructors and read paths for the 6
  joint sensors. Old `new(usize)` (and `new(RobotId, usize)`) ctors
  remain, marked `#[deprecated]`.
- Adds `MotorOverrides::with_layout`,
  `Vec<(JointHandle, MotorOverrideParams)>` storage, and
  `From<HashMap>` convenience.
- Adds `RobotGroup::validate_motor_coverage` +
  `SceneBuilder::try_build`.
- Adds `ActionApplicator::layout` trait method with a default
  `unimplemented!()` body.
- Adds the three conformance tests
  (`same_urdf_same_sensor_vector_order`,
  `action_index_maps_to_layout_entity`,
  `motor_override_missing_joint_is_rejected`).
- **No call-site changes.** Every existing example continues to work
  against the deprecated API; warnings appear but build passes.
- Diff target: ≤300 LOC per file modified, ~1100 LOC total across
  ~10 files (sensors.rs largest at ~250 LOC).

### PR2 — `refactor(env,physics,examples): migrate sensors and motor overrides to JointLayout`

Scope:

- Mechanically migrates every call site in the lists in section 4
  ("Call-site migration") to the layout-bound API.
- Adds `validate_motor_coverage` calls at each of the 4
  `MotorOverrides::default()` insertion sites.
- Deletes the deprecated `new(usize)` and `new(RobotId, usize)`
  ctors, the `MotorOverrides::legacy_map` field, the
  `From<HashMap>` impl, and the `unimplemented!()` default on
  `ActionApplicator::layout`.
- CI lint added: `cargo clippy --workspace --all-targets --tests
  --benches -- -D warnings -D deprecated` (passes only because no
  deprecated APIs remain).
- Diff target: ~600 LOC, almost entirely mechanical search-and-replace
  in example bins.

After PR2, finding #1 of `notes/clankers_codebase_quality_report_
2026-05-25.md` is closed for the sensor/action surface, and the
`MEMORY.md` "MotorOverrides — ALL Joints Must Be Overridden"
convention is enforced by the type system at every `MotorOverrides`
insertion site, not by comments.
