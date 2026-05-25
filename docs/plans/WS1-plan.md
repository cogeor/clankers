# WS1 — `clankers-core` Contracts

Workstream 1 of 8. Source: Workstream Catalogue in
`.delegate/work/20260525-185618-workstream-plans/TASK.md`.
Authoritative gap reference:
`notes/clankers_codebase_quality_report_2026-05-25.md` (finding #1).

## 1. Goal

Introduce four typed, versioned, serialisable contracts in `clankers-core` —
`JointLayout`, `ObservationSchema`, `ActionSchema` (with `ActionSemantics`),
and `FrameSchema` / `RecorderSchema` — so every downstream system in the
workspace consumes a single source of truth for joint ordering, observation
slots, action encoding, and recorder channels.

## 2. Why this workstream, why this order

W1 is the foundation. Every other workstream consumes at least one of the
four contracts:

- W2 (layout-bound sensors and actions) needs `JointLayout` to replace
  `JointStateSensor::new(n_joints)` with `JointStateSensor::new(layout)`.
- W3 (typed Action/Observation views) needs `ActionSchema` /
  `ObservationSchema` to define the typed variants that the new fallible
  helpers return.
- W4 (protocol parity) negotiates `ObservationSchema` between Rust server
  and Python client so reset and step shapes can be validated.
- W6 (MCAP parity) consumes `RecorderSchema` for topic constants and
  channel discovery.
- W7 (performance: binary protocol, dense `Vec<…Runtime>`, async recorder)
  consumes all four to compile setup-time hashmaps into hot-path vectors.
- W5 (`clankers validate`) and W8 (CI tightening) surface validation errors
  raised by the new types.

The canonical quality report
(`notes/clankers_codebase_quality_report_2026-05-25.md`) **finding #1 —
"Deterministic Joint Layout Is Promised But Not Guaranteed"** lists the four
runtime locations that today depend on `HashMap` iteration or Bevy query
order: `crates/clankers-urdf/src/types.rs`, `crates/clankers-urdf/src/spawner.rs`,
`crates/clankers-sim/src/builder.rs`, and `crates/clankers-env/src/sensors.rs`.
This workstream replaces the implicit alphabetic-sorting promise with an
explicit, hashable `JointLayout` type that can be checked across processes
and platforms.

W1 must land before W2–W7 because each of them migrates call sites to the
new types; without the types, those workstreams have nothing to migrate to.

## 3. Out of scope

The following look related but are deliberately deferred:

- **Sensor refactor** — `JointStateSensor::new(layout)` migration, all
  `RobotJoint*Sensor` call sites, and `MotorOverrides` conversion to
  ordered `Vec<…>` are W2.
- **Panicking `Action`/`Observation` helpers** — replacing `as_slice` /
  `as_mut_slice` / `into_vec` with `as_continuous()` /
  `try_into_continuous()` and adding `ObservationView<'a>` is W3.
- **Protocol changes** — promoting `ObsEncoding` to `EncodedObservation`
  and fixing the `protocol.rs` doc/impl mismatch is W4. W1 only ships the
  types; the negotiation handshake is W4.
- **MCAP topic discovery** — the Python multi-camera loader and shared
  topic constants are W6.
- **CLI work** — `clankers validate --scenario` calling the new
  `validate_against` helpers is W5.
- **Performance** — `ParallelVecEnvRunner`, binary frame, and dense
  hot-path vectors are W7.
- **Public-API stability shims** — any cross-crate consumer that imports
  `RobotModel.joints: HashMap<String, JointData>` keeps working in PR1
  via a parallel ordered `Vec`; the HashMap is removed only in PR2 once
  the layout-backed iterator covers the existing call sites in
  `crates/clankers-urdf/`.
- **`SchemaRegistry` / cross-version migration** — versioning is a `u32`
  per type today; an explicit registry and migration ladder is a Phase 5
  concern noted in the quality report.

## 4. Files to change

### NEW

| Path | Purpose |
|---|---|
| `crates/clankers-core/src/layout.rs` | `JointSpec`, `JointLayout`, `JointLayoutBuilder`, `SchemaMismatch` variants for layout. |
| `crates/clankers-core/src/schema.rs` | `ObservationSchema`, `ObservationSlot`, `ActionSchema`, `ActionSemantics`, `FrameSchema`, `RecorderSchema`, the shared `SchemaMismatch` error enum. |
| `crates/clankers-core/tests/layout_determinism.rs` | Integration tests: `same_urdf_produces_same_layout_hash`, `layout_hash_is_order_sensitive`, `layout_from_robot_model_orders_alphabetically`. |
| `crates/clankers-core/tests/schema_roundtrip.rs` | Serde roundtrip and `validate_against` integration tests for all four schema types. |

### MODIFY

| Path:line | Change |
|---|---|
| `crates/clankers-core/src/lib.rs:3` | Add `pub mod layout;` and `pub mod schema;` next to the existing `pub mod` block. |
| `crates/clankers-core/src/lib.rs:84-114` | Extend `prelude` to re-export `JointLayout`, `JointSpec`, `ObservationSchema`, `ObservationSlot`, `ActionSchema`, `ActionSemantics`, `FrameSchema`, `RecorderSchema`, `SchemaMismatch`. |
| `crates/clankers-core/Cargo.toml` | (PR2 only) add a `[dev-dependencies]` entry for `clankers-urdf` so the integration test in `tests/layout_determinism.rs` can parse a sample URDF. The dev-dep is one-directional and does not create a runtime cycle. |
| `crates/clankers-urdf/src/types.rs:230` | Add a private `actuated_joints_ordered: Vec<String>` field next to `joints: HashMap<String, JointData>` (PR2). Field is populated by the parser in alphabetic order. The public `joints: HashMap` is retained for name lookup until W2 finishes its sensor migration. |
| `crates/clankers-urdf/src/types.rs:251` | Replace `actuated_joints()` body so it iterates `self.actuated_joints_ordered.iter().filter_map(|n| self.joints.get(n))` instead of `self.joints.values()`. Public signature `impl Iterator<Item = &JointData>` is unchanged. |
| `crates/clankers-urdf/src/types.rs:268` | Mark `actuated_joint_names()` `#[deprecated(note = "use JointLayout::from_robot_model(&model).joint_names() instead")]`. The body's `names.sort_unstable()` becomes redundant once the layout exists, but the function is kept until W2 migrates the few existing callers in `clankers-env`. |
| `crates/clankers-urdf/src/types.rs` (impl block) | Add `pub fn to_layout(&self) -> clankers_core::layout::JointLayout` constructor that wraps the same alphabetic-order walk used by `actuated_joints_ordered`. |

### DELETE

None in this workstream. The `HashMap<String, JointData>` and
`actuated_joint_names()` helper survive W1; they are deleted in W2 once
every consumer has migrated to layout-bound iteration.

## 5. Checklist items

Each item is one focused commit at the ≤300-LOC ceiling.

- [ ] Add `crates/clankers-core/src/layout.rs` with `JointSpec { name: String, entity: Option<Entity>, joint_type: JointKind, limits: JointSpecLimits, axis: [f32; 3] }` plus `JointLayout { joints: Vec<JointSpec>, version: u32 }`. Derive `Debug`, `Clone`, `PartialEq`, `Serialize`, `Deserialize`.
- [ ] Implement `Hash` and `Eq` on `JointLayout`. Hash the ordered `(name, joint_type)` pairs only — `f32` limits are hashed via `f32::to_bits()` in a separate `limits_hash()` helper but not folded into the structural hash so two URDFs that differ only by limit precision do not perturb cross-process determinism checks.
- [ ] Implement `JointLayout::from_robot_model(model: &clankers_urdf::types::RobotModel) -> Self` (lives in `clankers-urdf` to avoid the reverse dependency; the constructor is re-exported by `clankers-core` only through trait-style helpers).
- [ ] Implement `JointLayout::version() -> u32`, `joint_names() -> impl Iterator<Item = &str>`, `dof() -> usize`, `index_of(name: &str) -> Option<usize>`, and `validate_against(&self, other: &Self) -> Result<(), SchemaMismatch>`.
- [ ] Add `SchemaMismatch` error enum in `crates/clankers-core/src/schema.rs` with variants `VersionMismatch { expected: u32, found: u32 }`, `JointCountMismatch { expected: usize, found: usize }`, `JointNameMismatch { index: usize, expected: String, found: String }`, `JointTypeMismatch { name: String, expected: JointKind, found: JointKind }`, `SlotMismatch { name: String, reason: String }`, `ActionSemanticsMismatch { expected: ActionSemantics, found: ActionSemantics }`, `EncodingMismatch { channel: String, expected: String, found: String }`. Derive `Debug`, `Clone`, `PartialEq`, `thiserror::Error`.
- [ ] Add `ObservationSchema` and `ObservationSlot { name: String, dtype: SchemaDtype, shape: Vec<usize>, units: Option<String>, source_sensor: String }` plus `SchemaDtype` enum (`F32`, `F64`, `U8`, `I32`, `Bool`). All `Serialize + Deserialize`. Include `version() -> u32` and `validate_against` returning `Result<(), SchemaMismatch>` (checks version, slot count, per-slot name/dtype/shape).
- [ ] Add `ActionSchema { semantics: ActionSemantics, dim: usize, low: Option<Vec<f32>>, high: Option<Vec<f32>>, version: u32 }` and `ActionSemantics` enum with variants `NormalizedPosition`, `AbsoluteJointPosition`, `JointVelocity`, `Torque`. Implement `version()` and `validate_against()`.
- [ ] Add `FrameSchema { channel: String, message_type: String, encoding: FrameEncoding }` and `FrameEncoding` enum (`Json`, `Cdr`, `RawBytes`, `ProtobufFqn(String)`). Implement `version()` and `validate_against()`.
- [ ] Add `RecorderSchema { channels: Vec<FrameSchema>, version: u32 }` with `validate_against()` performing per-channel checks (channel name set equality, then per-channel `validate_against`).
- [ ] Re-export the four schemas + `JointLayout` + `SchemaMismatch` from `crates/clankers-core/src/lib.rs` prelude.
- [ ] Add the integration test file `crates/clankers-core/tests/layout_determinism.rs` containing `same_urdf_produces_same_layout_hash`.
- [ ] Add the integration test file `crates/clankers-core/tests/schema_roundtrip.rs` for serde and `validate_against`.
- [ ] In PR2: add the private `actuated_joints_ordered: Vec<String>` field at `crates/clankers-urdf/src/types.rs:230` and populate it in the parser (alphabetic sort on insert).
- [ ] In PR2: replace the `actuated_joints()` body at `crates/clankers-urdf/src/types.rs:251` to iterate the ordered vector. Public signature unchanged.
- [ ] In PR2: deprecate `actuated_joint_names()` at `crates/clankers-urdf/src/types.rs:268` with a note pointing to `JointLayout`.
- [ ] In PR2: add `RobotModel::to_layout()` in `crates/clankers-urdf/src/types.rs`.

## 6. Tests required before implementation (test-first)

All tests below must be written and committed (red) **before** the
production code in the same PR. Test placement follows the conventions
in `.delegate/work/20260525-185618-workstream-plans/TASK.md` § "Test
placement conventions".

### `crates/clankers-core/tests/layout_determinism.rs`

- `same_urdf_produces_same_layout_hash` — parse the in-repo fixture URDF
  via `clankers_urdf::parser::parse_string` ten times, construct a
  `JointLayout` from each `RobotModel`, assert `std::hash::Hash` digests
  computed via `SipHasher24` agree across all ten copies and that
  `layout_a == layout_b` for every pair. This is the JointLayout
  deterministic-hash conformance test called out by LOOPS.yaml loop 1's
  gate.
- `layout_hash_is_order_sensitive` — given two layouts that hold the same
  joints in different orders, assert their hashes differ. Confirms the
  hash is structural (cannot accidentally collide across permutations).
- `layout_from_robot_model_orders_alphabetically` — parse the existing
  `ARM_URDF` fixture used by `crates/clankers-urdf/src/spawner.rs:158`
  (joints `shoulder`, `elbow`, `wrist_fixed`) and assert the resulting
  layout names are `["elbow", "shoulder"]` (alphabetic, fixed joint
  excluded).
- `layout_validate_against_self_is_ok` — `layout.validate_against(&layout)`
  returns `Ok(())`.
- `layout_validate_against_version_mismatch_is_err` — two layouts with
  different `version()` values return `Err(SchemaMismatch::VersionMismatch
  { .. })`.
- `layout_validate_against_joint_name_mismatch_is_err` — swap one joint
  name; assert `Err(SchemaMismatch::JointNameMismatch { index: _, .. })`.

### `crates/clankers-core/tests/schema_roundtrip.rs`

- `observation_schema_serde_roundtrip` — build a schema with two slots
  (one `F32` shape `[6]`, one `U8` shape `[64, 64, 3]`),
  `serde_json::to_string` then `from_str`, assert structural equality.
- `action_schema_serde_roundtrip` — build a schema with
  `ActionSemantics::NormalizedPosition`, `dim = 7`, finite `low`/`high`;
  roundtrip via JSON.
- `action_semantics_roundtrip_all_variants` — tag each of the four
  variants and roundtrip; assert exhaustive variants survive (guards
  against accidental enum-tag breakage).
- `frame_schema_serde_roundtrip` — build a schema for each
  `FrameEncoding` variant and roundtrip.
- `recorder_schema_serde_roundtrip` — build a recorder schema with three
  channels (`/joints`, `/camera/front`, `/camera/wrist`), roundtrip,
  assert order preserved.
- `recorder_schema_validate_against_self_is_ok`.
- `recorder_schema_validate_against_missing_channel_is_err` — drop one
  channel and assert `SchemaMismatch::EncodingMismatch` or a new
  `ChannelSetMismatch` variant (decided when the type is implemented;
  whichever ships, this test pins the behaviour).
- `action_schema_validate_against_semantics_mismatch_is_err` — same dim,
  different `ActionSemantics`, assert
  `Err(SchemaMismatch::ActionSemanticsMismatch { .. })`.

### Inline `#[cfg(test)] mod tests` in `crates/clankers-core/src/layout.rs`

- `joint_spec_eq_and_hash` — two equal specs hash equally; differing on
  `joint_type` hash differently.
- `joint_layout_index_of_round_trip` — `layout.index_of(name)` round-trips
  for every joint in a freshly built layout.
- `joint_layout_dof_matches_actuated_count` — `dof()` equals the number
  of actuated joints.

### Inline `#[cfg(test)] mod tests` in `crates/clankers-core/src/schema.rs`

- `observation_slot_shape_size` — `slot.shape.iter().product()` matches a
  helper `slot.size()`.
- `action_schema_validate_against_dim_mismatch_is_err`.
- `frame_schema_validate_against_encoding_mismatch_is_err`.

All tests must compile and **fail** when committed in the test-first
sub-step of each PR; the same PR's subsequent production commit turns
them green.

## 7. Success criteria

Each criterion is checkable with a concrete command. All `cargo`
invocations use `-j 24` per `CLAUDE.md`.

- `cargo test -j 24 -p clankers-core` passes, including the new
  `layout::tests`, `schema::tests`, and the two new integration tests
  under `crates/clankers-core/tests/`.
- `cargo test -j 24 -p clankers-core --test layout_determinism` runs
  `same_urdf_produces_same_layout_hash` and all five sibling cases
  green.
- `cargo test -j 24 -p clankers-core --test schema_roundtrip` runs all
  serde and `validate_against` cases green.
- `cargo test -j 24 -p clankers-urdf` passes after the PR2 migration
  (existing `model_dof`, `model_actuated_joint_names`, `spawn_*` tests
  still pass because `actuated_joints()` keeps its public signature; the
  iteration order is now stable instead of HashMap-arbitrary).
- `cargo build -j 24 --workspace` succeeds after PR2 (no consumer broke).
- `cargo doc -j 24 -p clankers-core --no-deps` succeeds; the new
  `JointLayout`, `ObservationSchema`, `ActionSchema`, `ActionSemantics`,
  `FrameSchema`, `RecorderSchema`, and `SchemaMismatch` items render
  with their doc comments (no `missing_docs` clippy warnings).
- `cargo clippy -j 24 -p clankers-core --all-targets --tests -- -D warnings`
  passes; new types respect the workspace `pedantic`/`nursery` lints.
- After PR2: `grep -rn 'self\.joints\.values()' crates/clankers-urdf/src/`
  returns **zero hits** outside the now-deprecated
  `actuated_joint_names` body — the canonical iterator at
  `crates/clankers-urdf/src/types.rs:251` is layout-backed.
- After PR2: running the layout-determinism test ten times in succession
  via `for i in 1..=10; do cargo test -j 24 -p clankers-core --test layout_determinism same_urdf_produces_same_layout_hash -- --nocapture; done`
  yields ten passes with identical reported hash digests.

## 8. Risks & mitigations

1. **Migrating `RobotModel.joints: HashMap<String, JointData>` to a
   layout-backed structure may break downstream code that imports the
   HashMap directly.** Several call sites in `clankers-sim` and
   `clankers-env` currently iterate `spawned.joints` (see
   `crates/clankers-sim/src/builder.rs` and the W2 pre-reading set).
   **Mitigation:** keep the public `joints: HashMap<String, JointData>`
   field in PR2 for name lookup and additionally store the private
   ordered `Vec<String>`. Only `actuated_joints()` switches to ordered
   iteration. W2 then migrates the call sites and W2's final PR can
   remove the HashMap if desired.

2. **Schema versioning bikeshed: u8 vs u16 vs u32 vs semver string.**
   **Mitigation:** pin every `version()` return to `u32` and document
   that it follows semver-like rules (bump major on breaking change of
   layout/shape/dtype; bump minor on additive change). The single u32
   keeps `Hash`/`Eq` trivial and avoids bringing in `semver` as a
   `clankers-core` dependency. A future `SchemaRegistry` (out of scope)
   can layer richer metadata on top.

3. **`Hash` over `JointLimits` with `f32` fields is awkward.** Hashing
   `f32::NAN` is non-canonical and bit-equal floats from different code
   paths may differ.
   **Mitigation:** structural `Hash` over `(name, joint_type)` only.
   Limits are surfaced via a separate `limits_hash() -> u64` helper that
   uses `f32::to_bits()` and is documented as "stable for the same URDF
   parsed by the same parser version; do not compare across major
   versions." The conformance test
   `same_urdf_produces_same_layout_hash` exercises only the structural
   hash, matching how W2/W3/W4 will check cross-process determinism.

4. **Re-export naming collision with existing `ObservationSpace` /
   `ActionSpace` in `crates/clankers-core/src/types.rs`.** Both names
   are already in the prelude.
   **Mitigation:** new types are `ObservationSchema` and `ActionSchema`,
   deliberately distinct from `*Space`. `ObservationSpace` describes a
   Gymnasium-shape contract (low/high/box); `ObservationSchema`
   describes a slot layout (name → dtype → shape → source sensor). Both
   are exported; doc comments cross-link them.

5. **Circular crate dependency risk if `JointLayout::from_robot_model`
   lives in `clankers-core`.** `clankers-urdf` already depends on
   `clankers-core` for `RobotId`; the reverse import would cycle.
   **Mitigation:** keep `JointLayout`, `JointSpec`, and schemas in
   `clankers-core` (no URDF dependency). Define `RobotModel::to_layout()`
   inside `clankers-urdf` so the conversion lives on the downstream
   side. The `clankers-core` test crate uses `clankers-urdf` only as a
   `[dev-dependencies]` entry to drive the determinism test; this is
   one-directional and only affects test builds.

## 9. PR breakdown

Exactly **2** commits, per LOOPS.yaml loop 1's
`expected_implementation_prs: 2` and the gate.

### PR1 — `feat(core): add JointLayout + four schema types`

**Scope summary:** Introduce the four contracts and their tests in
`clankers-core` only. No consumer migration. The crate compiles, all
new tests pass, and no other workspace crate changes.

**Files (diff estimate):**

- `crates/clankers-core/src/layout.rs` (+~220 LOC new).
- `crates/clankers-core/src/schema.rs` (+~260 LOC new).
- `crates/clankers-core/src/lib.rs` (+12 LOC: `pub mod` and prelude
  re-exports).
- `crates/clankers-core/tests/layout_determinism.rs` (+~120 LOC new,
  uses `clankers_urdf` as a `[dev-dependencies]` entry).
- `crates/clankers-core/tests/schema_roundtrip.rs` (+~140 LOC new).
- `crates/clankers-core/Cargo.toml` (+3 LOC: `clankers-urdf` dev-dep).

Total ≈ 755 LOC across six files. Sub-commits inside the PR keep each
logical step under 300 LOC: (a) tests-first commit (red), (b) layout
module commit (green for layout tests), (c) schema module commit (green
for schema tests), (d) prelude re-export commit.

**Checklist items from section 5 included:** items 1, 2, 3, 4, 5, 6,
7, 8, 9, 10, 11, 12.

### PR2 — `refactor(urdf): back actuated_joints with JointLayout-ordered Vec`

**Scope summary:** Migrate the in-tree producer of the determinism bug
— `RobotModel::actuated_joints()` — to iterate a layout-derived ordered
vector. Preserve the public iterator signature so downstream crates
keep compiling. Mark `actuated_joint_names()` deprecated (deletion is
W2's job).

**Files (diff estimate):**

- `crates/clankers-urdf/src/types.rs` (+~80 LOC, −~10 LOC: new field,
  new parser sort, body rewrite at line 251, `#[deprecated]` at line
  268, new `to_layout()` method).
- `crates/clankers-urdf/src/parser.rs` (+~20 LOC: populate the ordered
  vector when inserting joints; the file already iterates the URDF in
  document order, the sort happens once on `RobotModel` construction).
- `crates/clankers-urdf/src/types.rs` test module (+~50 LOC: new
  `actuated_joints_iteration_is_alphabetic` test, parse the existing
  multi-joint URDF and assert the iterator order).
- `crates/clankers-urdf/Cargo.toml` — no change (already depends on
  `clankers-core`).

Total ≈ 140 LOC. Single commit, comfortably under the 300-LOC
checklist-item ceiling.

**Checklist items from section 5 included:** items 13, 14, 15, 16.

After PR2: every grep target listed in section 7 holds, the
determinism conformance test passes ten times in a row, and the W2
plan can begin its sensor-migration work against a stable API.
