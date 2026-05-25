# Workstream 3 — Strip Panicking `Action`/`Observation` APIs; Introduce Views

Status: planning only (no code in this PR).
Quality report reference: `notes/clankers_codebase_quality_report_2026-05-25.md`,
Core Types section, "Action/Observation panic" gap.

## 1. Goal

Remove every `panic!`-on-wrong-variant accessor from `clankers_core::types::Action`
and replace the cloning `ObservationBuffer::as_observation()` hot-path read
with a zero-copy `ObservationView<'a>` borrow that downstream sensors,
vec-runners, and protocol encoders consume.

## 2. Why this workstream, why this order

- **Upstream dependency (W1):** the typed variants exposed by
  `as_continuous()`/`try_into_continuous()` and the dtype/shape carried by
  `ObservationView<'a>` are defined by W1's `ActionSchema` and
  `ObservationSchema`. Without those schemas the view cannot self-describe its
  layout to a serialiser.
- **Downstream unblocks (W4, W7):**
  - W4 (protocol parity) needs a non-panicking continuous accessor so that
    the gym server's `Step` handler at `crates/clankers-gym/src/server.rs:285`
    can return a `ProtocolError` instead of crashing the worker thread when
    a discrete-action env is misconfigured by a client.
  - W7 (binary protocol + `ParallelVecEnvRunner`) needs `ObservationView<'a>`
    so that the batch path can write directly into a contiguous output buffer
    without the per-env `Vec<f32>` clone that
    `crates/clankers-env/src/vec_buffer.rs:72` performs today.
- **Order:** must land after W1 and before W4/W7. Independent of W2 and W6.

## 3. Out of scope

- Schema design (`ActionSchema`/`ObservationSchema` definitions live in W1).
- Sensor refactor / layout-bound action application (W2).
- Promotion of `ObsEncoding` to `EncodedObservation` enum, image-on-reset
  parity, Python `ProtocolError` (W4).
- Binary protocol frame layout (W7).
- `Observation::as_slice()` at `crates/clankers-core/src/types.rs:41-49`,
  `Observation::as_mut_slice()` at line 45, `Observation::into_vec()` at
  line 49. These are **not panicking** — `Observation` is unconditionally
  `Vec<f32>`-backed. They stay source-compatible. `ObservationView<'a>` is
  added **alongside**, not in place of, these helpers.
- The `as_slice` / `as_mut_slice` / `into_vec` names on `Vec<f32>` and `[T]`
  themselves (these are std API; grep zero-hits criterion in section 7
  excludes them).

## 4. Files to change

Evidence below is taken from PLAN.md's pre-gathered grep at
`.delegate/work/20260525-185618-workstream-plans/03/PLAN.md`.

### NEW

- `crates/clankers-core/src/error.rs` — add `ActionKindError` enum variant
  (existing `ValidationError` module).
- `crates/clankers-core/src/view.rs` — new module defining
  `ObservationView<'a>` (borrowed `&'a [f32]` + `Dtype` + `shape: &'a [usize]`).
  Re-export from `crates/clankers-core/src/lib.rs`.
- `crates/clankers-core/tests/action_fallible.rs` — fallible-API test
  (see section 6).
- `crates/clankers-env/tests/observation_view_roundtrip.rs` — view zero-copy
  test (see section 6).

### MODIFY (PR1 — additive, deprecate)

- `crates/clankers-core/src/types.rs:121-126` — `Action::as_slice()`: keep
  body, add `#[deprecated(since = "<ver>", note = "use as_continuous()")]`.
- `crates/clankers-core/src/types.rs:134-139` — `Action::as_mut_slice()`:
  same `#[deprecated]` attribute.
- `crates/clankers-core/src/types.rs:142-146` — `Action::into_vec()`: same.
- `crates/clankers-core/src/types.rs:~119` — insert
  `Action::as_continuous(&self) -> Option<&[f32]>` and
  `Action::try_into_continuous(self) -> Result<Vec<f32>, ActionKindError>`
  immediately above the deprecated trio. Also adjust `Action::scale()` at
  line 159-165 (which today calls the soon-to-be-deprecated `as_slice()`)
  to call `as_continuous().expect("scale requires continuous action")` —
  the explicit `.expect` documents the contract at the call site rather
  than hiding it behind a panicking accessor. (Internal use only, no
  behaviour change.)
- `crates/clankers-env/src/buffer.rs:127` — add
  `ObservationBuffer::view(&self) -> ObservationView<'_>` alongside the
  existing `as_observation()` (kept for serialisation boundaries).
- `crates/clankers-env/src/buffer.rs:137` — mark
  `ObservationBuffer::as_slice` `#[deprecated(note = "use view()")]`.

### MODIFY (PR2 — migrate + delete)

- `crates/clankers-core/src/types.rs:121-146` — **DELETE** the three
  panicking methods.
- `crates/clankers-core/src/types.rs:977-1022` — **DELETE** the six
  `#[should_panic]` tests (`action_discrete_as_slice_panics`,
  `action_discrete_values_panics`, `action_discrete_as_mut_slice_panics`,
  `action_discrete_into_vec_panics`, `action_multi_discrete_as_slice_panics`,
  `action_multi_discrete_as_mut_slice_panics`,
  `action_multi_discrete_into_vec_panics`).
- `crates/clankers-core/src/types.rs:951, 958, 1029, 1107, 1113-1114` —
  in-module unit tests still call `.as_slice()`/`.as_mut_slice()`/
  `.into_vec()` on `Action::Continuous`. Migrate to `.as_continuous().unwrap()`
  and `.try_into_continuous().unwrap()`.
- `crates/clankers-core/src/types.rs:160` — `Action::scale()`'s internal
  caller updated to use `as_continuous().expect(...)`. The deprecated
  path is removed in this PR.
- `crates/clankers-env/src/buffer.rs:104, 121, 127, 137` — `as_observation`
  retained; `as_slice` deleted. Internal `data.clone()` in `as_observation`
  remains (it is the documented serialisation boundary).
- `crates/clankers-env/src/vec_buffer.rs:72` — replace
  `flat.copy_from_slice(obs.as_slice())` with
  `flat.copy_from_slice(buf.view().as_f32())` (where `buf` is the source
  `ObservationBuffer`). Add
  `VecObsBuffer::row(env_idx) -> ObservationView<'_>` so callers can read
  per-env rows without per-row `Observation` allocation.
- `crates/clankers-env/src/vec_runner.rs:336, 374, 397, 403, 421-423` —
  test sites currently using `Action::as_slice` / `Observation::as_slice`
  for assertions: where the value is an `Action`, migrate to
  `.as_continuous().unwrap()`; where the value is an `Observation`, leave
  unchanged (out of scope per section 3).
- `crates/clankers-gym/src/server.rs:285` — `action.as_slice()` →
  `action.as_continuous().ok_or(ProtocolError::DiscreteActionForContinuousSpace)?`
  (the error variant is added in W4 — for PR2 of W3, use
  `.expect("server contract: continuous action expected")` and file a
  TODO citing W4).
- `crates/clankers-gym/src/env.rs:288` — same migration as server.rs:285.
- `apps/clankers-app/src/main.rs:85` — `action.as_slice()` → `.as_continuous()
  .expect("CLI demo uses continuous action")`.
- `examples/src/bin/arm_gym.rs:26` — same.
- `crates/clankers-viz/src/ui.rs:353, 388` — same.

### Test-only sites that stay on `Observation::as_slice` (out of scope)

These are test assertion code where the value is an `Observation` (not an
`Action`) and `as_slice` is the non-panicking helper on the always-`Vec<f32>`
struct. Per section 3 they remain compatible:

- `crates/clankers-env/src/vec_buffer.rs` — assertion-style reads.
- `crates/clankers-env/src/sensors.rs:805, 829, 882, 920, 942, 965, 1018,
  1359, 1459` — sensor unit tests reading `Observation` slices.
- `crates/clankers-gym/src/vec_env.rs` — vec env tests.

### DELETE

- `crates/clankers-core/src/types.rs:121-146` (PR2).
- `crates/clankers-core/src/types.rs:977-1022` (PR2).
- `crates/clankers-env/src/buffer.rs:137` (PR2).

## 5. Checklist items

Each item is ≤300 LOC.

- [ ] PR1: Define `ActionKindError` `thiserror` enum in
      `crates/clankers-core/src/error.rs` with variants
      `ExpectedContinuous { got: &'static str }` and a `Display` body
      mentioning the offending variant.
- [ ] PR1: Add `Action::as_continuous(&self) -> Option<&[f32]>` in
      `crates/clankers-core/src/types.rs` at line ~119 (before the
      deprecated trio). Matches `Self::Continuous(v)` → `Some(v.as_slice())`,
      anything else → `None`.
- [ ] PR1: Add `Action::try_into_continuous(self) -> Result<Vec<f32>,
      ActionKindError>` adjacent to `as_continuous`.
- [ ] PR1: Apply `#[deprecated(since = "<next-version>", note = "use
      as_continuous() / try_into_continuous()")]` to
      `Action::as_slice`, `Action::as_mut_slice`, `Action::into_vec`.
      Leave bodies intact so existing callers compile with a warning.
- [ ] PR1: Define `ObservationView<'a> { data: &'a [f32], dtype: Dtype,
      shape: &'a [usize] }` in `crates/clankers-core/src/view.rs`. Provide
      `fn as_f32(&self) -> &[f32]` and `fn shape(&self) -> &[usize]`.
- [ ] PR1: Add `ObservationBuffer::view(&self) -> ObservationView<'_>` to
      `crates/clankers-env/src/buffer.rs` returning a borrow of
      `&self.data` plus the buffer's flat shape (`[self.total_dim]`).
      Zero allocations on the hot path.
- [ ] PR1: Add `VecObsBuffer::row(env_idx: usize) -> ObservationView<'_>`
      to `crates/clankers-env/src/vec_buffer.rs` returning a borrow of
      `&self.flat[env_idx * dim .. (env_idx + 1) * dim]` with the per-env
      shape metadata.
- [ ] PR1: Write `crates/clankers-core/tests/action_fallible.rs` (per
      section 6). Write
      `crates/clankers-env/tests/observation_view_roundtrip.rs`.
      Verify `cargo test -j 24 -p clankers-core action_fallible`
      and `cargo test -j 24 -p clankers-env observation_view_roundtrip`
      pass.
- [ ] PR1: CHANGELOG entry: "Deprecate Action::as_slice/as_mut_slice/
      into_vec — use Action::as_continuous() / try_into_continuous()
      instead. ObservationBuffer::view() replaces as_slice() for hot-path
      reads. Removal in next minor version." Land in PR1 so users have
      one release window to migrate.
- [ ] PR2: Migrate every `Action::as_slice`/`as_mut_slice`/`into_vec`
      call site enumerated in section 4 ("MODIFY (PR2 — migrate + delete)").
- [ ] PR2: Migrate `crates/clankers-env/src/vec_buffer.rs:72` to use
      `buf.view().as_f32()`. Replace at least one other per-env
      `Observation` clone in the same file with `VecObsBuffer::row()`.
- [ ] PR2: Delete `Action::as_slice`/`as_mut_slice`/`into_vec` (lines
      121-146). Delete the seven `#[should_panic]` tests (lines
      977-1022). Delete `ObservationBuffer::as_slice` (line 137).
- [ ] PR2: Run `cargo clippy -j 24 --workspace --all-targets -- -D
      deprecated` and fix any straggling callers. (No `#[allow(deprecated)]`
      escape hatches.)
- [ ] PR2: Sweep `docs/` and example READMEs for any reference to the
      removed panic semantics; update or delete.

## 6. Tests required before implementation

Test-first discipline: these files are committed in PR1 alongside the
new API, and **must fail to compile or fail at runtime** if the new
APIs are not present.

### `crates/clankers-core/tests/action_fallible.rs`

```rust
//! Proves that the new Action accessors never panic on non-Continuous variants.

use clankers_core::error::ActionKindError;
use clankers_core::types::Action;

#[test]
fn as_continuous_returns_none_for_discrete() {
    let action = Action::Discrete(0);
    assert!(action.as_continuous().is_none(), "must not panic, must return None");
}

#[test]
fn as_continuous_returns_none_for_multi_discrete() {
    let action = Action::MultiDiscrete(vec![1, 2]);
    assert!(action.as_continuous().is_none());
}

#[test]
fn as_continuous_returns_some_for_continuous() {
    let action = Action::Continuous(vec![0.5, -0.5]);
    assert_eq!(action.as_continuous(), Some(&[0.5_f32, -0.5_f32][..]));
}

#[test]
fn try_into_continuous_returns_err_for_discrete() {
    let action = Action::Discrete(0);
    let err = action.try_into_continuous().unwrap_err();
    assert!(matches!(err, ActionKindError::ExpectedContinuous { .. }));
}

#[test]
fn try_into_continuous_returns_err_for_multi_discrete() {
    let action = Action::MultiDiscrete(vec![1, 2]);
    assert!(action.try_into_continuous().is_err());
}

#[test]
fn try_into_continuous_returns_ok_for_continuous() {
    let action = Action::Continuous(vec![1.0, 2.0]);
    assert_eq!(action.try_into_continuous().unwrap(), vec![1.0, 2.0]);
}
```

Assertion shape: every assertion is a structural one (`is_none`,
`is_err`, `matches!`). **Zero `#[should_panic]` attributes** — the entire
point is that panics are gone.

### `crates/clankers-env/tests/observation_view_roundtrip.rs`

```rust
//! Proves ObservationBuffer::view() returns a byte-equal borrow with no
//! allocations across repeated calls.

use clankers_env::buffer::ObservationBuffer;

#[test]
fn view_matches_as_observation_clone() {
    let mut buf = ObservationBuffer::new();
    let a = buf.register("pos", 3);
    let b = buf.register("vel", 2);
    buf.write(a, &[1.0, 2.0, 3.0]);
    buf.write(b, &[4.0, 5.0]);

    let owned = buf.as_observation();
    let view = buf.view();
    assert_eq!(view.as_f32(), owned.as_slice());
    assert_eq!(view.shape(), &[5][..]);
}

#[test]
fn view_is_zero_alloc_across_calls() {
    let mut buf = ObservationBuffer::new();
    buf.register("pos", 3);
    buf.write(0, &[1.0, 2.0, 3.0]);

    // 1000 view calls must reuse the same pointer.
    let first_ptr = buf.view().as_f32().as_ptr();
    for _ in 0..1000 {
        assert_eq!(buf.view().as_f32().as_ptr(), first_ptr);
    }
}
```

The pointer-stability assertion is the proxy for "zero allocations" that
does not require a custom allocator harness. (Wired-in alloc counters are
overkill for a unit test; the contract is satisfied if the borrow is of
`&self.data` directly.)

## 7. Success criteria

Each is verifiable from the project root with one command.

- `cargo test -j 24 -p clankers-core action_fallible` — passes (six asserts,
  all structural, zero `#[should_panic]`).
- `cargo test -j 24 -p clankers-env observation_view_roundtrip` — passes.
- `cargo test -j 24 --workspace` — passes (the existing test suite still
  green after migration).
- `cargo clippy -j 24 --workspace --all-targets -- -D deprecated` — passes
  after PR2 lands. (Proves every call site migrated; no
  `#[allow(deprecated)]` workarounds were added.)
- `cargo build -j 24 --workspace` — passes with zero `deprecated` warnings
  after PR2.
- `grep -rn 'as_slice()\|as_mut_slice()\|into_vec()' crates/ apps/
  examples/` — every remaining hit is on either (a) `Observation` (the
  non-panicking always-`Vec<f32>` helper, see section 3), (b) `Vec<f32>` /
  `&[f32]` / std slice methods, or (c) a `&str`/`&[u8]` `as_slice`.
  PR2 ships an explicit inspection list in the commit message enumerating
  every remaining hit and classifying it. Zero hits classify as "Action
  panicker" or "ObservationBuffer cloning read".

## 8. Risks & mitigations

1. **`ObservationView<'a>` lifetime ergonomics in server response paths.**
   The gym server in `crates/clankers-gym/src/server.rs` constructs a
   `Step` response that is serialised and sent over the socket. A
   `&'a [f32]` view cannot cross that `await` boundary if the underlying
   buffer is borrowed from a `Resource`.
   **Mitigation:** keep owned `Observation` (and `as_observation()`) at
   the serialisation boundary. `ObservationView<'_>` is exclusively for
   internal hot loops (sensor write→read, vec-runner row copy, MPC
   read-back). PR1's `view()` API is added; PR2 only migrates internal
   callers, never the server's response struct.

2. **Removing `as_slice`/`as_mut_slice`/`into_vec` breaks downstream
   embedders** who depend on the public `clankers_core::types::Action`
   surface (e.g. external training scripts in `python/` via PyO3 bindings
   or example bins users have cloned).
   **Mitigation:** two-PR deprecate-then-delete. PR1 ships
   `#[deprecated]` with a `note` pointing at the new API, so users get a
   compile-time warning for one release window. CHANGELOG entry lands in
   PR1 with a migration snippet. PR2 (deletion) ships in the next minor
   version, not the same release.

3. **`#[should_panic]` tests document current invariants** and may be
   externally referenced from blog posts, talks, or downstream test
   suites that copy patterns.
   **Mitigation:** during PR2, grep `docs/`, `notes/`, `README.md`,
   `apps/clankers-app/README.md`, and the `examples/` README files for
   "panic" / "should_panic" / "as_slice". Update or remove any reference.
   Replace deleted should_panic tests with the new structural assertions
   in `tests/action_fallible.rs` so behavioural coverage does not drop.

4. **Internal `Action::scale()` at types.rs:159-165 calls the
   soon-deprecated `as_slice()`.** Without action, PR1's deprecation
   warning would fire from within the library itself.
   **Mitigation:** PR1 rewrites `scale()` to call
   `self.as_continuous().expect("scale() requires Action::Continuous")`.
   Documents the contract at the call site. Zero `#[allow(deprecated)]`
   needed.

## 9. PR breakdown

Exactly **two commits**.

### PR1 — Additive: fallible APIs, `ObservationView`, deprecate panickers

Scope: pure addition + `#[deprecated]` markers. No call sites change.
Existing tests keep passing (the deprecated methods still work). New
tests prove the new API.

- New: `crates/clankers-core/src/error.rs` — `ActionKindError`.
- New: `crates/clankers-core/src/view.rs` — `ObservationView<'a>`.
- New: `crates/clankers-core/tests/action_fallible.rs`.
- New: `crates/clankers-env/tests/observation_view_roundtrip.rs`.
- Modify: `crates/clankers-core/src/types.rs` (add `as_continuous` +
  `try_into_continuous`; `#[deprecated]` on the three panickers; rewrite
  `scale()` internal caller).
- Modify: `crates/clankers-env/src/buffer.rs` (add `view()`; `#[deprecated]`
  on `as_slice`).
- Modify: `crates/clankers-env/src/vec_buffer.rs` (add
  `VecObsBuffer::row()`).
- Modify: `CHANGELOG.md` — migration entry.
- **Diff target:** ~250 LOC across 7 files.
- **Commit message:** `feat(core): add fallible Action accessors and
  ObservationView; deprecate panicking helpers`.

### PR2 — Migrate call sites, delete panickers, delete should_panic tests

Scope: migrate every enumerated call site in section 4 ("MODIFY (PR2)"),
delete the deprecated methods and their `#[should_panic]` tests, prove
clippy `-D deprecated` passes.

- Modify: `crates/clankers-core/src/types.rs` — delete lines 121-146,
  delete lines 977-1022, migrate inline unit-test calls at lines
  951/958/1029/1107/1113.
- Modify: `crates/clankers-env/src/buffer.rs` — delete `as_slice` at
  line 137.
- Modify: `crates/clankers-env/src/vec_buffer.rs` — migrate line 72
  copy hotpath to `view()`.
- Modify: `crates/clankers-env/src/vec_runner.rs` — migrate test sites
  at lines 336, 374, 397, 403, 421-423 (Action sites only).
- Modify: `crates/clankers-gym/src/server.rs` — line 285.
- Modify: `crates/clankers-gym/src/env.rs` — line 288.
- Modify: `apps/clankers-app/src/main.rs` — line 85.
- Modify: `examples/src/bin/arm_gym.rs` — line 26.
- Modify: `crates/clankers-viz/src/ui.rs` — lines 353, 388.
- Modify: `docs/` sweep — remove panic references.
- **Diff target:** ~400 LOC across 12 files.
- **Commit message:** `refactor(core): migrate to fallible Action API,
  delete panicking helpers`.

Verification before merging PR2:

- `cargo test -j 24 --workspace` — green.
- `cargo clippy -j 24 --workspace --all-targets -- -D deprecated` — green.
- Section 7 grep — every remaining hit classified.
