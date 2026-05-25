# WS1.5 — Stabilisation: physics build + W1 polish

Workstream 1.5 of 8. A narrow stabilisation pass between W1 (foundations
landed in `28ed8b5` + `3b463eb` under deferred-polish) and W2 (sensor /
motor migration onto `JointLayout`). Source: investigation triggered
during W1 PR2 implementation that surfaced a pre-existing
`clankers-physics` build break which had been masked by stale cargo
incremental cache.

## 1. Goal

Restore green workspace-wide `cargo check --all-targets`, run the
verification commands deferred from W1 PR1 and PR2 (test, clippy, doc,
build), and fix any issues surfaced. Leave the tree at a state where
**W2's implementer can safely run the full gate** instead of inheriting
unknown latent breakage.

## 2. Why this workstream, why this order

W1 shipped under a deferred-polish exception (user signalled resource
contention; verification scope was reduced to `cargo check` per crate).
That was the right call at the time. But W2 modifies
`crates/clankers-physics/src/rapier/systems.rs` heavily — and we now
know `clankers-physics` doesn't even build. Letting W2 inherit that
state means W2's implementer either (a) can't compile-check its own
work, or (b) silently steps over a pre-existing break, conflating its
own bugs with the inherited one.

W1.5 lands **after** W1 and **before** W2. It is not a "new" workstream
— it's the formal name for the polish pass that W1 deferred plus the
single pre-existing fix that polish surfaced.

The bug also pre-dates W1: introduced by commit `65aff4a` on 2026-03-12
("fix: resolve all clippy warnings and remove unused pre-commit
config"). A clippy auto-fix likely rewrote an `eprintln!`/`println!`
into `bevy::log::warn!` without checking the dependent crate's feature
set. Workspace bevy is `default-features = false`; only
`clankers-record/Cargo.toml` opts into `features = ["bevy_log"]`.
`clankers-physics` does not.

## 3. Out of scope

- Migrating call sites of `actuated_joint_names()` — W2's job. The 3
  deprecation warnings shown in workspace check
  (`examples/src/bin/cartpole_gym.rs:58`,
  `crates/clankers-urdf/src/types.rs:453+455` in test code) stay until
  W2 ships layout-bound sensors.
- Any new feature, refactor, or contract change.
- The polish pass for W2-W8 — each workstream owns its own polish.
- Hunting for OTHER stale-cache surprises beyond what `cargo check
  --workspace --all-targets` and the W1 gate commands report. If new
  breaks emerge that aren't W1-caused, file them as a separate
  follow-up; do not bloat W1.5 into a workspace-wide audit.

## 4. Files to change

### MODIFY

| Path:line | Change |
|---|---|
| `crates/clankers-physics/Cargo.toml` | Bevy dep line: `bevy = { workspace = true, features = ["bevy_log"] }`. Matches the pattern in `crates/clankers-record/Cargo.toml`. This is the single source-code fix in W1.5. |

### NO CODE CHANGES expected elsewhere

If the W1 polish-pass commands surface other issues, fixes land as
sub-commits in W1.5 (one commit per logical fix). Do NOT amend the
existing W1 commits (`28ed8b5`, `3b463eb`); the deferred-polish
convention is explicit: fixes from polish land as `chore(<scope>):
polish W1 …` commits, not amendments.

## 5. Checklist items

Each item is one focused commit at the ≤300-LOC ceiling.

- [ ] Add `bevy_log` feature to `crates/clankers-physics/Cargo.toml`'s
  bevy dependency line.
- [ ] Verify `cargo check -j 24 -p clankers-physics --all-targets`
  passes (no warnings beyond the W1 deprecation ones).
- [ ] Verify `cargo check -j 24 --workspace --all-targets` passes from
  a clean build: `cargo clean && cargo check -j 24 --workspace
  --all-targets`. The clean build avoids the stale-cache illusion.
- [ ] Run W1 deferred polish for `clankers-core`:
  - `cargo test -j 24 -p clankers-core`
  - `cargo test -j 24 -p clankers-core --test layout_determinism`
  - `cargo test -j 24 -p clankers-core --test schema_roundtrip`
  - `cargo clippy -j 24 -p clankers-core --all-targets --tests -- -D warnings`
  - `cargo doc -j 24 -p clankers-core --no-deps`
- [ ] Run W1 deferred polish for `clankers-urdf`:
  - `cargo test -j 24 -p clankers-urdf`
  - `cargo clippy -j 24 -p clankers-urdf --all-targets --tests -- -D warnings`
  - `cargo doc -j 24 -p clankers-urdf --no-deps`
- [ ] Run W1 conformance loop: `for i in 1..=10; do cargo test -j 24
  -p clankers-core --test layout_determinism
  same_urdf_produces_same_layout_hash -- --nocapture; done` — assert
  10 passes with identical hash digests printed.
- [ ] Run workspace build: `cargo build -j 24 --workspace`.
- [ ] Run grep success-criterion from W1 PR2:
  `grep -rn 'self\.joints\.values()' crates/clankers-urdf/src/` — expect
  zero hits outside the `#[deprecated] fn actuated_joint_names()` body.
- [ ] For each polish command that fails, land a focused fix commit:
  - `chore(core): polish W1 PR1 — <one-line description>`
  - `chore(urdf): polish W1 PR2 — <one-line description>`
  - `chore(physics): re-enable bevy_log feature` (the bevy_log fix above)
- [ ] After all checks green, delete this plan's POLISH TODO from
  `.delegate/work/20260525-194710-w1-clankers-core-contracts/01/IMPLEMENTATION.md`
  and `…/02/IMPLEMENTATION.md` (or annotate "resolved by W1.5 at
  <commit>"). Keeps future readers from chasing already-fixed TODOs.

## 6. Tests required before implementation (test-first)

W1.5 is largely test execution, not new code. The single new
implementation step (bevy_log feature flag) does not warrant a new
test — its correctness is proven by `cargo check -p clankers-physics`
flipping from red to green.

The test-first discipline applies to **any unexpected fixes** the polish
pass surfaces:

- If a `clankers-core` test fails, write a smaller reproducer test first,
  then fix.
- If clippy raises a lint that requires a real refactor (not a one-line
  attribute), add a test that locks the expected behaviour, then fix.
- If a doc warning is structural (missing item, broken link), no test —
  doc warnings are surfaced by `cargo doc`.

## 7. Success criteria

Each criterion is checkable with a concrete command. All `cargo`
invocations use `-j 24` per `CLAUDE.md`.

- `cargo check -j 24 --workspace --all-targets` from a clean build
  (`cargo clean` first) exits 0 with no errors.
- `cargo test -j 24 --workspace` exits 0 (modulo any tests already
  `#[ignore]`d before W1 — e.g. `mpc_walk`).
- `cargo clippy -j 24 --workspace --all-targets --tests -- -D warnings`
  exits 0. The 3 deprecation warnings from `actuated_joint_names()`
  must be silenced inside the `#[deprecated]` function's own callers in
  `clankers-urdf` tests via `#[allow(deprecated)]` on the test;
  `examples/src/bin/cartpole_gym.rs:58` is similarly suppressed for
  now (W2 deletes the call).
- `cargo doc -j 24 --workspace --no-deps` exits 0.
- `cargo build -j 24 --workspace` exits 0.
- 10× determinism loop yields 10 identical hash digests.
- `grep -rn 'self\.joints\.values()' crates/clankers-urdf/src/` returns
  only the body of `#[deprecated] fn actuated_joint_names`.
- The POLISH TODO blocks in W1's loop IMPLEMENTATION.md files are
  removed or annotated as resolved.

## 8. Risks & mitigations

1. **W1's deferred clippy might surface real lint debt, not just style
   noise.** `pedantic` + `nursery` are warn-level workspace-wide and
   become deny under `-D warnings`. The new `layout.rs` and `schema.rs`
   have ~480 LOC of new public API that has never been linted.
   **Mitigation:** if a lint requires non-trivial refactoring, prefer a
   targeted `#[allow(clippy::<lint>)]` with a TODO comment over a
   sprawling rewrite. W1.5 is stabilisation, not redesign. File the
   bigger refactor as a separate `chore(core): polish W1 PR1 — clean up
   <X>` commit if scope justifies, or defer to W2 if it lives in code
   W2 will touch anyway.

2. **The bevy_log fix may not be enough — `clankers-physics` might have
   other compile errors masked by the first one.** Rust stops at the
   first hard error; subsequent ones surface only after the first is
   fixed.
   **Mitigation:** after flipping the feature flag, run `cargo check -p
   clankers-physics --all-targets` and address any cascaded errors. If
   the cascade is small (< ~5 errors), fix in the same commit. If
   larger, split into multiple `chore(physics): polish` commits.

3. **The determinism test might fail.** `JointLayout`'s `Hash` impl was
   written carefully, but the test depends on `clankers_urdf::parser`
   yielding deterministic `RobotModel` instances and W1 PR2's
   `actuated_joints_ordered` sort. If either has subtle non-determinism
   (e.g. HashMap key insertion order leaking into joint axis floats),
   the 10× loop will catch it.
   **Mitigation:** if it fails intermittently, add print-diff between
   the differing pair of hashes and trace back to the source field.
   Most likely culprit: the parser's `joints: HashMap` is iterated
   somewhere to construct `JointData.limits` in non-deterministic
   order. The W1 PR2 sort handles joint-name order; if internal joint
   data is non-deterministic, that's a separate bug to fix here.

4. **The `cargo clean` requirement is expensive on a 32-core
   constrained machine.** Full rebuild of the workspace takes minutes.
   **Mitigation:** run `cargo clean` once at the start of the polish
   pass and accept the cost. The whole point of W1.5 is to make sure
   the next workstream starts from a known-clean tree.

5. **W1.5's commit count is unbounded** — if 5 polish issues surface,
   that's 5 commits.
   **Mitigation:** that's by design. Each commit is small and reviewable.
   The PR breakdown in section 9 names the expected commits; surplus
   ones append.

## 9. PR breakdown

Exactly **1 mandatory PR + N polish follow-up commits** (N depends on
what the deferred polish surfaces).

### PR (mandatory) — `chore(physics): re-enable bevy_log feature`

**Scope summary:** Single-line change to
`crates/clankers-physics/Cargo.toml` flipping `bevy_log` on. Restores
workspace-wide `cargo check --all-targets`. Unblocks W2.

**Files (diff estimate):**
- `crates/clankers-physics/Cargo.toml` (+1 LOC).

Total ≈ 1 LOC. The commit message body cites commit `65aff4a` (where
the bug was introduced) and the bevy `default-features = false`
workspace decision.

### Polish follow-ups (as needed) — `chore(<scope>): polish W1 PR<N> — <desc>`

One commit per logical fix. Expected categories:

- **clippy fixes** in `clankers-core` `layout.rs` or `schema.rs`.
- **doc warning fixes** (missing summary line, broken intra-doc link).
- **test failure fixes** if the determinism test or any schema serde
  test reveals a real bug.
- **deprecation suppression** for the 3 known `actuated_joint_names()`
  call sites (or — preferred — preempt one by removing the
  `cartpole_gym.rs:58` call entirely, since it's just a debug print).

Each follow-up commit:
- Touches one crate (scope tag in commit message).
- Includes a one-line summary in IMPLEMENTATION.md noting which polish
  command surfaced the issue.
- Does NOT mix unrelated fixes.

After all follow-ups land, the W1 IMPLEMENTATION.md POLISH TODO blocks
are annotated `resolved by W1.5 at <commit-range>`.

## How to launch

This plan is self-contained. To execute:

```
/dg:work w1.5
```

The Delegate task folder
`.delegate/work/{timestamp}-w1.5-polish-and-fix/` is set up alongside
this commit with a single-loop `LOOPS.yaml` for the mandatory PR plus
an open-ended slot for polish follow-ups (the orchestrator spawns
additional loops as polish failures surface).
