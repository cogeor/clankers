# WS6 — MCAP Recorder/Loader Schema Parity

Workstream 6 of 8. Source: Workstream Catalogue in
`.delegate/work/20260525-185618-workstream-plans/TASK.md`.
Authoritative gap reference:
`notes/clankers_codebase_quality_report_2026-05-25.md` (finding #3).

## 1. Goal

Make the Rust MCAP recorder and the Python MCAP loader agree on a single
versioned topic schema so that multi-camera recordings written by
`clankers-record` are losslessly discoverable by
`python/clankers/mcap_loader.py` (and the four CI-ignored data-pipeline
tests are turned back on).

## 2. Why this workstream, why this order

The canonical quality report
(`notes/clankers_codebase_quality_report_2026-05-25.md`) **finding #3 —
"MCAP Camera Recording And Loading Disagree On Topics"** documents the
exact bug this workstream fixes: the Rust recorder writes per-label
camera topics (`/camera/{label}` at
`crates/clankers-record/src/recorder.rs:556`) while the Python loader
hard-codes a single channel (`CHANNEL_IMAGE = "/camera/image"` at
`python/clankers/mcap_loader.py:75`). Any multi-camera recording is
silently dropped on the loader side, returning `images=None`.

**Dependency edge in:** W1. The `RecorderSchema` and `FrameSchema` types
introduced by W1 (`crates/clankers-core/src/schema.rs` in WS1-plan.md
§4) carry the `version()` field that the new
`crates/clankers-record/src/schema.rs` module attaches to its topic
constants. Until W1 lands, W6's `schema::topics` module can ship the
string constants but cannot emit a versioned recorder manifest.

**Independent of:** W2 (joint-layout sensors), W3 (typed action/observation
APIs), W4 (gym protocol parity). None of those reach into MCAP topic
strings.

**Unblocks:**

- The Cosmos sim-to-real prepare path
  (`python/clankers/cosmos/prepare.py`) which currently assumes one
  camera per episode and silently no-ops on multi-camera datasets.
- The diffuser training scripts under `python/clankers/training/` that
  load image episodes via `McapEpisodeLoader.to_sb3_replay_buffer()`.
- W7's async/buffered recorder (PR4) which will need stable channel
  identifiers to attribute backpressure metrics per camera.

W6 runs in parallel with W2/W3/W4 and lands after W1.

## 3. Out of scope

The following look adjacent but are deliberately deferred:

- **Recorder rewrite for async/buffered writes** — bounded-channel,
  dropped-frame metrics, and frame decimation are W7 PR4. This
  workstream changes only the channel naming/discovery surface.
- **New sensor channels** — no new payload types are introduced; only
  the existing `/joint_states`, `/actions`, `/reward`, `/body_poses`,
  and `/camera/{label}` channels are touched.
- **Schema type design** — `RecorderSchema`, `FrameSchema`, and
  `FrameEncoding` are defined by W1. W6 only consumes them.
- **Python dataset loader refactor beyond `mcap_loader.py`** —
  `python/clankers/dataset.py`, `trajectory_dataset.py`, and the
  diffuser data pipelines keep their current APIs. They get the new
  `images_by_camera` key for free; callers that ignore it see no
  behaviour change.
- **Compression / frame decimation** — left to W7.
- **Cosmos pipeline migration** to multi-camera input — out of scope;
  surfaced as a follow-up risk in §8.
- **Deprecating the `images` flat-ndarray key** — back-compat is
  preserved indefinitely in this workstream. The `DeprecationWarning`
  ladder is explicitly not started here (see §8 risk 2).

## 4. Files to change

### NEW

| Path | Purpose |
|---|---|
| `crates/clankers-record/src/schema.rs` | Topic name constants: `CAMERA_TOPIC_PREFIX = "/camera/"`, `JOINT_STATES_TOPIC = "/joint_states"`, `ACTIONS_TOPIC = "/actions"`, `REWARD_TOPIC = "/reward"`, `BODY_POSES_TOPIC = "/body_poses"`, plus `pub fn camera_topic(label: &str) -> String` and `pub fn recorder_schema() -> clankers_core::schema::RecorderSchema`. Owns the W1 `RecorderSchema` builder. |
| `crates/clankers-record/tests/multi_camera_roundtrip.rs` | Rust integration test that runs the recorder feature-gated `camera` system end-to-end with two labels and reads the MCAP back via the `mcap::reader` API, asserting both channels are present and that messages on each channel decode as raw `uint8`. |
| `python/tests/fixtures/two_camera.mcap` | Binary fixture (~10–15 KB) produced by a small recorder run; two camera channels (`/camera/front`, `/camera/wrist`), 3–5 frames each at 16×16 RGB. Committed via `git add -f` because `*.mcap` is gitignored. |
| `python/tests/fixtures/README.md` | One-page document explaining the fixture (channel list, frame count, dimensions) and the exact `cargo run -p clankers-record --example fixture_gen --features camera` command used to regenerate it. |
| `.github/workflows/nightly.yml` | New GitHub Actions workflow stub running `pytest tests/ -m slow` against the two slow-marked tests on a `schedule: cron: '0 6 * * *'` trigger. No matrix; Python 3.12 only. |

### MODIFY

| Path:line | Change |
|---|---|
| `crates/clankers-record/src/lib.rs:27-29` | Add `pub mod schema;` next to the existing `pub mod plugin; pub mod recorder; pub mod types;`. |
| `crates/clankers-record/src/lib.rs:35-41` | Extend `prelude` to re-export `schema::{camera_topic, recorder_schema, CAMERA_TOPIC_PREFIX, JOINT_STATES_TOPIC, ACTIONS_TOPIC, REWARD_TOPIC, BODY_POSES_TOPIC}`. |
| `crates/clankers-record/src/recorder.rs:18` | Update the doc table to cite the constants by name (`schema::CAMERA_TOPIC_PREFIX` etc.) rather than spelling the strings, so the doc cannot drift from the constants. |
| `crates/clankers-record/src/recorder.rs:556` | Replace `let topic = format!("/camera/{label}");` with `let topic = schema::camera_topic(label);`. Replace any remaining `"/joint_states"`, `"/actions"`, `"/reward"`, `"/body_poses"` string literals in the same file with the new constants (a grep before writing confirms the set; line numbers updated per actual hits). |
| `crates/clankers-record/Cargo.toml` | Confirm `clankers-core` is already a dependency (it is, transitively via `clankers_core::time::SimTime` at `recorder.rs:30`). No change expected; declared here to make the dependency edge explicit for the W1 consumer. |
| `python/clankers/mcap_loader.py:7-19` | Update the module docstring to describe the multi-camera channel model (`/camera/<label>` glob) and the new return key `images_by_camera`. |
| `python/clankers/mcap_loader.py:75` | Replace `CHANNEL_IMAGE = "/camera/image"` with `CAMERA_TOPIC_PREFIX = "/camera/"`. Keep `CHANNEL_IMAGE` as a class constant alias = `CAMERA_TOPIC_PREFIX + "image"` so external imports of `McapEpisodeLoader.CHANNEL_IMAGE` continue to work. |
| `python/clankers/mcap_loader.py:100-101` | Extend the docstring's return-key table to document `images_by_camera : dict[str, NDArray[np.uint8]] | None` (shape `(T, H, W, C)` per camera). |
| `python/clankers/mcap_loader.py:106-116` | Replace `raw["images"]: list[Any]` with `raw["images_by_camera"]: dict[str, list[bytes]]` and `image_meta: dict[str, int]` with `image_meta_by_camera: dict[str, dict[str, int]]`. |
| `python/clankers/mcap_loader.py:123-131` | Change the channel-metadata scan to match every channel whose `topic.startswith(self.CAMERA_TOPIC_PREFIX)`; extract the label as `topic[len(self.CAMERA_TOPIC_PREFIX):]`; index `image_meta_by_camera[label]`. |
| `python/clankers/mcap_loader.py:157-162` | Replace the `elif topic == self.CHANNEL_IMAGE:` branch with `elif topic.startswith(self.CAMERA_TOPIC_PREFIX):` and append to `raw["images_by_camera"][label]`. |
| `python/clankers/mcap_loader.py:164-173` | Add `"images_by_camera": None` to the `result` dict initialisation. |
| `python/clankers/mcap_loader.py:192-204` | Replace the single `raw["images"] and image_meta` branch with a loop over `raw["images_by_camera"]`; build a `dict[str, np.ndarray]`; assign to `result["images_by_camera"]`. **Back-compat:** if the keys are exactly `{"image"}`, also assign `result["images"] = result["images_by_camera"]["image"]`. |
| `python/clankers/mcap_loader.py:246-250` | Update the `to_sb3_replay_buffer()` image branch to prefer `data["images"]` when populated (back-compat path), otherwise pick the first camera key from `data["images_by_camera"]` (deterministic via `sorted(keys)[0]`) and document the choice. |
| `python/tests/test_mcap_loader.py` | Un-ignored from CI (see ci.yml line 69). Add `test_multi_camera_discovery`, `test_single_camera_back_compat`, `test_no_camera_returns_none`. Reuse the `two_camera.mcap` fixture. |
| `python/tests/test_mcap_augmentor.py` | Un-ignored from CI (see ci.yml line 70). No assertion changes — the augmentor today operates on `data["images"]`; the back-compat path makes the existing test pass unchanged. |
| `python/tests/test_dl_pipeline_e2e.py` | Add `pytest.mark.slow` at the top of the file (module-level). |
| `python/tests/test_trajectory_dataset.py` | Add `pytest.mark.slow` at the top of the file (module-level). |
| `python/pyproject.toml` | Register the `slow` mark under `[tool.pytest.ini_options].markers` so `pytest -m 'not slow'` doesn't warn about unknown marks. |
| `.github/workflows/ci.yml:67-70` | Drop `--ignore=tests/test_mcap_loader.py` and `--ignore=tests/test_mcap_augmentor.py`. Replace the remaining `--ignore=tests/test_dl_pipeline_e2e.py` and `--ignore=tests/test_trajectory_dataset.py` with `-m 'not slow'`. |

### DELETE

None. All existing code paths remain reachable; `CHANNEL_IMAGE` survives
as an alias for one release cycle (see §8 risk 2).

## 5. Checklist items

Each item is one focused commit at the ≤300-LOC ceiling.

### PR1 — Rust schema constants + recorder refactor + fixture + roundtrip test

- [ ] Add `crates/clankers-record/src/schema.rs` with the five topic
  constants and `camera_topic(label)` const-fn-shaped builder
  (regular `fn` because `format!` is not const). Add unit tests in an
  inline `#[cfg(test)] mod tests`: `camera_topic_concatenates_prefix_and_label`,
  `topic_constants_match_recorder_doc_table`.
- [ ] Add `recorder_schema() -> clankers_core::schema::RecorderSchema`
  builder that returns a `RecorderSchema` with one `FrameSchema` per
  fixed channel (`/joint_states`, `/actions`, `/reward`, `/body_poses`)
  plus per-camera entries discovered at call time from a
  `&[String]` of labels. Inline unit test:
  `recorder_schema_includes_all_fixed_channels`.
- [ ] Refactor `crates/clankers-record/src/recorder.rs` to consume the
  topic constants. Single behaviour-neutral commit: every `"/joint…"`,
  `"/actions"`, `"/reward"`, `"/body_poses"` string literal and the
  `format!("/camera/{label}")` at line 556 routed through `schema::*`.
  Existing `recorder_plugin_builds_without_panic` test must stay green.
- [ ] Re-export `schema::*` from the `crates/clankers-record/src/lib.rs`
  prelude.
- [ ] Add a minimal `clankers-record` example binary (`examples/fixture_gen.rs`)
  used solely to regenerate the two-camera MCAP fixture. ≤80 LOC.
  Documented in `python/tests/fixtures/README.md`.
- [ ] Commit the binary fixture `python/tests/fixtures/two_camera.mcap`
  via `git add -f` (the file is matched by `*.mcap` in `.gitignore`).
  Target size <20 KB. Two camera labels (`front`, `wrist`), 3 frames
  each at 16×16×3 = 768 bytes/frame ⇒ ~5 KB total payload + MCAP
  framing.
- [ ] Add `crates/clankers-record/tests/multi_camera_roundtrip.rs`
  exercising `recorder_writes_per_label_topics`: build a minimal Bevy
  `App` with `MinimalPlugins`, insert a synthetic
  `CameraFrameBuffers` with two labels, run one `app.update()`,
  reopen the MCAP via `mcap::reader::make_reader`, assert
  `summary.channels.values().map(|c| c.topic).collect::<HashSet<_>>()`
  contains both `/camera/front` and `/camera/wrist`.

### PR2 — Python multi-camera loader + un-ignore CI tests + nightly workflow

- [ ] Refactor `python/clankers/mcap_loader.py` channel-discovery logic
  to match the `/camera/*` glob: rename `CHANNEL_IMAGE` and add
  `CAMERA_TOPIC_PREFIX`; introduce `images_by_camera` accumulator;
  populate `result["images_by_camera"]`.
- [ ] Implement back-compat: when `set(images_by_camera.keys()) ==
  {"image"}`, also populate `result["images"]` with the same ndarray
  (no copy). Document in the load() docstring.
- [ ] Update `McapEpisodeLoader.to_sb3_replay_buffer()` and
  `convert_to_dataframe()` (and any other internal consumer surfaced by
  grep before writing) to handle the new shape. The default policy when
  multiple cameras are present is "pick `sorted(keys)[0]`" with a
  `warnings.warn(stacklevel=2)` notifying the caller — not an error.
- [ ] Add `python/tests/test_mcap_loader.py::test_multi_camera_discovery`,
  `::test_single_camera_back_compat`, `::test_no_camera_returns_none`.
  All three use fixtures committed in PR1 (the multi-camera case uses
  `two_camera.mcap`; the single-camera case writes a tiny one-camera
  MCAP via the `mcap` Python writer inside the test setup; the no-camera
  case reuses an existing joint-only fixture or writes one inline).
- [ ] Register the `slow` mark in `python/pyproject.toml` under
  `[tool.pytest.ini_options].markers`. Add `pytest.mark.slow` to
  `test_dl_pipeline_e2e.py` and `test_trajectory_dataset.py` at module
  scope.
- [ ] Update `.github/workflows/ci.yml`: remove the two
  `test_mcap_loader.py` / `test_mcap_augmentor.py` `--ignore` lines
  (lines 69–70); replace the two remaining ignores with `-m 'not slow'`
  so the slow tests are simply skipped on PR runs.
- [ ] Create `.github/workflows/nightly.yml` running
  `pytest tests/ -m slow --timeout=600` on a daily cron. Single Python
  matrix entry (3.12) on `ubuntu-latest`. Same `pip install` block as
  the PR-run `python` job.

## 6. Tests required before implementation (test-first)

All tests below are written and committed (red) in a tests-first
sub-step of each PR; the production code in the next sub-step turns
them green. Test placement follows
`.delegate/work/20260525-185618-workstream-plans/TASK.md`
§ "Test placement conventions".

### `crates/clankers-record/tests/multi_camera_roundtrip.rs`

- `recorder_writes_per_label_topics` — build a minimal `App` with
  `bevy::MinimalPlugins` + `SimTime` + `CameraFrameBuffers` populated
  with two labels (`"front"`, `"wrist"`) each backed by a 16×16×3
  `Vec<u8>`. Insert `RecordingConfig { output_path: tmp,
  record_joints: false, .. }`, add `RecorderPlugin`, `app.update()`
  once, then open the file with `mcap::reader::make_reader` and assert:
  `summary.channels.values().map(|c| c.topic).collect::<HashSet<_>>() ==
  {"/camera/front", "/camera/wrist"}`.
- `recorder_topic_constants_match_doc_table` — parse the
  `crates/clankers-record/src/recorder.rs` module docs (via
  `include_str!`) and assert each row of the topic-doc table appears as
  a const in `crates/clankers-record/src/schema.rs`. Pins the
  doc-vs-impl invariant.
- `camera_topic_label_round_trip` — `assert_eq!(camera_topic("front"),
  "/camera/front");` plus an `assert!(topic.starts_with(
  CAMERA_TOPIC_PREFIX));`. Pure-fn test, no Bevy.

### `python/tests/test_mcap_loader.py`

- `test_multi_camera_discovery` — `loader =
  McapEpisodeLoader("python/tests/fixtures/two_camera.mcap"); data =
  loader.load(); assert set(data["images_by_camera"].keys()) ==
  {"front", "wrist"}; assert data["images_by_camera"]["front"].shape ==
  (3, 16, 16, 3); assert data["images_by_camera"]["front"].dtype ==
  np.uint8`.
- `test_single_camera_back_compat` — write a temporary 1-camera MCAP
  with topic `/camera/image` (one frame, 8×8×3) using the `mcap`
  Python writer in test setup; load it; assert
  `data["images"] is not None and data["images"].shape == (1, 8, 8, 3)`
  AND `data["images_by_camera"] == {"image":
  data["images"]}` (object-identity via `is`).
- `test_no_camera_returns_none` — write a temporary joint-only MCAP
  with `/joint_states` only; load it; assert `data["images"] is None`
  and `data["images_by_camera"] == {}` (empty dict, **not** `None`).
- `test_multi_camera_sb3_replay_buffer_picks_first_camera_sorted` —
  load `two_camera.mcap`; call `loader.to_sb3_replay_buffer()`; with
  `pytest.warns(UserWarning, match="Multiple cameras")` asserted;
  verify the returned `observations.shape[1]` matches the channel
  count of the alphabetically-first camera (`"front"`).
- `test_loader_constants_exported` — `from clankers.mcap_loader import
  McapEpisodeLoader; assert McapEpisodeLoader.CAMERA_TOPIC_PREFIX ==
  "/camera/"; assert McapEpisodeLoader.CHANNEL_IMAGE == "/camera/image"`
  (back-compat alias survives).

### Inline `#[cfg(test)] mod tests` in `crates/clankers-record/src/schema.rs`

- `camera_topic_concatenates_prefix_and_label` —
  `assert_eq!(camera_topic("wrist"), "/camera/wrist");`
- `topic_constants_match_recorder_doc_table` — assert each constant's
  value matches the literal string the recorder docs promised, so a
  future refactor cannot silently rename the wire format.
- `recorder_schema_includes_all_fixed_channels` —
  `let s = recorder_schema(&["front".into(), "wrist".into()]);` assert
  channel names = `{"/joint_states", "/actions", "/reward",
  "/body_poses", "/camera/front", "/camera/wrist"}`.

All tests above must compile and fail when committed in the test-first
sub-step of each PR; the subsequent production commit turns them green.

## 7. Success criteria

Each criterion is checkable with a concrete command. All `cargo`
invocations use `-j 24` per `CLAUDE.md`.

- `cargo test -j 24 -p clankers-record --test multi_camera_roundtrip`
  passes (3/3 cases green).
- `cargo test -j 24 -p clankers-record --features camera` passes
  (the schema unit tests, the existing
  `recorder_plugin_builds_without_panic`, and
  `joint_frame_write_read_roundtrip` all green).
- `cargo clippy -j 24 -p clankers-record --all-targets --features camera
  -- -D warnings` passes.
- `pytest python/tests/test_mcap_loader.py -v` passes — invoked
  from CI without any `--ignore` argument.
- `pytest python/tests/test_mcap_augmentor.py -v` passes — same.
- `grep -n 'format!.*/camera/' crates/clankers-record/src/recorder.rs`
  returns zero hits (PowerShell equivalent:
  `Select-String -Path 'crates/clankers-record/src/recorder.rs'
  -Pattern 'format!.*/camera/'` produces no matches).
- `grep -cE -- '--ignore=tests/test_mcap' .github/workflows/ci.yml`
  returns `0`.
- `grep -cE -- '--ignore=tests/test_dl_pipeline_e2e' .github/workflows/ci.yml`
  returns `0` (replaced by `-m 'not slow'`).
- `.github/workflows/nightly.yml` exists and `gh workflow view nightly`
  shows the cron trigger.
- Workspace gate: `cargo test -j 24 --workspace --exclude
  clankers-examples` passes after PR1 and PR2.
- New-user acceptance: `python -c "from clankers.mcap_loader import
  McapEpisodeLoader; d = McapEpisodeLoader(
  'python/tests/fixtures/two_camera.mcap').load();
  print(sorted(d['images_by_camera']))"` prints `['front', 'wrist']`.

## 8. Risks & mitigations

1. **Binary fixture (`two_camera.mcap`) committed to git inflates
   repository size and is opaque in PR review.** The file is matched
   by `*.mcap` in `.gitignore` and must be added with `git add -f`.
   **Mitigation:** keep the fixture under 20 KB (3 frames × 2 cameras
   × 16×16×3 ≈ 4.6 KB payload + MCAP framing ≈ 10 KB total).
   Commit a `python/tests/fixtures/README.md` describing the
   regeneration command (the `examples/fixture_gen.rs` binary from
   PR1) so reviewers can rebuild and diff offline. If anyone ever
   wants to bump frame count or resolution, the README is the
   authoritative source — never edit the binary by hand.

2. **The back-compat `images` flat-ndarray key creates an ambiguous
   public API: two keys for the same data when one camera is present.**
   Long-term, callers should migrate to `images_by_camera`.
   **Mitigation:** leave both keys populated when exactly one
   `/camera/image` channel is present. Do **not** emit
   `DeprecationWarning` from this workstream — deprecation begins one
   release later, after downstream training scripts have had a chance
   to migrate. The `to_sb3_replay_buffer()` helper prefers `images`
   for now (zero behaviour change for existing pipelines) and
   transparently falls through to `sorted(images_by_camera)[0]` only
   for multi-camera datasets. Adding the deprecation ladder is
   explicitly a follow-up, not part of W6.

3. **`python/clankers/mcap_augmentor.py` may depend on the single-image
   assumption.** A grep before PR2 will confirm; if the augmentor
   directly indexes `data["images"]`, the back-compat key makes
   single-camera fixtures keep working, but multi-camera fixtures will
   trigger the same first-camera-picking behaviour as `to_sb3_replay_buffer`.
   **Mitigation:** scope check inside PR2. If `mcap_augmentor.py`
   accepts a `camera_label: str | None` constructor arg without surgery,
   add it there too. If the change exceeds 50 LOC, defer the augmentor
   multi-camera support to a follow-up and document in
   `python/tests/test_mcap_augmentor.py` that the augmentor operates on
   the back-compat `images` key only (matching today's behaviour).

4. **Slow tests promoted to `slow` mark may silently rot in nightly.**
   Without a check on the nightly workflow's result, the tests could
   start failing and nobody would notice.
   **Mitigation:** the nightly workflow stub created in PR2 includes a
   `notification-on-failure` step (GitHub Actions emits an email by
   default to the repo owner). A follow-up workstream (not W6) can
   wire Slack/Discord webhooks if the email channel proves
   insufficient. The stub is the minimum viable surface.

5. **Test-first discipline conflicts with the binary fixture
   ordering.** The Python tests cannot run before `two_camera.mcap`
   exists, but PR1 (Rust) ships the fixture only at the end of its
   commit sequence.
   **Mitigation:** PR1 commits the fixture *before* the Python work in
   PR2 begins. PR2's failing-first commit pulls the fixture from the
   merged PR1. The two PRs are sequential, not parallel, exactly
   matching `expected_implementation_prs: 2` in LOOPS.yaml.

## 9. PR breakdown

Exactly **2** commits, per LOOPS.yaml loop 6's
`expected_implementation_prs: 2` and the gate.

### PR1 — `feat(record): topic schema constants + 2-camera roundtrip`

**Scope summary:** Introduce the canonical topic constants and the
versioned `RecorderSchema` integration in `clankers-record`. Refactor
the recorder hot path to consume them. Commit a 2-camera binary fixture
and a Rust integration test that asserts the recorder writes one MCAP
channel per camera label.

**Files (diff estimate):**

- `crates/clankers-record/src/schema.rs` (+~110 LOC new).
- `crates/clankers-record/src/lib.rs` (+8 LOC: `pub mod schema;` and
  prelude re-exports).
- `crates/clankers-record/src/recorder.rs` (+~10 LOC, −~10 LOC:
  swap string literals for constants, update doc table to cite
  constants).
- `crates/clankers-record/examples/fixture_gen.rs` (+~70 LOC new):
  minimal binary that builds a 2-camera Bevy app and writes
  `python/tests/fixtures/two_camera.mcap`.
- `crates/clankers-record/tests/multi_camera_roundtrip.rs` (+~120 LOC
  new): three integration tests.
- `python/tests/fixtures/two_camera.mcap` (~10–15 KB binary, committed
  with `git add -f`).
- `python/tests/fixtures/README.md` (+~30 LOC new).

Total ≈ 250 LOC of source code + one binary fixture + one README.
Sub-commits keep each logical step under 300 LOC: (a) tests-first
(red) commit for the schema unit tests + roundtrip, (b) schema module
+ recorder refactor commit, (c) fixture generator + binary fixture +
README commit.

**Checklist items from §5 included:** all PR1 items
(schema.rs constants, recorder refactor, prelude re-export, fixture
generator binary, fixture commit, roundtrip test).

### PR2 — `feat(python): multi-camera mcap loader + un-ignore CI tests`

**Scope summary:** Migrate `python/clankers/mcap_loader.py` from a
single-topic to a glob-based channel discovery. Preserve back-compat
via the dual `images` / `images_by_camera` keys. Un-ignore the two
fast Python integration tests in CI; promote the two slow ones behind
a pytest `slow` mark; create the nightly workflow stub that runs the
slow tests on cron.

**Files (diff estimate):**

- `python/clankers/mcap_loader.py` (+~120 LOC, −~50 LOC: glob
  discovery, accumulator rewrite, back-compat populate,
  `to_sb3_replay_buffer` first-camera selection).
- `python/clankers/mcap_augmentor.py` (+~10 LOC if back-compat suffices;
  otherwise minimal `camera_label` constructor arg per §8 risk 3).
- `python/tests/test_mcap_loader.py` (+~140 LOC new tests).
- `python/tests/test_dl_pipeline_e2e.py` (+1 LOC:
  `pytestmark = pytest.mark.slow`).
- `python/tests/test_trajectory_dataset.py` (+1 LOC: same).
- `python/pyproject.toml` (+3 LOC: `[tool.pytest.ini_options].markers`
  entry for `slow`).
- `.github/workflows/ci.yml` (+4 LOC, −4 LOC: drop the two
  mcap-related `--ignore` lines; replace the other two ignores with
  `-m 'not slow'`).
- `.github/workflows/nightly.yml` (+~50 LOC new).

Total ≈ 300 LOC. Single PR but two logical commits inside:
(a) tests-first Python failures + un-ignore CI, (b) loader rewrite +
nightly workflow.

**Checklist items from §5 included:** all PR2 items (loader rewrite,
back-compat, `to_sb3_replay_buffer` update, new Python tests, `slow`
mark, CI un-ignore, nightly workflow).

After PR2: every grep target listed in §7 holds, the four CI-ignored
Python tests are either running on every PR (the two fast ones) or
running nightly (the two slow ones), and the recorder/loader topic
contract is asserted end-to-end by the two-camera roundtrip.
