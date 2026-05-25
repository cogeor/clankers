# Workstream 4 — Protocol parity (docs match wire format; image observations on Reset)

## 1. Goal

Fix the two protocol-boundary contract bugs identified in the codebase
quality report — finding #5 (`protocol.rs` documents newline-delimited JSON
while the implementation uses a 4-byte little-endian length-prefixed frame)
and finding #2 (image observations ship on `Step` only, never on `Reset`,
violating the Gymnasium space contract) — by promoting `ObsEncoding` to a
typed `EncodedObservation` enum applied uniformly across `Reset` and `Step`,
rewriting the protocol module docs to match the wire reality, and replacing
the three `assert self._sock is not None` sites in the Python client with
explicit `ProtocolError` raises plus schema-aware observation validation.

## 2. Why this workstream, why this order

Two report findings drive this workstream:

- **Finding #2** (quality report `notes/clankers_codebase_quality_report_2026-05-25.md`)
  "Image Observation Reset Violates Gymnasium Space Contract" — image envs
  declare a `uint8` Box space but reset returns a flat float JSON
  observation. Vision wrappers (SB3, Gymnasium) reject the env at the first
  reset because `obs.shape` and `obs.dtype` disagree with the declared
  space.
- **Finding #5** (same report) "Protocol Documentation Contradicts
  Implementation" — `crates/clankers-gym/src/protocol.rs:11` reads:

  > `//! All messages are newline-delimited JSON.`

  but the Rust framing helper and the Python client both use 4-byte
  little-endian `u32` length prefixes (see `python/clankers/client.py:147`
  in `recv_binary_frame` and the matching `_send_raw` at line 177).
  Third-party clients implementing the documented protocol will fail at
  the first byte.

**Dependency edges:**

- **Depends on W1** — the Python client validates that the returned
  observation matches the negotiated `ObservationSchema` (from W1). Without
  a typed `ObservationSchema` we can only assert `len()`, which is what
  today's wrappers do and exactly why finding #2 was not caught.
- **Depends on W3** — the new uniform encoder takes
  `&ObservationView` (from W3's zero-copy view types) to avoid double-
  materialising the observation when we already need the bytes for the
  binary frame. Today's `Response::from_step_binary` calls
  `result.observation.as_slice()` which panics on non-continuous variants
  (the very thing W3 fixes).
- **Unblocks W7** — the binary batch protocol (W7 PR2) extends the same
  `EncodedObservation` enum with a `Batch { num_envs, EncodedObservation }`
  variant. Without uniform reset/step encoding W7 ends up shipping yet
  another parallel decoder.

The order — W4 after W1+W3, before W7 — is forced: define the typed
observation contract once, decode it consistently in JSON+binary, then add
the parallel batch frame.

## 3. Out of scope

The following look adjacent but are explicitly **not** done in W4:

- **No binary batch protocol.** `BatchReset` / `BatchStep` continue to
  carry JSON `Vec<Observation>`. Binary batches land in W7 PR2 on top of
  this workstream's `EncodedObservation` enum.
- **No `ObservationSchema` design.** W1 owns the schema type. W4 only
  consumes it for client-side shape/dtype validation.
- **No new view types.** `ObservationView` / `ActionView` come from W3.
  W4 calls `encode_observation(&view, &schema)` and trusts the view API.
- **No MCAP topic-schema changes.** Camera topic parity (`/camera/{label}`
  vs `/camera/image`) is finding #3 and lives in W6.
- **No new transport.** Wire framing stays 4-byte LE length-prefixed; only
  the doc comment, the reset payload encoding, and the Python error type
  change. `Request` enum is unchanged.
- **No protocol version bump beyond a minor.** Servers continue to accept
  `protocol_version: "1.0.0"` clients; we ship `1.1.0` so that clients
  understanding `EncodedObservation::RawU8Image` on reset can negotiate it
  via capabilities (`image_on_reset: bool`).

## 4. Files to change

### NEW

| Action | Path | Purpose |
|--------|------|---------|
| NEW | `python/clankers/_errors.py` | Define `ProtocolError` exception class; re-exported from `python/clankers/__init__.py` and from `python/clankers/client.py`. |
| NEW | `crates/clankers-gym/src/encoding.rs` | New module owning `EncodedObservation`, `ImageLayout`, and the `encode_observation(view, schema)` helper used by both reset and step handlers. |
| NEW | `crates/clankers-gym/tests/protocol_image_reset.rs` | Rust integration test `protocol_image_reset_returns_raw_u8`. |
| NEW | `python/tests/test_image_reset_matches_step_shape.py` | Python test `test_image_reset_matches_step_shape`. |
| NEW | `python/tests/test_client_protocol_error.py` | Python test `test_client_raises_protocol_error_when_not_connected`. |

### MODIFY

| Action | Path | Lines / sites | Change |
|--------|------|---------------|--------|
| MODIFY | `crates/clankers-gym/src/protocol.rs` | `:1–11` (module doc), `:151–155` (`Response::Reset`), `:207–244` (`Response::from_reset` / `from_step_binary`), `:276–299` (`ObsEncoding` enum) | Rewrite module doc with the verbatim wire format and an annotated hex example. Promote `ObsEncoding` → `EncodedObservation` (re-export `ObsEncoding` as deprecated alias for one minor). Add `Option<EncodedObservation>` to `Response::Reset`. |
| MODIFY | `crates/clankers-gym/src/server.rs` | `:271` (`Request::Reset` handler), `:272–302` (`Request::Step` handler) | Both arms call the shared `encode_observation(view, schema, session.binary_obs)` helper. Reset gains the same `if binary_obs && Image { … } => return (Response::Reset { encoded: Some(RawU8Image), … }, Some(pixel_bytes))` branch that step has at `:274–300` today. |
| MODIFY | `crates/clankers-gym/src/lib.rs` | module declaration | `pub mod encoding;` and `pub use encoding::{EncodedObservation, ImageLayout};`. |
| MODIFY | `python/clankers/client.py` | `:179`, `:189`, `:203` | Replace the three sites (verbatim today): `assert self._sock is not None` → `if self._sock is None: raise ProtocolError("not connected")`. Import `ProtocolError` from the new `_errors.py` module (currently defined inline at `:30`); the inline definition becomes a re-export. |
| MODIFY | `python/clankers/client.py` | `:119–145` (`send`), `:13` (docstring) | Update docstring to describe the reset binary path. The existing `obs_encoding` branch in `send()` already decodes RawU8 on any response — once the server emits it on reset, this code path activates without changes (just needs the docstring update + the explicit `ProtocolError` import). |
| MODIFY | `python/clankers/env.py` | `:49–65` (`reset`), `:67–92` (`step`) | After every response, call `_validate_obs(resp, self.observation_space)` which raises `ProtocolError` on shape/dtype mismatch. Image-mode `reset()` returns the decoded `resp["_image_obs"]` instead of `resp["observation"]["data"]`. |
| MODIFY | `python/clankers/__init__.py` | top of file | Re-export `ProtocolError` from `_errors`. |

### Verbatim quotes

The misleading line in protocol.rs:

```text
crates/clankers-gym/src/protocol.rs:11
//! All messages are newline-delimited JSON.
```

The three offending assertions in `python/clankers/client.py`:

```python
# line 179, in _send_raw
assert self._sock is not None
# line 189, in _recv_raw
assert self._sock is not None
# line 203, in _recv_exact
assert self._sock is not None
```

These three are replaced verbatim with:

```python
if self._sock is None:
    raise ProtocolError("not connected")
```

## 5. Checklist items

Each item is atomic, ≤300 LOC, and lands as part of one of the two PRs in
section 9.

PR1 (Rust):

- [ ] Define `ImageLayout` enum (`Hwc`, `Chw`) with `Serialize + Deserialize`
      in `crates/clankers-gym/src/encoding.rs`.
- [ ] Define `EncodedObservation` enum with variants `FlatF32(Vec<f32>)`,
      `RawU8Image { width: u32, height: u32, channels: u8, layout: ImageLayout, payload: Vec<u8> }`,
      `Dict(BTreeMap<String, EncodedObservation>)`. `#[serde(tag = "type")]`,
      `RawU8Image.payload` marked `#[serde(skip)]` so JSON carries only the
      header — bytes follow as a separate length-prefixed binary frame.
- [ ] Implement `encode_observation(view: &ObservationView, schema: &ObservationSchema, binary: bool) -> (EncodedObservation, Option<Vec<u8>>)`.
      Returns `(FlatF32, None)` for continuous; `(RawU8Image { payload: vec![] }, Some(bytes))` for image with `binary == true`; falls back to flat float for image with `binary == false`.
- [ ] Add `Option<EncodedObservation>` field `obs_encoding` to
      `Response::Reset` (was: bare `Observation` only). Mirror existing
      `Response::Step` shape.
- [ ] Update `Response::from_reset` and add `Response::from_reset_binary`
      paired with the existing `from_step` / `from_step_binary`.
- [ ] Refactor `crates/clankers-gym/src/server.rs:271` `Request::Reset` arm:
      use the shared `encode_observation` helper instead of bare
      `from_reset`. Mirror the `if session.binary_obs && Image { … }` branch
      from `:274–300`.
- [ ] Refactor `Request::Step` arm to call the same helper (deduplicates
      the `pixel_bytes` conversion).
- [ ] Rewrite `crates/clankers-gym/src/protocol.rs:1–11` module doc:
      explicit 4-byte little-endian length prefix, JSON or binary payload,
      hex frame example with annotation, link to the new
      `EncodedObservation` enum doc. Quote the wire bytes for a Reset
      response with a 64×64×3 image.
- [ ] Mark `pub use protocol::ObsEncoding;` `#[deprecated(note = "Use EncodedObservation")]` for one minor version.
- [ ] Add doctest `protocol_doc_matches_wire_format` inside the rewritten
      module doc: constructs the documented hex frame, hands it to the
      decoder, asserts decode succeeds and yields the expected variant.
- [ ] Add integration test `protocol_image_reset_returns_raw_u8` in
      `crates/clankers-gym/tests/protocol_image_reset.rs`.

PR2 (Python):

- [ ] Create `python/clankers/_errors.py` with `class ProtocolError(Exception)`.
      Re-export from `python/clankers/__init__.py` and `python/clankers/client.py`.
- [ ] Replace `python/clankers/client.py:179` `assert self._sock is not None`
      with `if self._sock is None: raise ProtocolError("not connected: call connect() first")`.
- [ ] Replace `python/clankers/client.py:189` (same pattern).
- [ ] Replace `python/clankers/client.py:203` (same pattern).
- [ ] Add `_validate_obs(resp: dict, space: Box | Discrete | Dict) -> None`
      in `python/clankers/env.py`. Checks: response carries an `observation`
      OR `_image_obs` field; ndarray shape matches `space.shape`; dtype
      matches `space.dtype`. Raises `ProtocolError` on mismatch.
- [ ] `ClankerEnv.reset()` calls `_validate_obs` after every reset; if
      `_image_obs` is present returns that instead of the flat sentinel.
- [ ] `ClankerEnv.step()` calls `_validate_obs` (gated by a one-shot flag —
      validate only once per episode to keep the hot path cheap).
- [ ] Cache the negotiated `ObservationSchema` at connect time on
      `ClankerEnv._schema`; `_validate_obs` reads from there.
- [ ] Add `test_image_reset_matches_step_shape` in
      `python/tests/test_image_reset_matches_step_shape.py`.
- [ ] Add `test_client_raises_protocol_error_when_not_connected` in
      `python/tests/test_client_protocol_error.py`.

## 6. Tests required before implementation

The two tests named verbatim in `LOOPS.yaml`'s gate plus two supporting
tests. All four MUST be authored and committed before the implementation
checklist items they cover.

| Test | Path | Assertion shape |
|------|------|-----------------|
| `protocol_image_reset_returns_raw_u8` | `crates/clankers-gym/tests/protocol_image_reset.rs` | Spin up a `GymServer` configured with an image observation scenario (64×64×3, `ObservationSpace::Image`). Connect a synthetic client; complete the Init handshake with `binary_obs: true` capability. Send `Request::Reset { seed: Some(0) }`. Assert: the response is `Response::Reset { obs_encoding: Some(EncodedObservation::RawU8Image { width: 64, height: 64, channels: 3, .. }), .. }` AND a follow-up binary frame is on the wire with exactly `64 * 64 * 3` bytes. |
| `test_image_reset_matches_step_shape` | `python/tests/test_image_reset_matches_step_shape.py` | Connect a `ClankerEnv` to an in-process server with an image scenario. Call `env.reset()`. Call `env.step(env.action_space.sample())`. Assert: `obs_reset.shape == (64, 64, 3)`, `obs_reset.dtype == np.uint8`, AND `obs_reset.shape == obs_step.shape`, AND `obs_reset.dtype == obs_step.dtype`. |
| `protocol_doc_matches_wire_format` | doctest inside `crates/clankers-gym/src/protocol.rs` module doc | Construct the byte sequence shown in the rewritten module doc's hex example (Reset request frame + Reset response frame + RawU8 binary frame). Feed it to `read_message` / `read_binary_frame`. Assert the decoder produces the documented variants. Failure here means doc and impl drifted. |
| `test_client_raises_protocol_error_when_not_connected` | `python/tests/test_client_protocol_error.py` | Construct a `GymClient` without calling `connect()`. Call `client.send({"type": "ping", "timestamp": 0})`. Assert `pytest.raises(ProtocolError, match="not connected")`. |

Test-data fixture path: none new — both tests construct their own in-process
server using the existing `GymServer::new` builder.

## 7. Success criteria

Each criterion is checkable with a concrete command. CLAUDE.md mandates
`-j 24` for every cargo invocation:

- `cargo test -j 24 -p clankers-gym --test protocol_image_reset` exits 0
  and reports `1 passed`.
- `cargo test -j 24 -p clankers-gym --doc protocol` exits 0 and reports
  the new `protocol_doc_matches_wire_format` doctest passing.
- `pytest -q python/tests/test_image_reset_matches_step_shape.py` exits 0.
- `pytest -q python/tests/test_client_protocol_error.py` exits 0.
- `cargo doc -j 24 -p clankers-gym --no-deps` succeeds and the rendered
  HTML for `clankers_gym::protocol` contains the string
  `"4-byte little-endian"` and a `<pre>` block with the hex example.
- `grep -n 'assert self._sock is not None' python/clankers/` returns 0
  matches.
- `grep -rn 'newline-delimited' crates/clankers-gym/src/` returns 0
  matches.
- `cargo clippy -j 24 -p clankers-gym --all-targets -- -D warnings` exits
  0 (no clippy regression from the new enum).

## 8. Risks & mitigations

- **Risk:** Third-party clients written against the documented (wrong)
  newline-delimited framing exist in the wild. Fixing the doc could be
  read as a breaking change.
  **Mitigation:** Bump protocol version to `1.1.0` (minor — no client
  needs to change for the doc fix because the wire was never newline-
  delimited). Update `protocol.rs:306` `PROTOCOL_VERSION` and add a
  `CHANGELOG.md` entry under `crates/clankers-gym/`. The hex example in
  the rewritten doc explicitly notes "wire format unchanged since 1.0.0;
  this is a documentation correction."

- **Risk:** Large images on Reset double the bandwidth at episode start
  (a 1024×1024×3 RGB frame is 3 MiB; sending it on every reset matters
  for high-throughput vision training).
  **Mitigation:** Make image-on-reset opt-in via the `image_on_reset:
  bool` capability in the Init handshake. Default `true` for
  `ObservationSpace::Image` envs (the bug fix), `false` otherwise.
  Clients that need the old reset-skip behaviour can pass
  `image_on_reset: false`. Document this trade-off in the new module doc.

- **Risk:** Schema-validation overhead on Python `step()` hot path.
  Per-step `np.ndarray.shape` and `dtype` comparison is cheap (<1µs) but
  not free over millions of steps.
  **Mitigation:** Validate once per episode (first step after each
  reset). Cache the negotiated `ObservationSchema` on
  `ClankerEnv._schema` at connect time. Add a `_validated_this_episode:
  bool` flag reset to `False` in `reset()`. Document in code comments
  that this is a contract check, not a runtime safeguard, so once-per-
  episode is sufficient.

- **Risk:** The Python `_image_obs` field is currently a backward-compat
  hack (the docstring at `client.py:127` says "the original `observation`
  sentinel field is still present for backward compatibility but its
  data should be ignored"). Reset now also produces `_image_obs`,
  meaning code paths that unconditionally read `resp["observation"]["data"]`
  break.
  **Mitigation:** Audit `python/clankers/` for `resp["observation"]`
  reads (currently `env.py:63` and `env.py:88`). Both sites move into
  the new `_validate_obs` helper which prefers `_image_obs` when present.
  Add a regression test that constructs a flat-obs env (cartpole) and
  asserts the legacy path still works.

## 9. PR breakdown

Exactly **2 commits** (matches `LOOPS.yaml` `expected_implementation_prs: 2`).

### PR1 — Rust: `EncodedObservation` + uniform reset/step encoding + protocol.rs doc rewrite

Touches Rust only. Approximately **350 LOC diff**.

- New `crates/clankers-gym/src/encoding.rs` (~120 LOC: `EncodedObservation`,
  `ImageLayout`, `encode_observation` helper).
- `crates/clankers-gym/src/protocol.rs` module doc rewrite (~40 LOC).
- `Response::Reset` gains `obs_encoding: Option<EncodedObservation>`
  field; `from_reset` / `from_reset_binary` constructors (~30 LOC).
- `crates/clankers-gym/src/server.rs` reset/step handlers refactored to
  call the shared helper (~50 LOC; net deletion of duplicated branch).
- Doctest `protocol_doc_matches_wire_format` in the module doc (~20 LOC).
- `crates/clankers-gym/tests/protocol_image_reset.rs` new integration test
  (~80 LOC including in-process server scaffolding).
- Deprecation of `ObsEncoding` re-export (~10 LOC).
- Bump `PROTOCOL_VERSION` to `1.1.0` and add `crates/clankers-gym/CHANGELOG.md` entry (~10 LOC).

Commit message:
`feat(gym): encode observations uniformly on reset and step; fix protocol docs`

### PR2 — Python: `ProtocolError` + schema validation + assert replacements

Touches Python only. Approximately **150 LOC diff**.

- `python/clankers/_errors.py` new (~10 LOC).
- `python/clankers/client.py:179,189,203` — three `assert` → `raise`
  conversions plus import (~10 LOC net).
- `python/clankers/client.py` send/recv docstring updates to describe the
  reset binary path (~10 LOC).
- `python/clankers/env.py` `_validate_obs` helper + once-per-episode flag
  + cached `_schema` + image-aware reset path (~60 LOC).
- `python/clankers/__init__.py` re-export (~3 LOC).
- `python/tests/test_image_reset_matches_step_shape.py` new test (~40 LOC).
- `python/tests/test_client_protocol_error.py` new test (~20 LOC).

Commit message:
`feat(client): raise ProtocolError instead of assert; validate obs on reset and step`
