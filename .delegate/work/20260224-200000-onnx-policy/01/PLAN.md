# Loop 01: Add ort workspace dependency and onnx feature flag to clankers-policy

## Overview

This loop adds the `ort` ONNX Runtime crate as an optional workspace dependency and introduces the first feature flag (`onnx`) to the `clankers-policy` crate. It also wires up `serde_json` (already a workspace dep) into `clankers-policy` for future metadata parsing. No functional code is added -- only Cargo manifest changes. The goal is to validate that the workspace builds cleanly both with and without the new feature.

## Tasks

### Task 1: Add `ort` as an optional workspace dependency

**Goal:** Register `ort` v2 in the root workspace dependency table so all crates can reference it uniformly.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `Cargo.toml` (workspace root) |

**Steps:**
1. Open `C:\Users\costa\src\clankers\Cargo.toml`.
2. In the `[workspace.dependencies]` section, after the existing external dependencies block (after line 44, `rapier3d = "0.32"`), add:
   ```toml
   ort = { version = "2.0.0-rc.11", default-features = false }
   ```
   Using `default-features = false` at the workspace level keeps the footprint minimal; individual crates can enable features they need.

**Verify:** The file parses correctly: `cargo metadata --no-deps --format-version 1 > /dev/null` exits 0.

---

### Task 2: Add `[features]` section and optional dependencies to `clankers-policy`

**Goal:** Introduce the `onnx` feature flag in `clankers-policy` that pulls in `ort`, and add `serde_json` as a regular dependency (needed by the ONNX metadata parser in Loop 02).

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `crates/clankers-policy/Cargo.toml` |

**Steps:**
1. Open `C:\Users\costa\src\clankers\crates\clankers-policy\Cargo.toml`.
2. After the `[dependencies]` block (after line 16, `rand_chacha.workspace = true`), add `ort` as an optional dep and `serde_json` as a regular dep:
   ```toml
   ort = { workspace = true, optional = true }
   serde_json.workspace = true
   ```
3. Before the `[dev-dependencies]` section (before line 18), insert a `[features]` section:
   ```toml
   [features]
   onnx = ["dep:ort"]
   ```

The resulting `crates/clankers-policy/Cargo.toml` should look like:

```toml
[package]
name = "clankers-policy"
description = "Policy implementations and inference runner for Clankers"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
rust-version.workspace = true

[dependencies]
clankers-core.workspace = true
clankers-env.workspace = true
clankers-actuator.workspace = true
bevy.workspace = true
rand.workspace = true
rand_chacha.workspace = true
ort = { workspace = true, optional = true }
serde_json.workspace = true

[features]
onnx = ["dep:ort"]

[dev-dependencies]
clankers-domain-rand.workspace = true

[lints]
workspace = true
```

**Verify:** `cargo metadata --no-deps -p clankers-policy --format-version 1 > /dev/null` exits 0.

---

### Task 3: Verify workspace builds without the `onnx` feature

**Goal:** Confirm the default build (no `onnx` feature) compiles cleanly across the entire workspace, meaning no existing code is broken.

**Files:**
| Action | Path |
|--------|------|
| -- | (no file changes) |

**Steps:**
1. Run `cargo check` from the workspace root.
2. Confirm exit code 0 and no errors.
3. Run `cargo test --no-run` to verify all test targets still compile.

**Verify:** `cargo check` and `cargo test --no-run` both exit 0.

---

### Task 4: Verify `clankers-policy` builds with the `onnx` feature enabled

**Goal:** Confirm that enabling the `onnx` feature pulls in `ort` and the crate compiles (even though no code uses it yet).

**Files:**
| Action | Path |
|--------|------|
| -- | (no file changes) |

**Steps:**
1. Run `cargo check -p clankers-policy --features onnx`.
2. Confirm exit code 0 and no errors.
3. If `ort` has linker or download issues (it fetches ONNX Runtime binaries on first build), troubleshoot by checking environment variables or network access.

**Verify:** `cargo check -p clankers-policy --features onnx` exits 0.

## Acceptance Criteria

- [ ] `ort = { version = "2.0.0-rc.11", default-features = false }` appears in root `Cargo.toml` under `[workspace.dependencies]`
- [ ] `crates/clankers-policy/Cargo.toml` has `[features]` section with `onnx = ["dep:ort"]`
- [ ] `crates/clankers-policy/Cargo.toml` has `ort` as an optional dependency via workspace
- [ ] `crates/clankers-policy/Cargo.toml` has `serde_json` as a dependency via workspace
- [ ] `cargo check` succeeds without `onnx` feature (no regressions)
- [ ] `cargo check -p clankers-policy --features onnx` succeeds
- [ ] All existing tests still pass (`cargo test`)
