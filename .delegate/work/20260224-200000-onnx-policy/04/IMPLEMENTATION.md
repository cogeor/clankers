# Implementation Log

## Task 1: Update examples/Cargo.toml dependencies

Completed: 2026-02-24T20:30:00Z

### Changes

- `examples/Cargo.toml`: Added `clap.workspace = true` to `[dependencies]` section. Changed `clankers-policy.workspace = true` to `clankers-policy = { workspace = true, features = ["onnx"] }` to enable ONNX runtime support.

### Verification

- [x] `clap.workspace = true` present in dependencies
- [x] `clankers-policy = { workspace = true, features = ["onnx"] }` present in dependencies

---

## Task 2: Create the cartpole_policy_viz binary

Completed: 2026-02-24T20:32:00Z

### Changes

- `examples/src/bin/cartpole_policy_viz.rs`: Created new binary that replicates the `pendulum_viz` scene but replaces teleop control with ONNX policy inference. Key elements:
  - `Cli` struct with `--model` argument via `clap::Parser`
  - Visual marker components (`RailVisual`, `CartVisual`, `PivotVisual`, `PoleVisual`) copied verbatim from `pendulum_viz.rs`
  - `CartPoleJoints` resource for joint entity references
  - `spawn_robot_meshes` startup system copied verbatim from `pendulum_viz.rs`
  - Visual sync systems (`sync_cart_visual`, `sync_pivot_visual`, `sync_pole_visual`) copied verbatim from `pendulum_viz.rs`
  - New `apply_policy_action` system that bridges `PolicyRunner.action()` to `JointCommand` components on cart and pole entities
  - `main()` assembles: ONNX policy loading, URDF parsing, scene building, Rapier physics, sensor registration (`JointStateSensor` only), `PolicyRunner` + `ClankersPolicyPlugin`, windowed rendering, `ClankersVizPlugin`, and episode start
  - No teleop plugin, no keyboard mappings, no sensor noise -- clean policy-only control

### Verification

- [x] File exists at `examples/src/bin/cartpole_policy_viz.rs`
- [x] `OnnxPolicy::from_file` called to load model
- [x] `PolicyRunner` resource inserted with loaded `OnnxPolicy`
- [x] `ClankersPolicyPlugin` added
- [x] `apply_policy_action` system bridges `PolicyRunner.action()` to `JointCommand`
- [x] `JointStateSensor` registered for observation buffer
- [x] No teleop plugin or keyboard mappings present
- [x] `ClankersVizPlugin` provides orbit camera and egui panel

---

## Task 3: Smoke-test the binary

Completed: 2026-02-24T20:35:00Z

### Changes

- (no file changes)

### Verification

- [x] `cargo build -p clankers-examples --bin cartpole_policy_viz --release`: Succeeded (40.36s)
- [x] `--help` output shows `--model <MODEL>` argument: Confirmed
- [x] Binary name shown as `cartpole_policy_viz.exe` in usage output

### Notes

Build completed successfully on first attempt with no compilation errors or warnings. The `--help` output correctly displays:

```
Visualize a trained ONNX policy on the cart-pole

Usage: cartpole_policy_viz.exe --model <MODEL>

Options:
      --model <MODEL>  Path to the ONNX policy model file
  -h, --help           Print help
```

Full visual smoke test (launching the window with a real ONNX model) was not performed because it requires a GPU display context and a pre-trained model file. The compilation and CLI parsing verification confirm the binary is structurally correct.

---
