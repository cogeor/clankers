# Loop 04: Add cartpole_policy_viz example binary for ONNX policy visualization

## Overview

Create a new example binary `cartpole_policy_viz` that loads a trained ONNX
policy and runs it in the same 3D visualization scene used by `pendulum_viz`.
Instead of teleop (keyboard) driving the joints, the `ClankersPolicyPlugin` and
`PolicyRunner` read observations from `ObservationBuffer` each frame and produce
actions via `OnnxPolicy::get_action`. A custom `apply_policy_action` system
copies the resulting action vector to `JointCommand` components so the actuator
and physics pipeline execute the policy's intent.

This requires:
1. Adding `clap` and enabling the `onnx` feature in `examples/Cargo.toml`.
2. Creating `examples/src/bin/cartpole_policy_viz.rs` with scene setup, policy
   loading, the action-application system, and visualization plugins.

## Tasks

### Task 1: Update examples/Cargo.toml dependencies

**Goal:** Add `clap` workspace dependency and enable the `onnx` feature on
`clankers-policy` so the example binary can parse CLI args and load ONNX models.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `examples/Cargo.toml` |

**Steps:**
1. Add `clap.workspace = true` to `[dependencies]`.
2. Change the `clankers-policy` line from:
   ```toml
   clankers-policy.workspace = true
   ```
   to:
   ```toml
   clankers-policy = { workspace = true, features = ["onnx"] }
   ```

**Verify:** `cargo check -p clankers-examples` compiles without errors (no
binary yet, just dependency resolution).

---

### Task 2: Create the cartpole_policy_viz binary

**Goal:** Create `examples/src/bin/cartpole_policy_viz.rs` that replicates the
`pendulum_viz` scene but replaces teleop control with ONNX policy inference.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `examples/src/bin/cartpole_policy_viz.rs` |

**Steps:**

1. **CLI struct** -- Define a `clap::Parser` struct with a single `--model`
   argument (type `PathBuf`, required) pointing to the `.onnx` file:
   ```rust
   use std::path::PathBuf;
   use clap::Parser;

   /// Cart-pole visualization driven by an ONNX policy.
   #[derive(Parser)]
   #[command(version, about)]
   struct Cli {
       /// Path to the ONNX policy model file.
       #[arg(long)]
       model: PathBuf,
   }
   ```

2. **Imports** -- Mirror the imports from `pendulum_viz.rs` but replace teleop
   imports with policy imports:
   ```rust
   use std::collections::HashMap;
   use bevy::prelude::*;
   use clankers_actuator::components::{JointCommand, JointState};
   use clankers_core::ClankersSet;
   use clankers_env::prelude::*;
   use clankers_examples::CARTPOLE_URDF;
   use clankers_physics::rapier::{bridge::register_robot, RapierBackend, RapierContext};
   use clankers_physics::ClankersPhysicsPlugin;
   use clankers_policy::prelude::*;
   use clankers_sim::SceneBuilder;
   use clankers_viz::ClankersVizPlugin;
   ```
   Note: no `clankers_teleop`, no `clankers_viz::input`, no `clankers_noise`,
   no `rand`/`rand_chacha` imports.

3. **Visual marker components** -- Copy verbatim from `pendulum_viz.rs`:
   `RailVisual`, `CartVisual`, `PivotVisual`, `PoleVisual`.

4. **`CartPoleJoints` resource** -- Same as `pendulum_viz.rs`:
   ```rust
   #[derive(Resource)]
   struct CartPoleJoints {
       cart: Entity,
       pole: Entity,
   }
   ```

5. **`spawn_robot_meshes` system** -- Copy verbatim from `pendulum_viz.rs`
   (lines 90-155). Spawns rail, cart, pivot, and pole mesh entities.

6. **Visual sync systems** -- Copy verbatim from `pendulum_viz.rs`:
   `sync_cart_visual`, `sync_pivot_visual`, `sync_pole_visual`.

7. **`apply_policy_action` system** -- This is the key new system. It reads the
   `PolicyRunner` resource and writes the action values to `JointCommand`
   components on the cart and pole entities:
   ```rust
   /// Copies PolicyRunner's current action to JointCommand components.
   ///
   /// The cartpole PPO policy produces a 2-element action:
   ///   action[0] -> cart_slide force (JointCommand on cart entity)
   ///   action[1] -> pole_hinge torque (JointCommand on pole entity)
   ///
   /// If the policy produces only 1 action (discrete cart force), only
   /// the cart joint is driven.
   #[allow(clippy::needless_pass_by_value)]
   fn apply_policy_action(
       runner: Res<PolicyRunner>,
       joints: Res<CartPoleJoints>,
       mut commands: Query<&mut JointCommand>,
   ) {
       let action = runner.action().as_slice();

       // Cart force (action index 0)
       if let Ok(mut cmd) = commands.get_mut(joints.cart) {
           cmd.value = action.first().copied().unwrap_or(0.0);
       }

       // Pole torque (action index 1), if present
       if action.len() > 1 {
           if let Ok(mut cmd) = commands.get_mut(joints.pole) {
               cmd.value = action.get(1).copied().unwrap_or(0.0);
           }
       }
   }
   ```

8. **`main` function** -- Assemble the application. Follow the numbered-step
   pattern from `pendulum_viz.rs` but replace teleop with policy:

   ```rust
   fn main() {
       let cli = Cli::parse();

       // 1. Load the ONNX policy
       let onnx_policy = OnnxPolicy::from_file(&cli.model)
           .unwrap_or_else(|e| panic!("failed to load ONNX model: {e}"));
       let action_dim = onnx_policy.action_dim();
       println!(
           "Loaded ONNX policy: obs_dim={}, action_dim={}",
           onnx_policy.obs_dim(),
           action_dim,
       );

       // 2. Parse URDF
       let model = clankers_urdf::parse_string(CARTPOLE_URDF)
           .expect("failed to parse cartpole URDF");

       // 3. Build scene
       let mut scene = SceneBuilder::new()
           .with_max_episode_steps(10_000)
           .with_robot(model.clone(), HashMap::new())
           .build();

       let cart = scene.robots["cartpole"]
           .joint_entity("cart_slide")
           .expect("missing cart_slide joint");
       let pole = scene.robots["cartpole"]
           .joint_entity("pole_hinge")
           .expect("missing pole_hinge joint");

       // 4. Add Rapier physics
       scene.app.add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

       // 5. Register robot with Rapier (fixed base)
       {
           let spawned = &scene.robots["cartpole"];
           let world = scene.app.world_mut();
           let mut ctx = world.remove_resource::<RapierContext>().unwrap();
           register_robot(&mut ctx, &model, spawned, world, true);
           world.insert_resource(ctx);
       }

       // 6. Register sensors (JointStateSensor fills ObservationBuffer
       //    with [cart_pos, cart_vel, pole_pos, pole_vel] = 4 obs)
       {
           let world = scene.app.world_mut();
           let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
           let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
           registry.register(
               Box::new(JointStateSensor::new(2)),
               &mut buffer,
           );
           world.insert_resource(buffer);
           world.insert_resource(registry);
       }

       // 7. Joint entity references
       scene.app.insert_resource(CartPoleJoints { cart, pole });

       // 8. PolicyRunner + ClankersPolicyPlugin
       let runner = PolicyRunner::new(Box::new(onnx_policy), action_dim);
       scene.app.insert_resource(runner);
       scene.app.add_plugins(ClankersPolicyPlugin);

       // 9. Windowed rendering
       scene.app.add_plugins(DefaultPlugins.set(WindowPlugin {
           primary_window: Some(Window {
               title: "Clankers - Cart-Pole Policy Viz".to_string(),
               resolution: (1280, 720).into(),
               ..default()
           }),
           ..default()
       }));

       // 10. Viz plugin (orbit camera + egui, NO teleop)
       scene.app.add_plugins(ClankersVizPlugin);

       // 11. Robot visual meshes
       scene.app.add_systems(Startup, spawn_robot_meshes);

       // 12. Visual sync: JointState -> mesh transforms
       scene.app.add_systems(
           Update,
           (sync_cart_visual, sync_pivot_visual, sync_pole_visual)
               .after(ClankersSet::Simulate),
       );

       // 13. Action applicator: PolicyRunner -> JointCommand
       //     Must run after Decide (policy has produced action)
       //     and before Act (actuator reads JointCommand).
       scene.app.add_systems(
           Update,
           apply_policy_action
               .after(ClankersSet::Decide)
               .before(ClankersSet::Act),
       );

       // 14. Start episode
       scene.app.world_mut().resource_mut::<Episode>().reset(None);

       println!("Cart-pole policy viz with Rapier physics");
       println!("  Scene camera: mouse (orbit/pan/zoom)");
       println!("  Joint control: ONNX policy (no keyboard)");
       println!("  Model: {}", cli.model.display());
       scene.app.run();
   }
   ```

**Key differences from `pendulum_viz.rs`:**

| Aspect | `pendulum_viz` | `cartpole_policy_viz` |
|--------|---------------|----------------------|
| Joint control | Teleop (keyboard) | ONNX policy |
| CLI args | None | `--model path.onnx` |
| Teleop plugin | `ClankersTeleopPlugin` | Not used |
| Keyboard map | `KeyboardTeleopMap` | Not used |
| Policy plugin | Not used | `ClankersPolicyPlugin` |
| Policy runner | Not used | `PolicyRunner` resource |
| Sensor noise | Yes (noisy slot) | No (clean obs only) |
| Action bridge | Teleop -> JointCommand | `apply_policy_action` system |
| Sensors | JointStateSensor + JointCommandSensor + noisy slot | JointStateSensor only |

**Verify:**
```bash
cargo build -p clankers-examples --bin cartpole_policy_viz --release
```

---

### Task 3: Smoke-test the binary

**Goal:** Verify the binary launches, parses the `--model` argument, and loads
the ONNX model without crashing (visual confirmation only; no automated test).

**Files:**
| Action | Path |
|--------|------|
| -- | (no file changes) |

**Steps:**
1. Run `cargo run -p clankers-examples --bin cartpole_policy_viz --release -- --model python/examples/cartpole_ppo.onnx`
2. Confirm the window opens, the 3D scene renders, and the policy drives the
   cart-pole (pole should attempt to balance).
3. Confirm no panics or error messages in the console.
4. Confirm `--help` output shows the `--model` argument.

**Verify:**
```bash
cargo run -p clankers-examples --bin cartpole_policy_viz -- --help
```
Should print usage with `--model <MODEL>` argument.

## Acceptance Criteria

- [ ] `examples/Cargo.toml` has `clap.workspace = true` in dependencies
- [ ] `examples/Cargo.toml` has `clankers-policy = { workspace = true, features = ["onnx"] }`
- [ ] `examples/src/bin/cartpole_policy_viz.rs` exists and compiles
- [ ] The binary accepts `--model <path>` via clap
- [ ] `OnnxPolicy::from_file` is called to load the model
- [ ] `PolicyRunner` resource is inserted with the loaded `OnnxPolicy`
- [ ] `ClankersPolicyPlugin` is added (drives `policy_decide_system`)
- [ ] `apply_policy_action` system bridges `PolicyRunner.action()` to `JointCommand` components
- [ ] `JointStateSensor` is registered so `ObservationBuffer` has the 4 obs the model expects
- [ ] No teleop plugin or keyboard mappings are present
- [ ] `ClankersVizPlugin` provides orbit camera and egui panel
- [ ] `cargo build -p clankers-examples --bin cartpole_policy_viz --release` succeeds
