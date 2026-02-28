# Clankers Engine State Report (2026-02-28)

## 1. Python-Rust Training Link
**Current State:**
The integration between the Rust simulation engine and the Python learning algorithms does not rely on direct FFI (like PyO3). Instead, it uses a custom TCP server protocol implemented in the `clankers-gym` crate.
- `clankers-gym` (Rust) spawns a headless TCP server that waits for connections and streams state.
- `clanker_gym` (Python) operates as a Gymnasium-compatible environment that connects to this TCP socket to send actions and receive observations.
**Distance from completion:** The link is already functioning well for basic environments. Moving forward, the next step would be handling parallel environments (Vectorized Envs) efficiently, potentially moving to Shared Memory (shmem) or zero-allocation IPC if network overhead becomes a bottleneck for large-scale RL training.

## 2. Quadruped Experiments: Example vs. Library Segregation
**Current State:**
The quadruped example (`examples/src/bin/quadruped_mpc*.rs`) currently houses significant core logic within `examples/src/mpc_control.rs` and `examples/src/quadruped_setup.rs`.

**What should be in the Library (`clankers-mpc`):**
- The `MpcLoopState`, `MpcStepResult`, and the overarching `compute_mpc_step` pipeline. This pipeline is largely generic across legged robots but contains hardcoded values (e.g., `STANCE_KP_JOINT`, `SWING_MAX_F`) that should be exposed cleanly as part of `MpcConfig` or `SwingConfig`.
- The Cartesian PD logic for swing legs and IK feedforward combinations.
- The `body_state_from_rapier` and contact detection functions, abstracted to operate on generic rigid bodies rather than Rapier specifically.

**What should remain in the Example:**
- The actual instantiation of `MpcConfig` with robot-specific gains (e.g., Quadruped hip abduction gains).
- Loading the specific target URDFs and mapping specific joint names to simulated entities.
- The simulation update systems that copy state into the MPC solver and apply computed `MotorCommand`s back to the physics motors.

## 3. Future Experiments and Features (Modular Sim-To-Real)
Drawing from current trends in Embodied AI and the `POTENTIAL_FEATURES.md` wishlist, the following roadmap is proposed for a robust, modular Sim-to-Real framework:
1. **Procedural Scene & Asset Randomization Engine:** Expanding `clankers-domain-rand` to randomize not just physics, but visual materials, lighting, and occlusions (crucial for training Vision-Language-Action policies like Octo or RT-1).
2. **Automated System Identification (SysId):** Building an optimization loop that ingests real-world log data (trajectories, torques) and automatically tunes the simulation's URDF parameters (mass, friction, damping) to minimize the sim-to-real gap.
3. **Hardware-In-The-Loop (HIL) Execution:** A synchronized real-time execution mode to test specific offline robot hardware components while sending constraints back against the simulated world acting as ground truth.
4. **Zero-Allocation Deterministic Logging:** Guaranteeing bit-for-bit replayability of crash states originating in the real world directly within the simulation engine for 1:1 bug reproduction.
5. **Soft-Body & Dense Tactile Interaction:** Essential for advanced manipulation tasks, moving beyond rigid URDFs into position-based dynamics and simulating high-res elastomeric touch sensors (e.g., GelSight).

## 4. Refactoring for Engine Swapping (Bevy & Rapier)
**Historical Context: Transition to Pure Rapier**
Initially, the project relied on `bevy_rapier3d` to bridge physics and the Bevy ECS. However, the codebase has since been refactored to use the pure `rapier3d` crate directly. This was a critical first step in decoupling the physics timeline from Bevy's rendering and ECS systems, allowing for headless deterministic execution. All mentions of `bevy_rapier` have been scrubbed from the codebase.

**Current Coupling:**
The `clankers-core` defines general traits like `Simulation` and `Sensor`, which is a good structural start. However, `clankers-physics` relies heavily on Bevy's ECS `Component` pattern (`Mass`, `SurfaceFriction`, `EndEffectorState`) to manage state, and the physics implementations tightly bind Rapier Contexts seamlessly into Bevy Systems (`plugin.rs` and `systems.rs`).

**Required Refactoring & API Design:**
To cleanly swap out Bevy (Visualization) or Rapier (Physics) for alternative engines without heavy downstream friction, the API interfaces need to be abstracted away from Bevy's `World` and `Entity` paradigms.

**Proposed Minimal Interface:**
1. **Abstracted Physics Backend Trait:**
   Instead of updating Bevy components inside a schedule and applying forces via Rapier handles, define a pure data interface leveraging a generic `BodyId`:
   ```rust
   pub trait PhysicsEngine {
       fn step(&mut self, dt: f64);
       fn get_body_state(&self, id: BodyId) -> BodyState;
       fn apply_joint_torque(&mut self, id: BodyId, dof: usize, torque: f64);
       fn get_active_contacts(&self, id: BodyId) -> Vec<ContactData>;
       fn randomize_domain_props(&mut self, id: BodyId, props: PhysicsProps);
   }
   ```

2. **Abstracted Rendering Backend Trait:**
   ```rust
   pub trait RenderEngine {
       fn load_scene(&mut self, urdf: &UrdfTree);
       fn update_transforms(&mut self, poses: &HashMap<BodyId, Pose>);
       fn capture_camera(&mut self, camera_id: CameraId) -> ImageBuffer;
   }
   ```

3. **Orchestrator Encapsulation (`clankers-sim`):**
   The core gym simulation loop should manage generic instances of `Box<dyn PhysicsEngine>` and `Box<dyn RenderEngine>`. Bevy can exist simply as one supported *implementation* of the `RenderEngine`, spinning up in an isolated thread. This ensures the pure numeric RL environment doesn't get bottlenecked by ECS scheduling overhead, and permits the framework to run completely headless or hot-swap physics solvers arbitrarily.
