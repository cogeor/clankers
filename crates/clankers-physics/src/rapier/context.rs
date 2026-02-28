//! Bevy resource wrapping all Rapier3D physics pipeline state.

use std::collections::HashMap;

use bevy::prelude::{Entity, Quat, Resource, Vec3};
use rapier3d::prelude::{
    CCDSolver, ColliderSet, DefaultBroadPhase, ImpulseJointHandle, ImpulseJointSet,
    IntegrationParameters, IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline,
    RigidBodyHandle, RigidBodySet,
};

// ---------------------------------------------------------------------------
// JointInfo
// ---------------------------------------------------------------------------

/// Per-joint metadata stored alongside the rapier handle.
pub struct JointInfo {
    /// Rapier handle for the parent link rigid body.
    pub parent_body: RigidBodyHandle,
    /// Rapier handle for the child link rigid body.
    pub child_body: RigidBodyHandle,
    /// Joint axis as a glam Vec3 (unit direction).
    pub axis: Vec3,
    /// Whether this is a prismatic joint (vs revolute).
    pub is_prismatic: bool,
}

// ---------------------------------------------------------------------------
// RapierContext
// ---------------------------------------------------------------------------

/// All rapier state in a single Bevy resource.
///
/// `PhysicsPipeline::step()` requires mutable access to every set
/// simultaneously, so they must all live together.
#[derive(Resource)]
pub struct RapierContext {
    // -- Rapier sets --
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,

    // -- Pipeline objects --
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub ccd_solver: CCDSolver,

    // -- Parameters --
    pub integration_parameters: IntegrationParameters,
    pub gravity: Vec3,
    /// Number of physics substeps per control frame.
    pub substeps: usize,

    // -- Entity ↔ handle mappings --
    /// Joint ECS entity → ImpulseJointHandle.
    pub joint_handles: HashMap<Entity, ImpulseJointHandle>,
    /// Joint ECS entity → joint metadata.
    pub joint_info: HashMap<Entity, JointInfo>,
    /// Link name → RigidBodyHandle.
    pub body_handles: HashMap<String, RigidBodyHandle>,

    // -- Initial state for reset --
    /// Initial body translations, stored after register_robot.
    pub initial_body_positions: HashMap<RigidBodyHandle, Vec3>,
    /// Initial body rotations, stored alongside translations.
    pub initial_body_rotations: HashMap<RigidBodyHandle, Quat>,

    // -- Floating origin for f32 precision --
    /// Accumulated world origin offset in f64 for precision over long distances.
    ///
    /// All Rapier body positions are relative to this origin. To get the true
    /// world position of a body: `world_pos = rapier_pos + world_origin`.
    /// Periodically call [`rebase_origin`] to shift bodies back near zero.
    pub world_origin: [f64; 3],
}

impl RapierContext {
    /// Create a new context with given gravity, timestep, and substep count.
    pub fn new(gravity: Vec3, dt: f32, substeps: usize) -> Self {
        let mut integration_parameters = IntegrationParameters::default();
        integration_parameters.dt = dt;

        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            ccd_solver: CCDSolver::new(),
            integration_parameters,
            gravity,
            substeps,
            joint_handles: HashMap::new(),
            joint_info: HashMap::new(),
            body_handles: HashMap::new(),
            initial_body_positions: HashMap::new(),
            initial_body_rotations: HashMap::new(),
            world_origin: [0.0; 3],
        }
    }

    /// Store current body positions and rotations as the initial state for reset.
    pub fn snapshot_initial_state(&mut self) {
        self.initial_body_positions.clear();
        self.initial_body_rotations.clear();
        for (name, &handle) in &self.body_handles {
            if let Some(body) = self.rigid_body_set.get(handle) {
                let t = body.translation();
                self.initial_body_positions
                    .insert(handle, Vec3::new(t.x, t.y, t.z));
                self.initial_body_rotations
                    .insert(handle, *body.rotation());
            }
            let _ = name; // suppress unused warning
        }
    }

    /// Reset all rigid bodies to their initial poses with zero velocity.
    pub fn reset_to_initial(&mut self) {
        for (&handle, &initial_pos) in &self.initial_body_positions {
            if let Some(body) = self.rigid_body_set.get_mut(handle) {
                body.set_translation(initial_pos, true);
                let rotation = self
                    .initial_body_rotations
                    .get(&handle)
                    .copied()
                    .unwrap_or(Quat::IDENTITY);
                body.set_rotation(rotation, true);
                body.set_linvel(Vec3::ZERO, true);
                body.set_angvel(Vec3::ZERO, true);
                body.wake_up(true);
            }
        }
    }

    /// Shift all rigid bodies so the reference body is near the local origin.
    ///
    /// Call this periodically (e.g., every frame) to keep f32 coordinates
    /// near zero, preserving precision for long-distance traversal.
    /// Only rebases when the reference body exceeds `threshold` meters from
    /// the local origin in XY.
    ///
    /// Returns `true` if a rebase occurred.
    pub fn rebase_origin(&mut self, reference_body: &str, threshold: f32) -> bool {
        let Some(&ref_handle) = self.body_handles.get(reference_body) else {
            return false;
        };
        let Some(ref_body) = self.rigid_body_set.get(ref_handle) else {
            return false;
        };

        let t = ref_body.translation();
        let dist_xy = (t.x * t.x + t.y * t.y).sqrt();
        if dist_xy < threshold {
            return false;
        }

        // Shift amount: current reference body XY position (keep Z unchanged)
        let shift = Vec3::new(t.x, t.y, 0.0);

        // Accumulate in f64
        self.world_origin[0] += f64::from(shift.x);
        self.world_origin[1] += f64::from(shift.y);

        // Shift all rigid bodies
        for (_, handle) in &self.body_handles {
            if let Some(body) = self.rigid_body_set.get_mut(*handle) {
                let pos = body.translation();
                body.set_translation(pos - shift, true);
            }
        }

        true
    }

    /// Get the true world position of a body as f64 (local position + origin offset).
    pub fn world_position_f64(&self, handle: RigidBodyHandle) -> Option<[f64; 3]> {
        let body = self.rigid_body_set.get(handle)?;
        let t = body.translation();
        Some([
            f64::from(t.x) + self.world_origin[0],
            f64::from(t.y) + self.world_origin[1],
            f64::from(t.z) + self.world_origin[2],
        ])
    }

    /// Run one physics substep.
    pub fn step(&mut self) {
        self.physics_pipeline.step(
            self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            &(),
            &(),
        );
    }
}
