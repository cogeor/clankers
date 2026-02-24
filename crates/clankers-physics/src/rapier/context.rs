//! Bevy resource wrapping all Rapier3D physics pipeline state.

use std::collections::HashMap;

use bevy::prelude::{Entity, Resource, Vec3};
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
        }
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
