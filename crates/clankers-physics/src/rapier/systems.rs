//! Rapier physics step system.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use rapier3d::prelude::JointAxis;

use clankers_actuator::components::{JointState, JointTorque};
use clankers_core::layout::JointLayout;
use clankers_core::physics::ContactData;
use clankers_core::types::{MissingJoints, RobotGroup};

#[cfg(feature = "dense-runtime")]
use super::context::JointInfo;
use super::context::RapierContext;
#[cfg(feature = "dense-runtime")]
use super::runtime::JointRuntimes;

/// Position motor parameters for a single joint.
///
/// When a joint entity has an entry in [`MotorOverrides`], the physics step
/// system uses these PD motor parameters (evaluated at physics rate by Rapier)
/// instead of the torque motor trick (constant torque ZOH).
#[derive(Clone, Debug)]
pub struct MotorOverrideParams {
    /// Target joint position (radians for revolute, meters for prismatic).
    pub target_pos: f32,
    /// Target joint velocity (also encodes feedforward: `target_vel = ff_torque / damping`).
    pub target_vel: f32,
    /// Position gain (spring stiffness).
    pub stiffness: f32,
    /// Velocity gain (damping).
    pub damping: f32,
    /// Maximum motor force/torque.
    pub max_force: f32,
}

/// Per-joint position motor overrides.
///
/// Joints listed here bypass the torque motor trick in [`rapier_step_system`]
/// and instead use Rapier's built-in PD motor evaluated at the physics rate.
/// This is essential for stiff PD gains on light links where ZOH torque
/// control at the frame rate would cause oscillation.
///
/// # Important: ALL joints must be overridden
///
/// When using this resource, **every joint** (arm AND gripper) must have an
/// entry.  Joints without an override fall through to the actuator PID →
/// torque motor trick path, which is ZOH at the frame rate and will
/// oscillate on light links.  This is the most common cause of "robot
/// flailing wildly" in arm visualization binaries.
///
/// Use [`validate_motor_coverage`] at scene-build time to promote this
/// rule from a prose comment to a setup-time invariant.
///
/// See `arm_ik_viz.rs` and `arm_pick_gym.rs` for the canonical pattern:
/// arm joints use stiffness=100/damping=10, gripper fingers use softer
/// stiffness=50/damping=5.
///
/// # Storage
///
/// - `joints` — per-entity override map keyed by Bevy [`Entity`].
///   Consumed by [`rapier_step_system`] which looks each entity up as
///   it iterates `JointTorque` components.
/// - `layout` — the [`JointLayout`] the override set was built against.
///   Populated by callers that intend to pair the resource with
///   [`validate_motor_coverage`] at scene-build time.
///
/// # PR2 deviation note
///
/// PR1's prose proposed a parallel `ordered: Vec<(ImpulseJointHandle, _)>`
/// storage that the step system would consume in place of the entity
/// map. PR2 dropped that field because every active call site
/// (`arm_setup`, `arm_pick_gym`, `quadruped_mpc_viz`, `mpc_walk`,
/// `arm_startup`, the `validate_motor_coverage` tests in
/// `clankers-physics` and `clankers-sim`, and the builder pipeline) is
/// built around per-entity insertion before the Rapier
/// `ImpulseJointHandle`s exist — keying by handle would have required
/// restructuring all six sites onto a deferred-registration system,
/// which is out of scope for WS2 PR2. The entity map remains the
/// single source of truth; the dead `ordered` / `legacy_map` /
/// `from_legacy_map` / `From<HashMap>` surfaces from PR1 are deleted.
#[derive(Resource, Default)]
pub struct MotorOverrides {
    /// Per-entity override map consumed by [`rapier_step_system`].
    pub joints: HashMap<Entity, MotorOverrideParams>,
    /// The shared layout the overrides were built against. `None` when
    /// the override set is built ad-hoc without a layout (callers that
    /// want [`validate_motor_coverage`] coverage should populate this).
    pub layout: Option<Arc<JointLayout>>,
}

impl MotorOverrides {
    /// Build a [`MotorOverrides`] from a per-entity override map and the
    /// [`JointLayout`] it was built against. Convenience for the four
    /// example sites that construct the map ad-hoc before scene
    /// insertion.
    #[must_use]
    pub const fn from_map_and_layout(
        joints: HashMap<Entity, MotorOverrideParams>,
        layout: Arc<JointLayout>,
    ) -> Self {
        Self {
            joints,
            layout: Some(layout),
        }
    }
}

// ---------------------------------------------------------------------------
// validate_motor_coverage
// ---------------------------------------------------------------------------

/// Validate that `overrides` covers every joint declared by `layout`.
///
/// # Why a free function
///
/// `clankers-core` (where [`RobotGroup`] lives) cannot depend on
/// `clankers-physics::MotorOverrides` without creating a reverse
/// dependency cycle, and `clankers-sim` (which depends on both) cannot
/// be a dev-dependency of `clankers-physics` (cargo would reject the
/// implicit cycle through the regular dep). So the validator lives in
/// `clankers-physics` next to [`MotorOverrides`]; `clankers-sim`'s
/// [`SceneBuilder::try_build`](../../../../clankers_sim/builder/struct.SceneBuilder.html#method.try_build)
/// re-exports the same surface.
///
/// The `group` argument is accepted for future extensibility (e.g.
/// per-robot validation in multi-robot scenes). The current
/// implementation only consults `layout` and `overrides`.
///
/// # Errors
///
/// Returns [`MissingJoints`] listing every layout joint whose entity is
/// absent from `overrides.joints`. Layout slots that have not been
/// bound to an entity yet (where `JointSpec.entity` is `None`) are
/// silently skipped — the layout owner must call
/// [`JointLayout::bind_entities`](clankers_core::layout::JointLayout::bind_entities)
/// before validation.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
///
/// use bevy::ecs::entity::Entity;
/// use clankers_core::layout::{
///     JointKind, JointLayoutBuilder, JointSpec, JointSpecLimits,
/// };
/// use clankers_core::types::RobotGroup;
/// use clankers_physics::rapier::systems::{
///     MotorOverrideParams, MotorOverrides, validate_motor_coverage,
/// };
///
/// let mut layout = JointLayoutBuilder::default()
///     .push(JointSpec {
///         name: "joint_a".into(),
///         entity: None,
///         joint_type: JointKind::Revolute,
///         limits: JointSpecLimits::default(),
///         axis: [0.0, 0.0, 1.0],
///     })
///     .build();
/// let entity = Entity::from_bits(1);
/// layout.bind_entities(&[entity]);
///
/// let mut overrides = MotorOverrides::default();
/// overrides.joints.insert(entity, MotorOverrideParams {
///     target_pos: 0.0, target_vel: 0.0,
///     stiffness: 100.0, damping: 10.0, max_force: 50.0,
/// });
///
/// assert!(validate_motor_coverage(
///     &RobotGroup::default(), &layout, &overrides,
/// ).is_ok());
/// ```
pub fn validate_motor_coverage(
    _group: &RobotGroup,
    layout: &JointLayout,
    overrides: &MotorOverrides,
) -> Result<(), MissingJoints> {
    let missing: Vec<String> = layout
        .joints()
        .iter()
        .filter_map(|spec| {
            spec.entity.and_then(|entity| {
                if overrides.joints.contains_key(&entity) {
                    None
                } else {
                    Some(spec.name.clone())
                }
            })
        })
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(MissingJoints {
            layout_joint_names: missing,
            override_joint_count: overrides.joints.len(),
        })
    }
}

/// Per-joint motor command rate limiting.
///
/// Clamps the change in motor target position between consecutive control
/// steps: `target = clamp(target, prev - delta_max, prev + delta_max)`.
/// Applied at the actuator output, NOT inside the MPC QP.
///
/// # Storage
///
/// - `prev_targets`: `HashMap<Entity, f32>` — used by the legacy
///   `rapier_step_system_hashmap` path (no `JointRuntimes` resource).
/// - `slot_prev`: layout-slot-indexed `Vec<f32>` — used by the dense
///   `rapier_step_system_dense` hot path. Sized lazily on first use to
///   match `runtimes.joints.len()`. **P2.4** / `CODE_QUALITY_REVIEW`
///   § Phase 2.4: replaces the per-entity `HashMap` probe inside the
///   per-joint loop with an O(1) array read.
#[derive(Resource)]
pub struct MotorRateLimits {
    /// Maximum position change per control step (radians for revolute).
    pub delta_max: f32,
    /// Previous target positions, keyed by entity. Used by the
    /// HashMap-based fallback path only.
    pub prev_targets: HashMap<Entity, f32>,
    /// Previous target positions, indexed by layout slot. Used by the
    /// dense (`JointRuntimes`-driven) hot path. Sized lazily.
    pub slot_prev: Vec<f32>,
    /// Per-slot occupancy flag tracking whether `slot_prev[slot]` has
    /// been seeded yet. Avoids treating a freshly-zero entry as a real
    /// prior target (which would clamp the first step to 0).
    pub slot_seen: Vec<bool>,
}

impl MotorRateLimits {
    /// Create rate limits with the given maximum position delta per step.
    pub fn new(delta_max: f32) -> Self {
        Self {
            delta_max,
            prev_targets: HashMap::new(),
            slot_prev: Vec::new(),
            slot_seen: Vec::new(),
        }
    }

    /// Ensure the slot-indexed previous-target buffer matches `n` joints.
    /// Newly added slots are flagged as not-yet-seen so the next step
    /// uses the live target as the prior.
    pub fn ensure_slot_capacity(&mut self, n: usize) {
        if self.slot_prev.len() != n {
            self.slot_prev.resize(n, 0.0);
            self.slot_seen.resize(n, false);
        }
    }
}

/// High-frequency inner PD interpolation state.
///
/// When this resource is present, motor target positions are linearly
/// interpolated across physics substeps instead of being set once (ZOH).
/// This provides effective 1000Hz PD control while the MPC runs at 50Hz.
///
/// # Storage
///
/// As with [`MotorRateLimits`], the dense path reads `slot_prev` /
/// `slot_seen` directly; the `HashMap` remains for the fallback.
#[derive(Resource, Default)]
pub struct InnerPdState {
    /// Previous control step's target positions per entity (`HashMap` fallback).
    prev_targets: HashMap<Entity, f32>,
    /// Layout-slot-indexed previous-target cache for the dense path.
    pub slot_prev: Vec<f32>,
    /// Per-slot occupancy flag (see [`MotorRateLimits::slot_seen`]).
    pub slot_seen: Vec<bool>,
}

impl InnerPdState {
    /// Ensure the slot-indexed previous-target buffer matches `n` joints.
    pub fn ensure_slot_capacity(&mut self, n: usize) {
        if self.slot_prev.len() != n {
            self.slot_prev.resize(n, 0.0);
            self.slot_seen.resize(n, false);
        }
    }
}

/// Apply joint torques, step physics, read back joint state.
///
/// When [`InnerPdState`] is present, motor target positions are linearly
/// interpolated across substeps for effective 1000Hz PD control.
///
/// # W7 PR3 dispatch
///
/// When the `dense-runtime` cargo feature is enabled (the default) and
/// a `JointRuntimes` (in `super::runtime`) resource has been populated
/// by `clankers_sim::builder::compile_runtime`, the hot path drives the
/// per-frame motor / readback loops from the dense `Vec<JointRuntime>`
/// instead of probing `RapierContext.joint_handles` and
/// `RapierContext.joint_info` per entity. The `HashMap` fallback runs
/// whenever the feature is disabled OR the resource is absent
/// (`SceneBuilder::build` does not populate it, only `try_build` does).
#[allow(clippy::needless_pass_by_value)]
pub fn rapier_step_system(
    context: ResMut<RapierContext>,
    joints: Query<(Entity, &JointTorque, &mut JointState)>,
    motor_overrides: Option<Res<MotorOverrides>>,
    #[cfg(feature = "dense-runtime")] runtimes: Option<Res<JointRuntimes>>,
    rate_limits: Option<ResMut<MotorRateLimits>>,
    inner_pd: Option<ResMut<InnerPdState>>,
) {
    #[cfg(feature = "dense-runtime")]
    {
        if let Some(rt) = runtimes.as_deref()
            && !rt.is_empty()
        {
            rapier_step_system_dense(
                context.into_inner(),
                joints,
                motor_overrides.as_deref(),
                rt,
                rate_limits,
                inner_pd,
            );
            return;
        }
    }
    rapier_step_system_hashmap(
        context.into_inner(),
        joints,
        motor_overrides.as_deref(),
        rate_limits,
        inner_pd,
    );
}

/// W7 PR3 dense-vec hot path. Iterates `runtimes.joints` in layout-
/// slot order, reading per-joint handle / info / motor params from the
/// pre-compiled `JointRuntime` instead of `HashMap::get(&entity)` per
/// joint per frame.
///
/// Behaviour invariants:
/// - Per-step output (rigid body poses) is byte-equal to the `HashMap`
///   fallback ([`rapier_step_system_hashmap`]) — see
///   `crates/clankers-physics/tests/dense_runtime.rs::dense_runtime_matches_hashmap_lookup`.
/// - Iteration order is layout slot order (deterministic) rather than
///   Bevy archetype query order — a strengthening, not a weakening.
/// - Rate-limit / inner-PD `HashMaps` stay as warm caches keyed by
///   `Entity`; this loop does NOT migrate those caches.
#[cfg(feature = "dense-runtime")]
#[allow(
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::branches_sharing_code
)]
fn rapier_step_system_dense(
    context: &mut RapierContext,
    mut joints: Query<(Entity, &JointTorque, &mut JointState)>,
    motor_overrides: Option<&MotorOverrides>,
    runtimes: &JointRuntimes,
    mut rate_limits: Option<ResMut<MotorRateLimits>>,
    mut inner_pd: Option<ResMut<InnerPdState>>,
) {
    let substeps = context.substeps;
    let use_inner_pd = inner_pd.is_some() && motor_overrides.is_some();

    // Collect override data for interpolation (needed if inner PD is active).
    struct OverrideEntry {
        joint_handle: rapier3d::dynamics::ImpulseJointHandle,
        axis: JointAxis,
        prev_pos: f32,
        target_pos: f32,
        target_vel: f32,
        stiffness: f32,
        damping: f32,
        max_force: f32,
    }
    let mut override_entries: Vec<OverrideEntry> = Vec::new();

    // P2.4: lazily size slot caches to match the runtime's joint count
    // so the per-joint loop reads `slot_prev[slot]` instead of probing
    // `prev_targets.get(&entity)`.
    let n_joints = runtimes.joints.len();
    if let Some(ref mut limits) = rate_limits {
        limits.ensure_slot_capacity(n_joints);
    }
    if let Some(ref mut pd) = inner_pd {
        pd.ensure_slot_capacity(n_joints);
    }

    // 1. Apply torques via slot-indexed JointRuntime traversal.
    for (slot, jr) in runtimes.joints.iter().enumerate() {
        // Per-joint Bevy-side data: torque only (JointState is read in
        // the readback pass below). Skip silently if the query slot
        // is absent — this can happen during teardown.
        let Ok((_e, torque, _)) = joints.get(jr.entity) else {
            continue;
        };

        let axis = if jr.info.is_prismatic {
            JointAxis::LinX
        } else {
            JointAxis::AngX
        };

        let Some(joint) = context.impulse_joint_set.get_mut(jr.handle, true) else {
            continue;
        };

        // Resolve the motor override: prefer the live HashMap entry
        // (fresh after any mid-frame mutation), fall back to the
        // compile-time snapshot in the runtime entry. Either way, no
        // `joint_handles` or `joint_info` probe — that is the W7 PR3
        // win.
        let motor_snapshot = jr.motor.as_ref();
        let motor_live = motor_overrides.and_then(|o| o.joints.get(&jr.entity));
        let motor = motor_live.or(motor_snapshot);

        if let Some(mo) = motor {
            // Apply rate limiting if configured (P2.4: slot-indexed Vec read).
            let target_pos = if let Some(ref mut limits) = rate_limits {
                let prev = if limits.slot_seen[slot] {
                    limits.slot_prev[slot]
                } else {
                    mo.target_pos
                };
                let clamped = mo
                    .target_pos
                    .clamp(prev - limits.delta_max, prev + limits.delta_max);
                limits.slot_prev[slot] = clamped;
                limits.slot_seen[slot] = true;
                clamped
            } else {
                mo.target_pos
            };

            if use_inner_pd {
                let pd = inner_pd.as_mut().unwrap();
                let prev_pos = if pd.slot_seen[slot] {
                    pd.slot_prev[slot]
                } else {
                    target_pos
                };
                pd.slot_prev[slot] = target_pos;
                pd.slot_seen[slot] = true;
                override_entries.push(OverrideEntry {
                    joint_handle: jr.handle,
                    axis,
                    prev_pos,
                    target_pos,
                    target_vel: mo.target_vel,
                    stiffness: mo.stiffness,
                    damping: mo.damping,
                    max_force: mo.max_force,
                });
                let alpha = 1.0 / substeps as f32;
                let interp = (target_pos - prev_pos).mul_add(alpha, prev_pos);
                joint
                    .data
                    .set_motor(axis, interp, mo.target_vel, mo.stiffness, mo.damping);
                joint.data.set_motor_max_force(axis, mo.max_force);
            } else {
                joint
                    .data
                    .set_motor(axis, target_pos, mo.target_vel, mo.stiffness, mo.damping);
                joint.data.set_motor_max_force(axis, mo.max_force);
            }
        } else {
            // Motor trick fallback: ForceBased motor with huge target
            // velocity, clamped to desired torque magnitude.
            let t = torque.value;
            if t.abs() > 1e-10 {
                let target_vel = t.signum() * 1e10;
                joint.data.set_motor(axis, 0.0, target_vel, 0.0, 1.0);
                joint.data.set_motor_max_force(axis, t.abs());
            } else {
                joint.data.set_motor(axis, 0.0, 0.0, 0.0, 0.0);
                joint.data.set_motor_max_force(axis, 0.0);
            }
        }
    }

    // 2. Step physics with inner PD interpolation.
    if use_inner_pd && !override_entries.is_empty() {
        context.step();
        for sub in 1..substeps {
            let alpha = (sub + 1) as f32 / substeps as f32;
            for entry in &override_entries {
                if let Some(joint) = context.impulse_joint_set.get_mut(entry.joint_handle, true) {
                    let interp = (entry.target_pos - entry.prev_pos).mul_add(alpha, entry.prev_pos);
                    joint.data.set_motor(
                        entry.axis,
                        interp,
                        entry.target_vel,
                        entry.stiffness,
                        entry.damping,
                    );
                    joint.data.set_motor_max_force(entry.axis, entry.max_force);
                }
            }
            context.step();
        }
    } else {
        for _ in 0..substeps {
            context.step();
        }
    }

    // 3. Read back joint state from rigid body transforms.
    for jr in &runtimes.joints {
        let Ok((_e, _t, mut state)) = joints.get_mut(jr.entity) else {
            continue;
        };
        let info: &JointInfo = &jr.info;

        let Some(parent_body) = context.rigid_body_set.get(info.parent_body) else {
            continue;
        };
        let Some(child_body) = context.rigid_body_set.get(info.child_body) else {
            continue;
        };

        if info.is_prismatic {
            let parent_pos = parent_body.position().translation;
            let child_pos = child_body.position().translation;
            let relative_pos = child_pos - parent_pos;
            state.position = relative_pos.dot(info.axis);
            let relative_vel = child_body.linvel() - parent_body.linvel();
            state.velocity = relative_vel.dot(info.axis);
        } else {
            let parent_rot = parent_body.position().rotation;
            let child_rot = child_body.position().rotation;
            let relative_rotation = parent_rot.inverse() * child_rot;
            let sin_half = Vec3::new(
                relative_rotation.x,
                relative_rotation.y,
                relative_rotation.z,
            );
            let cos_half = relative_rotation.w;
            let sin_half_proj = sin_half.dot(info.axis);
            state.position = 2.0 * f32::atan2(sin_half_proj, cos_half);
            let relative_angvel = child_body.angvel() - parent_body.angvel();
            state.velocity = relative_angvel.dot(info.axis);
        }
    }
}

/// Legacy HashMap-driven hot path. Used as a fallback when the
/// `dense-runtime` feature is disabled, or when the `JointRuntimes`
/// resource has not been populated (e.g. `SceneBuilder::build`,
/// the non-`try_build` variant). The body is the pre-W7-PR3
/// implementation, unchanged in behaviour.
#[allow(
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::branches_sharing_code
)]
fn rapier_step_system_hashmap(
    context: &mut RapierContext,
    mut joints: Query<(Entity, &JointTorque, &mut JointState)>,
    motor_overrides: Option<&MotorOverrides>,
    mut rate_limits: Option<ResMut<MotorRateLimits>>,
    mut inner_pd: Option<ResMut<InnerPdState>>,
) {
    let substeps = context.substeps;
    let use_inner_pd = inner_pd.is_some() && motor_overrides.is_some();

    // Collect override data for interpolation (needed if inner PD is active)
    struct OverrideEntry {
        joint_handle: rapier3d::dynamics::ImpulseJointHandle,
        axis: JointAxis,
        prev_pos: f32,
        target_pos: f32,
        target_vel: f32,
        stiffness: f32,
        damping: f32,
        max_force: f32,
    }

    let mut override_entries: Vec<OverrideEntry> = Vec::new();

    // 1. Apply torques to rapier joints via motor trick (or position motor override)
    for (entity, torque, _) in &joints {
        let Some(&joint_handle) = context.joint_handles.get(&entity) else {
            continue;
        };
        let Some(info) = context.joint_info.get(&entity) else {
            continue;
        };

        let axis = if info.is_prismatic {
            JointAxis::LinX
        } else {
            JointAxis::AngX
        };

        if let Some(joint) = context.impulse_joint_set.get_mut(joint_handle, true) {
            // Check for position motor override first
            if let Some(overrides) = motor_overrides
                && let Some(mo) = overrides.joints.get(&entity)
            {
                // Apply rate limiting if configured
                let target_pos = if let Some(ref mut limits) = rate_limits {
                    let prev = limits
                        .prev_targets
                        .get(&entity)
                        .copied()
                        .unwrap_or(mo.target_pos);
                    let clamped = mo
                        .target_pos
                        .clamp(prev - limits.delta_max, prev + limits.delta_max);
                    limits.prev_targets.insert(entity, clamped);
                    clamped
                } else {
                    mo.target_pos
                };

                if use_inner_pd {
                    // Store for substep interpolation
                    let pd = inner_pd.as_mut().unwrap();
                    let prev_pos = pd.prev_targets.get(&entity).copied().unwrap_or(target_pos);
                    pd.prev_targets.insert(entity, target_pos);
                    override_entries.push(OverrideEntry {
                        joint_handle,
                        axis,
                        prev_pos,
                        target_pos,
                        target_vel: mo.target_vel,
                        stiffness: mo.stiffness,
                        damping: mo.damping,
                        max_force: mo.max_force,
                    });
                    // Set initial interpolated target (substep 0)
                    let alpha = 1.0 / substeps as f32;
                    let interp = (target_pos - prev_pos).mul_add(alpha, prev_pos);
                    joint
                        .data
                        .set_motor(axis, interp, mo.target_vel, mo.stiffness, mo.damping);
                    joint.data.set_motor_max_force(axis, mo.max_force);
                } else {
                    joint
                        .data
                        .set_motor(axis, target_pos, mo.target_vel, mo.stiffness, mo.damping);
                    joint.data.set_motor_max_force(axis, mo.max_force);
                }
            } else {
                // Motor trick: ForceBased motor with huge target velocity,
                // clamped to desired torque magnitude.
                let t = torque.value;
                if t.abs() > 1e-10 {
                    let target_vel = t.signum() * 1e10;
                    joint.data.set_motor(axis, 0.0, target_vel, 0.0, 1.0);
                    joint.data.set_motor_max_force(axis, t.abs());
                } else {
                    joint.data.set_motor(axis, 0.0, 0.0, 0.0, 0.0);
                    joint.data.set_motor_max_force(axis, 0.0);
                }
            }
        }
    }

    // 2. Step physics with inner PD interpolation
    if use_inner_pd && !override_entries.is_empty() {
        // First substep already has interpolated target set above
        context.step();

        // Remaining substeps: update interpolated targets
        for sub in 1..substeps {
            let alpha = (sub + 1) as f32 / substeps as f32;
            for entry in &override_entries {
                if let Some(joint) = context.impulse_joint_set.get_mut(entry.joint_handle, true) {
                    let interp = (entry.target_pos - entry.prev_pos).mul_add(alpha, entry.prev_pos);
                    joint.data.set_motor(
                        entry.axis,
                        interp,
                        entry.target_vel,
                        entry.stiffness,
                        entry.damping,
                    );
                    joint.data.set_motor_max_force(entry.axis, entry.max_force);
                }
            }
            context.step();
        }
    } else {
        for _ in 0..substeps {
            context.step();
        }
    }

    // 3. Read back joint state from rigid body transforms
    for (entity, _, mut state) in &mut joints {
        let Some(info) = context.joint_info.get(&entity) else {
            continue;
        };

        let Some(parent_body) = context.rigid_body_set.get(info.parent_body) else {
            continue;
        };
        let Some(child_body) = context.rigid_body_set.get(info.child_body) else {
            continue;
        };

        if info.is_prismatic {
            // Prismatic: displacement along joint axis
            let parent_pos = parent_body.position().translation;
            let child_pos = child_body.position().translation;
            let relative_pos = child_pos - parent_pos;
            state.position = relative_pos.dot(info.axis);

            // Velocity along axis
            let relative_vel = child_body.linvel() - parent_body.linvel();
            state.velocity = relative_vel.dot(info.axis);
        } else {
            // Revolute: rotation around joint axis
            let parent_rot = parent_body.position().rotation;
            let child_rot = child_body.position().rotation;
            let relative_rotation = parent_rot.inverse() * child_rot;

            // Extract angle around joint axis from quaternion
            let sin_half = Vec3::new(
                relative_rotation.x,
                relative_rotation.y,
                relative_rotation.z,
            );
            let cos_half = relative_rotation.w;
            let sin_half_proj = sin_half.dot(info.axis);
            state.position = 2.0 * f32::atan2(sin_half_proj, cos_half);

            // Angular velocity around axis
            let relative_angvel = child_body.angvel() - parent_body.angvel();
            state.velocity = relative_angvel.dot(info.axis);
        }
    }
}

/// Read contact forces from Rapier's narrow phase and populate [`ContactData`] components.
///
/// For each active contact pair, computes the total impulse and converts to force (N).
/// Entities not involved in any contact are reset to zero.
#[allow(clippy::needless_pass_by_value)]
pub fn contact_update_system(context: Res<RapierContext>, mut contacts: Query<&mut ContactData>) {
    // Reset all ContactData to zero
    for mut cd in &mut contacts {
        cd.normal_force = Vec3::ZERO;
    }

    let dt = context.integration_parameters.dt;
    if dt <= 0.0 {
        return;
    }

    // Iterate all contact pairs from the narrow phase
    for contact_pair in context.narrow_phase.contact_pairs() {
        if !contact_pair.has_any_active_contact() {
            continue;
        }

        // total_impulse() sums per-manifold impulses scaled by their normals
        let impulse = contact_pair.total_impulse();
        let force = impulse / dt;

        // Find which rigid bodies own these colliders
        let body1 = context
            .collider_set
            .get(contact_pair.collider1)
            .and_then(rapier3d::geometry::Collider::parent);
        let body2 = context
            .collider_set
            .get(contact_pair.collider2)
            .and_then(rapier3d::geometry::Collider::parent);

        // Accumulate force on the entity owning body1
        if let Some(body_handle) = body1
            && let Some(&entity) = context.body_to_entity.get(&body_handle)
            && let Ok(mut cd) = contacts.get_mut(entity)
        {
            cd.normal_force += force;
        }

        // Accumulate negative force on body2 (Newton's 3rd law)
        if let Some(body_handle) = body2
            && let Some(&entity) = context.body_to_entity.get(&body_handle)
            && let Ok(mut cd) = contacts.get_mut(entity)
        {
            cd.normal_force -= force;
        }
    }
}
