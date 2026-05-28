//! Structure-of-Arrays simulation buffers (P3.1).
//!
//! `CODE_QUALITY_REVIEW` § "`SoA` Simulation Buffers" / P3.1. The hot
//! simulation loop currently reads commands and writes state through
//! Bevy ECS queries. That's ergonomic for systems, but not ideal for
//! device execution and not cache-friendly even on CPU: per-joint
//! data is interleaved with unrelated ECS components, and the
//! per-step iteration touches one cache line per joint.
//!
//! These types declare the canonical hot-loop data layout Phase 3
//! will converge on:
//!
//! - [`JointCommandBuffer`] — what the controller wants the joint to
//!   do this step. Filled by the action applicator before the
//!   physics step.
//! - [`JointStateBuffer`] — what the simulator observed at the joint
//!   this step. Filled by the physics step.
//! - [`BodyStateBuffer`] — per-body pose / velocity readback.
//!
//! Every field is a `Vec<f32>` (or `Vec<[f32; N]>`) indexed by the
//! slot id allocated at scene-build time. Slot ids are stable for
//! the lifetime of the scene; the buffers grow / shrink only on
//! scene rebuild.
//!
//! # What's NOT here yet
//!
//! - The backend `step(commands, &mut outputs)` signature that
//!   consumes / produces these buffers (P3.2).
//! - ECS mirroring (P3.3).
//! - Sensor / protocol / recorder views into these buffers (P3.4).
//!
//! This is a foundation commit: defines the data shapes and the
//! grow / clear / resize lifecycle so subsequent work can layer on
//! without re-arguing the layout.

// ---------------------------------------------------------------------------
// JointCommandBuffer
// ---------------------------------------------------------------------------

/// Per-slot joint command targets read by the physics backend before
/// each step.
///
/// All three vectors are the same length (= number of joint slots
/// in the active layout). Index by slot id, NOT by Bevy `Entity`.
///
/// # Default semantics per slot
///
/// - `target_pos`: NaN means "no position target this step".
/// - `target_vel`: NaN means "no velocity target this step".
/// - `max_force`: `f32::INFINITY` means "no force ceiling".
///
/// The backend ignores NaN slots; this lets sparse commands flow
/// through the same buffer as dense ones without an extra "is this
/// slot set?" mask.
#[derive(Debug, Clone, Default)]
pub struct JointCommandBuffer {
    /// Desired joint position per slot (rad or m). NaN = unset.
    pub target_pos: Vec<f32>,
    /// Desired joint velocity per slot (rad/s or m/s). NaN = unset.
    pub target_vel: Vec<f32>,
    /// Maximum applied force per slot (Nm or N). `f32::INFINITY` =
    /// unbounded.
    pub max_force: Vec<f32>,
}

impl JointCommandBuffer {
    /// Allocate a buffer for `n` joint slots, initialised to "no
    /// target" semantics: NaN positions / velocities, unbounded
    /// force.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            target_pos: vec![f32::NAN; n],
            target_vel: vec![f32::NAN; n],
            max_force: vec![f32::INFINITY; n],
        }
    }

    /// Number of joint slots.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.target_pos.len()
    }

    /// Whether the buffer has zero slots.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.target_pos.is_empty()
    }

    /// Reset every slot to "no target this step". Called by the
    /// action applicator at the top of each step before writing the
    /// fresh batch of commands.
    pub fn clear_targets(&mut self) {
        for v in &mut self.target_pos {
            *v = f32::NAN;
        }
        for v in &mut self.target_vel {
            *v = f32::NAN;
        }
        for v in &mut self.max_force {
            *v = f32::INFINITY;
        }
    }

    /// Resize the buffer to `n` slots, preserving existing values
    /// where possible. New slots use the "unset" defaults.
    pub fn resize(&mut self, n: usize) {
        self.target_pos.resize(n, f32::NAN);
        self.target_vel.resize(n, f32::NAN);
        self.max_force.resize(n, f32::INFINITY);
    }
}

// ---------------------------------------------------------------------------
// JointStateBuffer
// ---------------------------------------------------------------------------

/// Per-slot joint state read out from the physics backend after each
/// step.
///
/// Same indexing convention as [`JointCommandBuffer`]: slot id, not
/// Bevy `Entity`.
#[derive(Debug, Clone, Default)]
pub struct JointStateBuffer {
    /// Joint position per slot (rad or m).
    pub position: Vec<f32>,
    /// Joint velocity per slot (rad/s or m/s).
    pub velocity: Vec<f32>,
    /// Applied torque / force per slot (Nm or N).
    pub torque: Vec<f32>,
}

impl JointStateBuffer {
    /// Allocate a zeroed state buffer for `n` slots.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            position: vec![0.0; n],
            velocity: vec![0.0; n],
            torque: vec![0.0; n],
        }
    }

    /// Number of joint slots.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.position.len()
    }

    /// Whether the buffer has zero slots.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.position.is_empty()
    }

    /// Resize the buffer to `n` slots, preserving existing values
    /// where possible. New slots are zeroed.
    pub fn resize(&mut self, n: usize) {
        self.position.resize(n, 0.0);
        self.velocity.resize(n, 0.0);
        self.torque.resize(n, 0.0);
    }
}

// ---------------------------------------------------------------------------
// BodyStateBuffer
// ---------------------------------------------------------------------------

/// Per-slot body state for the post-step readback path.
///
/// Position / orientation use the same convention as
/// [`crate::neutral::BodyPose`]: xyz position, xyzw quaternion. The
/// linear and angular velocity vectors are world-frame.
#[derive(Debug, Clone, Default)]
pub struct BodyStateBuffer {
    /// World-frame position per body slot (m).
    pub position_xyz: Vec<[f32; 3]>,
    /// World-frame orientation per body slot, quaternion [x, y, z, w].
    pub orientation_xyzw: Vec<[f32; 4]>,
    /// Linear velocity per body slot (m/s).
    pub linvel_xyz: Vec<[f32; 3]>,
    /// Angular velocity per body slot (rad/s).
    pub angvel_xyz: Vec<[f32; 3]>,
}

impl BodyStateBuffer {
    /// Allocate an identity-pose buffer for `n` body slots.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            position_xyz: vec![[0.0; 3]; n],
            orientation_xyzw: vec![[0.0, 0.0, 0.0, 1.0]; n],
            linvel_xyz: vec![[0.0; 3]; n],
            angvel_xyz: vec![[0.0; 3]; n],
        }
    }

    /// Number of body slots.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.position_xyz.len()
    }

    /// Whether the buffer has zero slots.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.position_xyz.is_empty()
    }

    /// Resize the buffer to `n` slots, preserving existing values.
    /// New slots use the identity pose (origin, unit quaternion,
    /// zero velocity).
    pub fn resize(&mut self, n: usize) {
        self.position_xyz.resize(n, [0.0; 3]);
        self.orientation_xyzw.resize(n, [0.0, 0.0, 0.0, 1.0]);
        self.linvel_xyz.resize(n, [0.0; 3]);
        self.angvel_xyz.resize(n, [0.0; 3]);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_command_buffer_defaults_are_unset() {
        let b = JointCommandBuffer::with_capacity(4);
        assert_eq!(b.len(), 4);
        for v in &b.target_pos {
            assert!(v.is_nan(), "target_pos should default NaN");
        }
        for v in &b.target_vel {
            assert!(v.is_nan(), "target_vel should default NaN");
        }
        for v in &b.max_force {
            assert!(v.is_infinite(), "max_force should default +inf");
        }
    }

    #[test]
    fn joint_command_buffer_clear_resets_all_slots() {
        let mut b = JointCommandBuffer::with_capacity(2);
        b.target_pos[0] = 1.0;
        b.target_vel[1] = 2.0;
        b.max_force[0] = 50.0;
        b.clear_targets();
        assert!(b.target_pos[0].is_nan());
        assert!(b.target_vel[1].is_nan());
        assert!(b.max_force[0].is_infinite());
    }

    #[test]
    fn joint_command_buffer_resize_preserves_and_pads() {
        let mut b = JointCommandBuffer::with_capacity(2);
        b.target_pos[0] = 1.0;
        b.target_pos[1] = 2.0;
        b.resize(4);
        assert_eq!(b.len(), 4);
        assert!((b.target_pos[0] - 1.0).abs() < f32::EPSILON);
        assert!((b.target_pos[1] - 2.0).abs() < f32::EPSILON);
        assert!(b.target_pos[2].is_nan());
        assert!(b.target_pos[3].is_nan());
    }

    #[test]
    fn joint_state_buffer_defaults_are_zero() {
        let b = JointStateBuffer::with_capacity(3);
        assert_eq!(b.len(), 3);
        for &v in &b.position {
            assert!((v - 0.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn body_state_buffer_default_quat_is_identity() {
        let b = BodyStateBuffer::with_capacity(2);
        for q in &b.orientation_xyzw {
            assert_eq!(*q, [0.0, 0.0, 0.0, 1.0]);
        }
        for p in &b.position_xyz {
            assert_eq!(*p, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn body_state_buffer_resize_preserves() {
        let mut b = BodyStateBuffer::with_capacity(1);
        b.position_xyz[0] = [1.0, 2.0, 3.0];
        b.resize(3);
        assert_eq!(b.len(), 3);
        assert_eq!(b.position_xyz[0], [1.0, 2.0, 3.0]);
        // New slots default to origin + identity quat.
        assert_eq!(b.position_xyz[1], [0.0; 3]);
        assert_eq!(b.orientation_xyzw[1], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn all_buffers_empty_by_default() {
        assert!(JointCommandBuffer::default().is_empty());
        assert!(JointStateBuffer::default().is_empty());
        assert!(BodyStateBuffer::default().is_empty());
    }
}
