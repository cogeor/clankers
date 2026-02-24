//! Kinematic chain extracted from a URDF [`RobotModel`].
//!
//! A [`KinematicChain`] is an ordered list of joints from the base link to
//! the end-effector link. It stores the static transforms (origins) and
//! joint axes needed for forward kinematics and Jacobian computation.

use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion, UnitVector3, Vector3};

use clankers_urdf::{JointType, Origin, RobotModel};

/// A single joint in the kinematic chain.
#[derive(Debug, Clone)]
pub struct ChainJoint {
    /// Name of this joint (from URDF).
    pub name: String,
    /// Static transform from parent link frame to this joint frame.
    pub origin: Isometry3<f32>,
    /// Joint axis in the joint's local frame.
    pub axis: UnitVector3<f32>,
    /// Whether this is a prismatic joint (false = revolute).
    pub is_prismatic: bool,
    /// Lower position limit (rad or m).
    pub lower_limit: f32,
    /// Upper position limit (rad or m).
    pub upper_limit: f32,
}

/// An ordered kinematic chain from base to end-effector.
///
/// Built from a [`RobotModel`] by tracing the joint tree from root to a
/// specified end-effector link. Only actuated joints (revolute, continuous,
/// prismatic) are included; fixed joints have their transforms folded into
/// the next actuated joint's origin.
#[derive(Debug, Clone)]
pub struct KinematicChain {
    /// Ordered joints from base to end-effector.
    joints: Vec<ChainJoint>,
    /// Transform from the last joint's child link to the end-effector frame.
    /// Accounts for any trailing fixed joints.
    ee_offset: Isometry3<f32>,
}

impl KinematicChain {
    /// Build a kinematic chain from a [`RobotModel`].
    ///
    /// Traces from `root_link` to `ee_link`, collecting actuated joints.
    /// Fixed joints are folded into the accumulated transform.
    ///
    /// # Errors
    ///
    /// Returns `None` if `ee_link` is not reachable from `root_link`.
    pub fn from_model(model: &RobotModel, ee_link: &str) -> Option<Self> {
        // Build parent-link -> children-joints map
        let path = find_path_to_link(model, &model.root_link, ee_link)?;

        let mut joints = Vec::new();
        let mut accumulated_fixed = Isometry3::identity();

        for joint_name in &path {
            let joint = model.joint(joint_name).ok()?;
            let joint_origin = origin_to_isometry(&joint.origin);

            if joint.joint_type.is_actuated() {
                // Compose any accumulated fixed transforms with this joint's origin
                let combined_origin = accumulated_fixed * joint_origin;
                accumulated_fixed = Isometry3::identity();

                let axis = Vector3::new(joint.axis[0], joint.axis[1], joint.axis[2]);
                let axis = UnitVector3::new_normalize(axis);

                let (lower, upper) = match joint.joint_type {
                    JointType::Continuous => (-std::f32::consts::PI, std::f32::consts::PI),
                    _ => (
                        joint.limits.lower.unwrap_or(-std::f32::consts::PI),
                        joint.limits.upper.unwrap_or(std::f32::consts::PI),
                    ),
                };

                joints.push(ChainJoint {
                    name: joint.name.clone(),
                    origin: combined_origin,
                    axis,
                    is_prismatic: joint.joint_type == JointType::Prismatic,
                    lower_limit: lower,
                    upper_limit: upper,
                });
            } else {
                // Fixed joint: accumulate its transform
                accumulated_fixed *= joint_origin;
            }
        }

        Some(Self {
            joints,
            ee_offset: accumulated_fixed,
        })
    }

    /// Number of actuated degrees of freedom.
    pub fn dof(&self) -> usize {
        self.joints.len()
    }

    /// Joint names in chain order.
    pub fn joint_names(&self) -> Vec<&str> {
        self.joints.iter().map(|j| j.name.as_str()).collect()
    }

    /// Access the joint definitions.
    pub fn joints(&self) -> &[ChainJoint] {
        &self.joints
    }

    /// End-effector offset after the last joint.
    pub fn ee_offset(&self) -> &Isometry3<f32> {
        &self.ee_offset
    }

    /// Compute forward kinematics: joint positions -> end-effector pose.
    ///
    /// Returns the end-effector pose in the base frame.
    ///
    /// # Panics
    ///
    /// Panics if `q.len() != self.dof()`.
    pub fn forward_kinematics(&self, q: &[f32]) -> Isometry3<f32> {
        assert_eq!(q.len(), self.dof(), "q.len() must equal chain DOF");

        let mut transform = Isometry3::identity();
        for (joint, &angle) in self.joints.iter().zip(q.iter()) {
            // Apply static origin transform
            transform *= joint.origin;
            // Apply joint motion
            transform *= joint_transform(&joint.axis, joint.is_prismatic, angle);
        }
        // Apply trailing fixed-joint offset
        transform * self.ee_offset
    }

    /// Compute per-joint transforms for Jacobian computation.
    ///
    /// Returns (joint_origins_in_base, joint_axes_in_base, ee_position).
    pub fn joint_frames(&self, q: &[f32]) -> (Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vector3<f32>) {
        assert_eq!(q.len(), self.dof());

        let mut transform = Isometry3::identity();
        let mut origins = Vec::with_capacity(self.dof());
        let mut axes = Vec::with_capacity(self.dof());

        for (joint, &angle) in self.joints.iter().zip(q.iter()) {
            transform *= joint.origin;

            // Record joint origin and axis in base frame BEFORE joint rotation
            origins.push(transform.translation.vector);
            axes.push(transform.rotation * joint.axis.into_inner());

            // Apply joint motion
            transform *= joint_transform(&joint.axis, joint.is_prismatic, angle);
        }

        // End-effector position
        let ee_transform = transform * self.ee_offset;
        let ee_pos = ee_transform.translation.vector;

        (origins, axes, ee_pos)
    }

    /// Clamp joint positions to their limits.
    pub fn clamp_joints(&self, q: &mut [f32]) {
        for (i, joint) in self.joints.iter().enumerate() {
            q[i] = q[i].clamp(joint.lower_limit, joint.upper_limit);
        }
    }
}

/// Convert a URDF [`Origin`] (xyz + rpy) to an [`Isometry3`].
fn origin_to_isometry(origin: &Origin) -> Isometry3<f32> {
    let translation = Translation3::new(origin.xyz[0], origin.xyz[1], origin.xyz[2]);
    let rotation = UnitQuaternion::from_matrix(&rotation_matrix_from_rpy(
        origin.rpy[0],
        origin.rpy[1],
        origin.rpy[2],
    ));
    Isometry3::from_parts(translation, rotation)
}

/// Build a rotation matrix from roll-pitch-yaw (intrinsic XYZ / extrinsic ZYX).
fn rotation_matrix_from_rpy(roll: f32, pitch: f32, yaw: f32) -> Matrix3<f32> {
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();

    // Extrinsic ZYX = Intrinsic XYZ
    Matrix3::new(
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    )
}

/// Compute the transform for a single joint at a given position.
fn joint_transform(
    axis: &UnitVector3<f32>,
    is_prismatic: bool,
    position: f32,
) -> Isometry3<f32> {
    if is_prismatic {
        Isometry3::from_parts(
            Translation3::from(axis.into_inner() * position),
            UnitQuaternion::identity(),
        )
    } else {
        Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_axis_angle(axis, position),
        )
    }
}

/// Find the ordered list of joint names from `root` to `target_link`.
fn find_path_to_link(model: &RobotModel, root: &str, target: &str) -> Option<Vec<String>> {
    if root == target {
        return Some(Vec::new());
    }

    // For each joint, try DFS from root to target
    for joint in model.joints.values() {
        if joint.parent == root {
            if joint.child == target {
                return Some(vec![joint.name.clone()]);
            }
            if let Some(mut path) = find_path_to_link(model, &joint.child, target) {
                path.insert(0, joint.name.clone());
                return Some(path);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use clankers_urdf::parse_string;

    const TWO_LINK_ARM: &str = r#"
        <robot name="two_link_arm">
            <link name="base"><inertial><mass value="10.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
            <link name="upper_arm"><inertial><mass value="2.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.002"/></inertial></link>
            <link name="forearm"><inertial><mass value="1.0"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/></inertial></link>
            <link name="end_effector"><inertial><mass value="0.1"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial></link>
            <joint name="shoulder" type="revolute">
                <parent link="base"/><child link="upper_arm"/>
                <origin xyz="0 0 0.05" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
                <limit lower="-2.617" upper="2.617" effort="50" velocity="3"/>
            </joint>
            <joint name="elbow" type="revolute">
                <parent link="upper_arm"/><child link="forearm"/>
                <origin xyz="0 0 0.3" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
                <limit lower="-2.094" upper="2.094" effort="30" velocity="5"/>
            </joint>
            <joint name="ee_fixed" type="fixed">
                <parent link="forearm"/><child link="end_effector"/>
                <origin xyz="0 0 0.25"/>
            </joint>
        </robot>
    "#;

    #[test]
    fn chain_from_two_link_arm() {
        let model = parse_string(TWO_LINK_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();
        assert_eq!(chain.dof(), 2);
        assert_eq!(chain.joint_names(), vec!["shoulder", "elbow"]);
    }

    #[test]
    fn fk_zero_position() {
        let model = parse_string(TWO_LINK_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();

        // At q=[0,0], EE should be straight up:
        // base_offset=0.05 + upper_arm=0.3 + forearm(ee_fixed)=0.25 = 0.6
        let ee = chain.forward_kinematics(&[0.0, 0.0]);
        assert_relative_eq!(ee.translation.x, 0.0, epsilon = 1e-5);
        assert_relative_eq!(ee.translation.y, 0.0, epsilon = 1e-5);
        assert_relative_eq!(ee.translation.z, 0.6, epsilon = 1e-5);
    }

    #[test]
    fn fk_shoulder_90_deg() {
        let model = parse_string(TWO_LINK_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();

        // Shoulder rotated 90 deg about Z. Both joints are Z-axis,
        // so upper_arm and forearm extend in the Y direction.
        let q = [std::f32::consts::FRAC_PI_2, 0.0];
        let ee = chain.forward_kinematics(&q);
        // The arm extends along +Z from base, but with Z-axis rotation
        // the links still go up (Z) since they're cylinders along Z.
        // shoulder rotates in XY plane: upper_arm origin is at z=0.3 from shoulder,
        // but since joint axis is Z and link extends along Z, rotation about Z
        // doesn't change the Z-height. EE stays at same height.
        assert_relative_eq!(ee.translation.z, 0.6, epsilon = 1e-5);
    }

    #[test]
    fn fk_matches_manual_2d() {
        // For a 2-link arm with Z-axis joints, rotation about Z means
        // the links extend in the XY plane at the Z heights.
        // With both joints at 0, everything is along Z.
        let model = parse_string(TWO_LINK_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();

        let ee = chain.forward_kinematics(&[0.0, 0.0]);
        assert_relative_eq!(ee.translation.z, 0.6, epsilon = 1e-5);
    }

    #[test]
    fn chain_from_nonexistent_link_returns_none() {
        let model = parse_string(TWO_LINK_ARM).unwrap();
        assert!(KinematicChain::from_model(&model, "nonexistent").is_none());
    }

    #[test]
    fn clamp_joints() {
        let model = parse_string(TWO_LINK_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();

        let mut q = [5.0, -5.0];
        chain.clamp_joints(&mut q);
        assert_relative_eq!(q[0], 2.617, epsilon = 1e-5);
        assert_relative_eq!(q[1], -2.094, epsilon = 1e-5);
    }

    #[test]
    fn origin_to_isometry_identity() {
        let origin = Origin::default();
        let iso = origin_to_isometry(&origin);
        let identity = Isometry3::<f32>::identity();
        assert_relative_eq!(iso.translation.x, identity.translation.x, epsilon = 1e-6);
        assert_relative_eq!(iso.translation.y, identity.translation.y, epsilon = 1e-6);
        assert_relative_eq!(iso.translation.z, identity.translation.z, epsilon = 1e-6);
    }

    #[test]
    fn origin_to_isometry_translation() {
        let origin = Origin {
            xyz: [1.0, 2.0, 3.0],
            rpy: [0.0, 0.0, 0.0],
        };
        let iso = origin_to_isometry(&origin);
        assert_relative_eq!(iso.translation.x, 1.0, epsilon = 1e-6);
        assert_relative_eq!(iso.translation.y, 2.0, epsilon = 1e-6);
        assert_relative_eq!(iso.translation.z, 3.0, epsilon = 1e-6);
    }

    const SIX_DOF_ARM: &str = r#"
        <robot name="six_dof_arm">
            <link name="base"><inertial><mass value="20.0"/><inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.5"/></inertial></link>
            <link name="shoulder_link"><inertial><mass value="3.0"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/></inertial></link>
            <link name="upper_arm"><inertial><mass value="2.5"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.003"/></inertial></link>
            <link name="elbow_link"><inertial><mass value="1.5"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.002"/></inertial></link>
            <link name="forearm"><inertial><mass value="1.0"/><inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.001"/></inertial></link>
            <link name="wrist_link"><inertial><mass value="0.5"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.0005"/></inertial></link>
            <link name="end_effector"><inertial><mass value="0.2"/><inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/></inertial></link>
            <joint name="j1_base_yaw" type="revolute">
                <parent link="base"/><child link="shoulder_link"/>
                <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
                <limit lower="-3.14159" upper="3.14159" effort="80" velocity="2"/>
            </joint>
            <joint name="j2_shoulder_pitch" type="revolute">
                <parent link="shoulder_link"/><child link="upper_arm"/>
                <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
                <limit lower="-1.5708" upper="2.356" effort="60" velocity="2"/>
            </joint>
            <joint name="j3_elbow_pitch" type="revolute">
                <parent link="upper_arm"/><child link="elbow_link"/>
                <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
                <limit lower="-2.356" upper="2.356" effort="40" velocity="3"/>
            </joint>
            <joint name="j4_forearm_roll" type="revolute">
                <parent link="elbow_link"/><child link="forearm"/>
                <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
                <limit lower="-3.14159" upper="3.14159" effort="20" velocity="5"/>
            </joint>
            <joint name="j5_wrist_pitch" type="revolute">
                <parent link="forearm"/><child link="wrist_link"/>
                <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
                <limit lower="-2.094" upper="2.094" effort="10" velocity="5"/>
            </joint>
            <joint name="j6_wrist_roll" type="revolute">
                <parent link="wrist_link"/><child link="end_effector"/>
                <origin xyz="0 0 0.06"/><axis xyz="0 0 1"/>
                <limit lower="-3.14159" upper="3.14159" effort="5" velocity="8"/>
            </joint>
        </robot>
    "#;

    #[test]
    fn chain_from_six_dof_arm() {
        let model = parse_string(SIX_DOF_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();
        assert_eq!(chain.dof(), 6);
    }

    #[test]
    fn fk_six_dof_zero() {
        let model = parse_string(SIX_DOF_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();

        let ee = chain.forward_kinematics(&[0.0; 6]);
        // Sum of Z offsets: 0.05 + 0.2 + 0.3 + 0.1 + 0.2 + 0.06 = 0.91
        assert_relative_eq!(ee.translation.z, 0.91, epsilon = 1e-4);
        assert_relative_eq!(ee.translation.x, 0.0, epsilon = 1e-5);
        assert_relative_eq!(ee.translation.y, 0.0, epsilon = 1e-5);
    }
}
