"""Pure-Python Damped Least Squares (DLS) inverse kinematics solver.

This module is self-contained and uses only numpy and the Python standard
library.  It does **not** import from the ``clankers`` package.

Provides:
- ``JointInfo`` / ``KinematicChain`` -- minimal kinematic chain representation
- ``forward_kinematics`` -- compute the 4x4 end-effector transform
- ``compute_jacobian`` -- 6xN geometric Jacobian
- ``DlsSolver`` -- iterative DLS IK with joint-limit clamping
- ``IKResult`` -- solver output dataclass
- ``rpy_to_rotation_matrix`` / ``homogeneous_transform`` -- helpers
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll-pitch-yaw (XYZ extrinsic / ZYX intrinsic) to a 3x3 rotation matrix.

    Convention follows URDF/SDF: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def homogeneous_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula: 3x3 rotation about a unit *axis* by *angle* (rad)."""
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _orientation_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Compute the 3-vector orientation error between two 3x3 rotation matrices.

    Uses the skew-symmetric part of R_err = R_target @ R_current.T to extract
    the axis-angle error as a 3-vector.
    """
    R_err = R_target @ R_current.T
    # Extract the skew-symmetric components (equivalent to log(R_err) for small angles)
    error = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )
    return 0.5 * error


# ---------------------------------------------------------------------------
# Kinematic chain data structures
# ---------------------------------------------------------------------------


@dataclass
class JointInfo:
    """Description of a single joint in the kinematic chain."""

    name: str
    type: str  # "revolute" or "prismatic"
    axis: np.ndarray  # 3D unit vector (joint axis in local frame)
    origin_xyz: np.ndarray  # translation from parent
    origin_rpy: np.ndarray  # rotation from parent (roll, pitch, yaw)
    lower_limit: float = -np.pi
    upper_limit: float = np.pi


@dataclass
class KinematicChain:
    """Ordered list of active joints forming a serial kinematic chain.

    The chain runs from a base frame to an end-effector frame.  An optional
    ``ee_offset`` 4x4 transform can be appended after the last joint.
    """

    joints: list[JointInfo] = field(default_factory=list)
    ee_offset: np.ndarray = field(default_factory=lambda: np.eye(4))

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_joint_info(
        cls,
        joints: list[JointInfo],
        ee_offset: np.ndarray | None = None,
    ) -> KinematicChain:
        """Build a chain from a list of ``JointInfo`` objects.

        Parameters
        ----------
        joints:
            Ordered list of joints from base to tip.
        ee_offset:
            Optional constant 4x4 transform (or 3-vector translation) appended
            after the last joint to reach the end-effector frame.
        """
        if ee_offset is None:
            T_ee = np.eye(4)
        elif ee_offset.shape == (3,):
            T_ee = homogeneous_transform(np.eye(3), ee_offset)
        elif ee_offset.shape == (4, 4):
            T_ee = ee_offset.copy()
        else:
            raise ValueError(
                f"ee_offset must be a 3-vector or 4x4 matrix, got shape {ee_offset.shape}"
            )
        return cls(joints=list(joints), ee_offset=T_ee)

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        base_link: str,
        ee_link: str,
    ) -> KinematicChain:
        """Parse a URDF file and extract the serial chain between two links.

        Only **revolute** and **prismatic** joints are treated as active; fixed
        joints contribute their static transform but do not add a DOF.

        Parameters
        ----------
        urdf_path:
            Path to the URDF XML file.
        base_link:
            Name of the base link (start of the chain).
        ee_link:
            Name of the end-effector link (end of the chain).
        """
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Build look-ups: child_link -> joint element
        child_to_joint: dict[str, ET.Element] = {}
        for joint_el in root.iter("joint"):
            child_el = joint_el.find("child")
            if child_el is not None:
                child_to_joint[child_el.get("link", "")] = joint_el

        # Walk from ee_link back to base_link collecting joints
        joint_elements: list[ET.Element] = []
        current = ee_link
        while current != base_link:
            if current not in child_to_joint:
                raise ValueError(
                    f"Cannot trace chain from '{ee_link}' to '{base_link}': "
                    f"link '{current}' has no parent joint in the URDF."
                )
            jel = child_to_joint[current]
            joint_elements.append(jel)
            parent_el = jel.find("parent")
            if parent_el is None:
                raise ValueError(f"Joint '{jel.get('name')}' has no <parent> element.")
            current = parent_el.get("link", "")

        joint_elements.reverse()  # base -> ee order

        # Convert to JointInfo, folding fixed-joint transforms into the next
        # active joint's origin.
        joints: list[JointInfo] = []
        pending_T = np.eye(4)  # accumulated fixed-joint transform

        for jel in joint_elements:
            jtype = jel.get("type", "fixed")

            # Parse <origin>
            origin_el = jel.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin_el is not None:
                xyz_str = origin_el.get("xyz", "0 0 0")
                rpy_str = origin_el.get("rpy", "0 0 0")
                xyz = np.array([float(v) for v in xyz_str.split()])
                rpy = np.array([float(v) for v in rpy_str.split()])

            T_origin = homogeneous_transform(rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2]), xyz)

            if jtype == "fixed":
                pending_T = pending_T @ T_origin
                continue

            # Parse <axis>
            axis_el = jel.find("axis")
            if axis_el is not None:
                axis = np.array([float(v) for v in axis_el.get("xyz", "0 0 1").split()])
            else:
                axis = np.array([0.0, 0.0, 1.0])

            # Parse <limit>
            limit_el = jel.find("limit")
            lower = -np.pi
            upper = np.pi
            if limit_el is not None:
                lower = float(limit_el.get("lower", str(-np.pi)))
                upper = float(limit_el.get("upper", str(np.pi)))

            # Fold any pending fixed-joint transforms into this joint's origin
            T_combined = pending_T @ T_origin
            combined_xyz = T_combined[:3, 3]
            # Recover RPY from the combined rotation for storage
            R_combined = T_combined[:3, :3]
            combined_rpy = _rotation_matrix_to_rpy(R_combined)
            pending_T = np.eye(4)

            joints.append(
                JointInfo(
                    name=jel.get("name", ""),
                    type=jtype,
                    axis=axis,
                    origin_xyz=combined_xyz,
                    origin_rpy=combined_rpy,
                    lower_limit=lower,
                    upper_limit=upper,
                )
            )

        # Any remaining fixed transform after the last active joint becomes
        # the ee_offset.
        return cls(joints=joints, ee_offset=pending_T)


def _rotation_matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Extract roll-pitch-yaw from a 3x3 rotation matrix (ZYX convention)."""
    sy = -R[2, 0]
    # Clamp to avoid numerical issues with arcsin
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)

    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0

    return np.array([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


def forward_kinematics(chain: KinematicChain, joint_angles: np.ndarray) -> np.ndarray:
    """Compute the 4x4 homogeneous transform of the end-effector.

    Parameters
    ----------
    chain:
        The kinematic chain definition.
    joint_angles:
        Array of length ``len(chain.joints)`` with the current joint values
        (angles in rad for revolute, displacement in metres for prismatic).

    Returns
    -------
    np.ndarray
        4x4 homogeneous transform from the base frame to the end-effector.
    """
    if len(joint_angles) != len(chain.joints):
        raise ValueError(f"Expected {len(chain.joints)} joint values, got {len(joint_angles)}.")

    T = np.eye(4)
    for i, joint in enumerate(chain.joints):
        # Static origin transform for this joint
        R_origin = rpy_to_rotation_matrix(
            joint.origin_rpy[0], joint.origin_rpy[1], joint.origin_rpy[2]
        )
        T_origin = homogeneous_transform(R_origin, joint.origin_xyz)

        # Joint-variable transform
        q = joint_angles[i]
        if joint.type == "revolute":
            R_joint = _rotation_about_axis(joint.axis, q)
            T_joint = homogeneous_transform(R_joint, np.zeros(3))
        elif joint.type == "prismatic":
            T_joint = homogeneous_transform(np.eye(3), joint.axis * q)
        else:
            T_joint = np.eye(4)

        T = T @ T_origin @ T_joint

    # Append end-effector offset
    T = T @ chain.ee_offset
    return T


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------


def compute_jacobian(chain: KinematicChain, joint_angles: np.ndarray) -> np.ndarray:
    """Compute the 6xN geometric Jacobian (linear + angular rows).

    The top 3 rows are the linear velocity Jacobian and the bottom 3 rows are
    the angular velocity Jacobian.

    For revolute joint *i*:
        J_v_i = z_i x (p_ee - p_i)
        J_w_i = z_i

    For prismatic joint *i*:
        J_v_i = z_i
        J_w_i = 0
    """
    n = len(chain.joints)
    if len(joint_angles) != n:
        raise ValueError(f"Expected {n} joint values, got {len(joint_angles)}.")

    # Pre-compute cumulative transforms up to each joint origin
    T_cumulative = [np.eye(4)]  # T_cumulative[i] = base -> joint_i origin (before joint rot)
    T = np.eye(4)
    for i, joint in enumerate(chain.joints):
        R_origin = rpy_to_rotation_matrix(
            joint.origin_rpy[0], joint.origin_rpy[1], joint.origin_rpy[2]
        )
        T_origin = homogeneous_transform(R_origin, joint.origin_xyz)

        q = joint_angles[i]
        if joint.type == "revolute":
            R_joint = _rotation_about_axis(joint.axis, q)
            T_joint = homogeneous_transform(R_joint, np.zeros(3))
        elif joint.type == "prismatic":
            T_joint = homogeneous_transform(np.eye(3), joint.axis * q)
        else:
            T_joint = np.eye(4)

        T = T @ T_origin
        T_cumulative.append(T.copy())  # store transform at joint origin (before joint rotation)
        T = T @ T_joint

    T_ee = T @ chain.ee_offset
    p_ee = T_ee[:3, 3]

    J = np.zeros((6, n))

    # Reconstruct per-joint z-axis and position in world frame
    T_accum = np.eye(4)
    for i, joint in enumerate(chain.joints):
        R_origin = rpy_to_rotation_matrix(
            joint.origin_rpy[0], joint.origin_rpy[1], joint.origin_rpy[2]
        )
        T_origin = homogeneous_transform(R_origin, joint.origin_xyz)

        T_accum = T_accum @ T_origin

        # z_i = joint axis expressed in world frame
        z_i = T_accum[:3, :3] @ joint.axis
        p_i = T_accum[:3, 3]

        if joint.type == "revolute":
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            J[3:, i] = z_i
        elif joint.type == "prismatic":
            J[:3, i] = z_i
            J[3:, i] = 0.0

        # Now apply joint rotation for accumulation to next joint
        q = joint_angles[i]
        if joint.type == "revolute":
            R_joint = _rotation_about_axis(joint.axis, q)
            T_joint = homogeneous_transform(R_joint, np.zeros(3))
        elif joint.type == "prismatic":
            T_joint = homogeneous_transform(np.eye(3), joint.axis * q)
        else:
            T_joint = np.eye(4)
        T_accum = T_accum @ T_joint

    return J


# ---------------------------------------------------------------------------
# IK result
# ---------------------------------------------------------------------------


@dataclass
class IKResult:
    """Result returned by the DLS IK solver."""

    joint_angles: np.ndarray
    position_error: float
    converged: bool
    iterations: int


# ---------------------------------------------------------------------------
# DLS IK solver
# ---------------------------------------------------------------------------


class DlsSolver:
    """Damped Least Squares (DLS) iterative IK solver.

    Parameters
    ----------
    chain:
        Kinematic chain definition.
    max_iterations:
        Maximum number of solver iterations.
    tolerance:
        Position error tolerance in metres.
    damping:
        Damping factor (lambda) for the DLS pseudo-inverse.
    """

    def __init__(
        self,
        chain: KinematicChain,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        damping: float = 0.01,
    ) -> None:
        self.chain = chain
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping

    def solve(
        self,
        target_position: np.ndarray,
        initial_angles: np.ndarray,
        target_orientation: np.ndarray | None = None,
    ) -> IKResult:
        """Solve for joint angles that place the end-effector at the target.

        Parameters
        ----------
        target_position:
            Desired XYZ position of the end-effector (3-vector).
        initial_angles:
            Starting joint configuration (array of length N).
        target_orientation:
            Optional 3x3 rotation matrix for the desired end-effector
            orientation.  When provided, the solver minimises a full 6-DOF
            error (3 position + 3 orientation).

        Returns
        -------
        IKResult
            Contains the solution joint angles, residual position error,
            convergence flag, and iteration count.
        """
        target_position = np.asarray(target_position, dtype=float)
        q = np.asarray(initial_angles, dtype=float).copy()

        lower = np.array([j.lower_limit for j in self.chain.joints])
        upper = np.array([j.upper_limit for j in self.chain.joints])

        use_orientation = target_orientation is not None
        R_target: np.ndarray = np.eye(3)
        if use_orientation:
            R_target = np.asarray(target_orientation, dtype=float)

        lam2 = self.damping**2

        converged = False
        pos_error = float("inf")
        _iteration = 0

        for _iteration in range(self.max_iterations):
            T_ee = forward_kinematics(self.chain, q)
            current_pos = T_ee[:3, 3]
            e_pos = target_position - current_pos
            pos_error = float(np.linalg.norm(e_pos))

            if use_orientation:
                R_current = T_ee[:3, :3]
                e_orient = _orientation_error(R_target, R_current)
                e = np.concatenate([e_pos, e_orient])
                J_full = compute_jacobian(self.chain, q)
                J = J_full  # 6xN
                m = 6
            else:
                e = e_pos
                J_full = compute_jacobian(self.chain, q)
                J = J_full[:3, :]  # 3xN (position rows only)
                m = 3

            if pos_error < self.tolerance:
                converged = True
                break

            # DLS update: dq = J^T (J J^T + lambda^2 I)^{-1} e
            JJT = J @ J.T + lam2 * np.eye(m)
            dq = J.T @ np.linalg.solve(JJT, e)

            q = q + dq
            q = np.clip(q, lower, upper)

        # Final error evaluation
        T_ee = forward_kinematics(self.chain, q)
        pos_error = float(np.linalg.norm(target_position - T_ee[:3, 3]))
        if pos_error < self.tolerance:
            converged = True

        return IKResult(
            joint_angles=q,
            position_error=pos_error,
            converged=converged,
            iterations=min(_iteration + 1, self.max_iterations) if self.max_iterations > 0 else 0,
        )
