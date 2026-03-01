"""Tests for clankers_synthetic.ik_solver -- DLS IK solver."""

from __future__ import annotations

import numpy as np
import pytest

from clankers_synthetic.ik_solver import (
    DlsSolver,
    IKResult,
    JointInfo,
    KinematicChain,
    compute_jacobian,
    forward_kinematics,
    homogeneous_transform,
    rpy_to_rotation_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_2link_chain(l1: float = 1.0, l2: float = 1.0) -> KinematicChain:
    """2-link planar arm: both revolute joints rotating around Z axis."""
    joints = [
        JointInfo(
            name="j1",
            type="revolute",
            axis=np.array([0.0, 0.0, 1.0]),
            origin_xyz=np.array([0.0, 0.0, 0.0]),
            origin_rpy=np.array([0.0, 0.0, 0.0]),
            lower_limit=-np.pi,
            upper_limit=np.pi,
        ),
        JointInfo(
            name="j2",
            type="revolute",
            axis=np.array([0.0, 0.0, 1.0]),
            origin_xyz=np.array([l1, 0.0, 0.0]),
            origin_rpy=np.array([0.0, 0.0, 0.0]),
            lower_limit=-np.pi,
            upper_limit=np.pi,
        ),
    ]
    return KinematicChain.from_joint_info(joints, ee_offset=np.array([l2, 0.0, 0.0]))


def make_3dof_chain() -> KinematicChain:
    """3-DOF chain in 3D: Z-rotation, Y-rotation, Z-rotation with unit link lengths."""
    joints = [
        JointInfo(
            name="j1",
            type="revolute",
            axis=np.array([0.0, 0.0, 1.0]),
            origin_xyz=np.array([0.0, 0.0, 0.0]),
            origin_rpy=np.array([0.0, 0.0, 0.0]),
            lower_limit=-np.pi,
            upper_limit=np.pi,
        ),
        JointInfo(
            name="j2",
            type="revolute",
            axis=np.array([0.0, 1.0, 0.0]),
            origin_xyz=np.array([0.0, 0.0, 1.0]),
            origin_rpy=np.array([0.0, 0.0, 0.0]),
            lower_limit=-np.pi,
            upper_limit=np.pi,
        ),
        JointInfo(
            name="j3",
            type="revolute",
            axis=np.array([0.0, 0.0, 1.0]),
            origin_xyz=np.array([1.0, 0.0, 0.0]),
            origin_rpy=np.array([0.0, 0.0, 0.0]),
            lower_limit=-np.pi,
            upper_limit=np.pi,
        ),
    ]
    return KinematicChain.from_joint_info(joints, ee_offset=np.array([1.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestRPYToRotationMatrix:
    def test_identity(self):
        """rpy(0, 0, 0) should give the identity rotation."""
        R = rpy_to_rotation_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90deg_yaw(self):
        """90-degree yaw rotation around Z."""
        R = rpy_to_rotation_matrix(0.0, 0.0, np.pi / 2)
        expected = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_90deg_pitch(self):
        """90-degree pitch rotation around Y."""
        R = rpy_to_rotation_matrix(0.0, np.pi / 2, 0.0)
        expected = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_90deg_roll(self):
        """90-degree roll rotation around X."""
        R = rpy_to_rotation_matrix(np.pi / 2, 0.0, 0.0)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_orthogonality(self):
        """Result should always be orthogonal."""
        R = rpy_to_rotation_matrix(0.3, -0.7, 1.2)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


class TestHomogeneousTransform:
    def test_identity(self):
        T = homogeneous_transform(np.eye(3), np.zeros(3))
        np.testing.assert_allclose(T, np.eye(4), atol=1e-15)

    def test_translation_only(self):
        T = homogeneous_transform(np.eye(3), np.array([1.0, 2.0, 3.0]))
        assert T[0, 3] == 1.0
        assert T[1, 3] == 2.0
        assert T[2, 3] == 3.0
        np.testing.assert_allclose(T[:3, :3], np.eye(3))


# ---------------------------------------------------------------------------
# KinematicChain tests
# ---------------------------------------------------------------------------


class TestFromJointInfo:
    def test_builds_chain(self):
        """from_joint_info should create a chain with the correct number of joints."""
        chain = make_2link_chain()
        assert len(chain.joints) == 2
        assert chain.joints[0].name == "j1"
        assert chain.joints[1].name == "j2"

    def test_ee_offset_vector(self):
        """A 3-vector ee_offset should be converted to a 4x4 translation."""
        chain = make_2link_chain(l1=1.0, l2=0.5)
        expected_ee = homogeneous_transform(np.eye(3), np.array([0.5, 0.0, 0.0]))
        np.testing.assert_allclose(chain.ee_offset, expected_ee, atol=1e-12)

    def test_ee_offset_matrix(self):
        """A 4x4 ee_offset should be stored as-is."""
        T = homogeneous_transform(np.eye(3), np.array([0.0, 0.0, 0.3]))
        joints = [
            JointInfo(
                name="j1",
                type="revolute",
                axis=np.array([0, 0, 1]),
                origin_xyz=np.zeros(3),
                origin_rpy=np.zeros(3),
            )
        ]
        chain = KinematicChain.from_joint_info(joints, ee_offset=T)
        np.testing.assert_allclose(chain.ee_offset, T, atol=1e-12)

    def test_no_ee_offset(self):
        """No ee_offset should default to identity."""
        joints = [
            JointInfo(
                name="j1",
                type="revolute",
                axis=np.array([0, 0, 1]),
                origin_xyz=np.zeros(3),
                origin_rpy=np.zeros(3),
            )
        ]
        chain = KinematicChain.from_joint_info(joints)
        np.testing.assert_allclose(chain.ee_offset, np.eye(4), atol=1e-12)


# ---------------------------------------------------------------------------
# Forward kinematics tests
# ---------------------------------------------------------------------------


class TestForwardKinematics:
    def test_identity_chain(self):
        """Single joint at origin with zero angle: EE at the ee_offset position."""
        joints = [
            JointInfo(
                name="j1",
                type="revolute",
                axis=np.array([0.0, 0.0, 1.0]),
                origin_xyz=np.zeros(3),
                origin_rpy=np.zeros(3),
            )
        ]
        chain = KinematicChain.from_joint_info(joints, ee_offset=np.array([1.0, 0.0, 0.0]))
        T = forward_kinematics(chain, np.array([0.0]))
        np.testing.assert_allclose(T[:3, 3], [1.0, 0.0, 0.0], atol=1e-12)

    def test_2link_planar_zero_angles(self):
        """2-link planar arm at zero angles: EE at (l1+l2, 0, 0)."""
        chain = make_2link_chain(l1=1.0, l2=1.0)
        T = forward_kinematics(chain, np.array([0.0, 0.0]))
        np.testing.assert_allclose(T[:3, 3], [2.0, 0.0, 0.0], atol=1e-12)

    def test_2link_planar_90deg(self):
        """2-link planar arm with j1=pi/2, j2=0: EE at (0, l1+l2, 0)."""
        chain = make_2link_chain(l1=1.0, l2=1.0)
        T = forward_kinematics(chain, np.array([np.pi / 2, 0.0]))
        np.testing.assert_allclose(T[:3, 3], [0.0, 2.0, 0.0], atol=1e-10)

    def test_2link_planar_folded(self):
        """2-link arm with j2=pi: second link folds back. EE at (0, 0, 0)."""
        chain = make_2link_chain(l1=1.0, l2=1.0)
        T = forward_kinematics(chain, np.array([0.0, np.pi]))
        np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, 0.0], atol=1e-10)

    def test_2link_planar_general(self):
        """2-link arm: verify against analytic FK."""
        l1, l2 = 1.0, 0.8
        chain = make_2link_chain(l1=l1, l2=l2)
        q1, q2 = 0.5, -0.3
        T = forward_kinematics(chain, np.array([q1, q2]))
        # Analytic: x = l1*cos(q1) + l2*cos(q1+q2), y = l1*sin(q1) + l2*sin(q1+q2)
        expected_x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        expected_y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        np.testing.assert_allclose(T[0, 3], expected_x, atol=1e-10)
        np.testing.assert_allclose(T[1, 3], expected_y, atol=1e-10)
        np.testing.assert_allclose(T[2, 3], 0.0, atol=1e-10)

    def test_wrong_number_of_angles_raises(self):
        chain = make_2link_chain()
        with pytest.raises(ValueError, match="Expected 2"):
            forward_kinematics(chain, np.array([0.0]))


# ---------------------------------------------------------------------------
# Jacobian tests
# ---------------------------------------------------------------------------


class TestJacobian:
    def test_shape(self):
        """Jacobian should be 6xN."""
        chain = make_2link_chain()
        J = compute_jacobian(chain, np.array([0.0, 0.0]))
        assert J.shape == (6, 2)

    def test_2link_planar_at_zero(self):
        """Verify analytical Jacobian for 2-link planar at q=[0,0]."""
        l1, l2 = 1.0, 1.0
        chain = make_2link_chain(l1=l1, l2=l2)
        J = compute_jacobian(chain, np.array([0.0, 0.0]))
        # At q=[0,0]: EE at (2,0,0)
        # j1 at origin, z1=[0,0,1], p_ee-p1=[2,0,0]
        # J_v1 = z1 x (p_ee-p1) = [0,0,1] x [2,0,0] = [0,2,0]
        # j2 at (1,0,0), z2=[0,0,1], p_ee-p2=[1,0,0]
        # J_v2 = z2 x (p_ee-p2) = [0,0,1] x [1,0,0] = [0,1,0]
        np.testing.assert_allclose(J[0, 0], 0.0, atol=1e-10)  # dx/dq1
        np.testing.assert_allclose(J[1, 0], 2.0, atol=1e-10)  # dy/dq1
        np.testing.assert_allclose(J[0, 1], 0.0, atol=1e-10)  # dx/dq2
        np.testing.assert_allclose(J[1, 1], 1.0, atol=1e-10)  # dy/dq2

    def test_finite_difference_agreement(self):
        """Jacobian should approximately match finite-difference computation."""
        chain = make_2link_chain(l1=1.0, l2=0.8)
        q = np.array([0.5, -0.3])
        J = compute_jacobian(chain, q)

        eps = 1e-6
        J_fd = np.zeros((6, 2))
        T0 = forward_kinematics(chain, q)
        p0 = T0[:3, 3]
        for i in range(2):
            q_plus = q.copy()
            q_plus[i] += eps
            T_plus = forward_kinematics(chain, q_plus)
            p_plus = T_plus[:3, 3]
            J_fd[:3, i] = (p_plus - p0) / eps

        # Compare position rows (linear velocity Jacobian)
        np.testing.assert_allclose(J[:3, :], J_fd[:3, :], atol=1e-4)

    def test_3dof_shape(self):
        """3-DOF chain should produce 6x3 Jacobian."""
        chain = make_3dof_chain()
        J = compute_jacobian(chain, np.array([0.0, 0.0, 0.0]))
        assert J.shape == (6, 3)


# ---------------------------------------------------------------------------
# DLS solver tests
# ---------------------------------------------------------------------------


class TestDlsSolver:
    def test_reachable_target(self):
        """Solver should reach a point within the workspace of a 2-link arm."""
        chain = make_2link_chain(l1=1.0, l2=1.0)
        solver = DlsSolver(chain, max_iterations=200, tolerance=1e-4, damping=0.01)
        # Target within reach (inside circle of radius 2)
        target = np.array([1.0, 1.0, 0.0])
        result = solver.solve(target_position=target, initial_angles=np.array([0.3, 0.3]))
        assert isinstance(result, IKResult)
        assert result.converged is True
        assert result.position_error < 1e-4

    def test_convergence_details(self):
        """Verify converged=True, error < tolerance, and iterations > 0."""
        chain = make_2link_chain(l1=1.0, l2=1.0)
        solver = DlsSolver(chain, max_iterations=200, tolerance=1e-4, damping=0.01)
        target = np.array([1.2, 0.8, 0.0])
        result = solver.solve(target_position=target, initial_angles=np.array([0.3, -0.3]))
        assert result.converged is True
        assert result.position_error < solver.tolerance
        assert result.iterations > 0

    def test_unreachable_target(self):
        """Target beyond workspace: converged=False, returns closest reachable point."""
        chain = make_2link_chain(l1=1.0, l2=1.0)
        solver = DlsSolver(chain, max_iterations=200, tolerance=1e-4, damping=0.01)
        # Target at distance 5, workspace radius is 2
        target = np.array([5.0, 0.0, 0.0])
        result = solver.solve(target_position=target, initial_angles=np.array([0.0, 0.0]))
        assert result.converged is False
        # Should be close to the fully extended arm (distance ~2)
        T_final = forward_kinematics(chain, result.joint_angles)
        reach = np.linalg.norm(T_final[:3, 3])
        assert reach < 2.1  # within workspace + margin

    def test_joint_limit_clamping(self):
        """Joint angles must stay within the specified limits."""
        joints = [
            JointInfo(
                name="j1",
                type="revolute",
                axis=np.array([0.0, 0.0, 1.0]),
                origin_xyz=np.zeros(3),
                origin_rpy=np.zeros(3),
                lower_limit=-0.5,
                upper_limit=0.5,
            ),
            JointInfo(
                name="j2",
                type="revolute",
                axis=np.array([0.0, 0.0, 1.0]),
                origin_xyz=np.array([1.0, 0.0, 0.0]),
                origin_rpy=np.zeros(3),
                lower_limit=-0.5,
                upper_limit=0.5,
            ),
        ]
        chain = KinematicChain.from_joint_info(joints, ee_offset=np.array([1.0, 0.0, 0.0]))
        solver = DlsSolver(chain, max_iterations=200, tolerance=1e-4, damping=0.01)
        # Target that would require large joint angles
        target = np.array([0.0, 2.0, 0.0])
        result = solver.solve(target_position=target, initial_angles=np.array([0.0, 0.0]))
        # All angles should be within limits
        for i, joint in enumerate(chain.joints):
            assert result.joint_angles[i] >= joint.lower_limit - 1e-10
            assert result.joint_angles[i] <= joint.upper_limit + 1e-10

    def test_verify_fk_matches_result(self):
        """The FK of the returned angles should match the reported error."""
        chain = make_2link_chain(l1=1.0, l2=1.0)
        solver = DlsSolver(chain, max_iterations=200, tolerance=1e-4, damping=0.01)
        target = np.array([0.5, 0.8, 0.0])
        result = solver.solve(target_position=target, initial_angles=np.array([0.5, 0.5]))
        T = forward_kinematics(chain, result.joint_angles)
        actual_error = np.linalg.norm(target - T[:3, 3])
        np.testing.assert_allclose(actual_error, result.position_error, atol=1e-10)

    def test_3dof_chain(self):
        """3-DOF chain in 3D should converge for a reachable target."""
        chain = make_3dof_chain()
        solver = DlsSolver(chain, max_iterations=300, tolerance=1e-3, damping=0.05)
        # At q=[0,0,0] the EE is at (2,0,1), so target something nearby
        target = np.array([1.5, 0.5, 1.0])
        result = solver.solve(
            target_position=target,
            initial_angles=np.array([0.0, 0.0, 0.0]),
        )
        # Should converge or at least get close
        assert result.position_error < 0.1  # reasonably close

    def test_3dof_convergence(self):
        """3-DOF chain should converge for a known reachable target."""
        chain = make_3dof_chain()
        solver = DlsSolver(chain, max_iterations=500, tolerance=1e-3, damping=0.01)
        # Use a known FK result as the target so we know it is reachable.
        q_target = np.array([0.2, -0.3, 0.1])
        T_target = forward_kinematics(chain, q_target)
        target = T_target[:3, 3].copy()
        result = solver.solve(
            target_position=target,
            initial_angles=np.array([0.0, 0.0, 0.0]),
        )
        assert result.converged is True
        assert result.position_error < solver.tolerance

    def test_with_orientation(self):
        """Solver with target_orientation should minimize both position and orientation error."""
        chain = make_2link_chain(l1=1.0, l2=1.0)
        solver = DlsSolver(chain, max_iterations=300, tolerance=1e-3, damping=0.01)
        target_pos = np.array([1.0, 1.0, 0.0])
        target_orient = rpy_to_rotation_matrix(0.0, 0.0, np.pi / 2)
        result = solver.solve(
            target_position=target_pos,
            initial_angles=np.array([0.3, 0.3]),
            target_orientation=target_orient,
        )
        # Position should be close even with orientation constraint
        assert result.position_error < 0.1  # may not fully converge with 2 DOF / 6 DOF target


# ---------------------------------------------------------------------------
# Integration / edge-case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_prismatic_joint(self):
        """A single prismatic joint should translate along its axis."""
        joints = [
            JointInfo(
                name="slide",
                type="prismatic",
                axis=np.array([0.0, 0.0, 1.0]),
                origin_xyz=np.zeros(3),
                origin_rpy=np.zeros(3),
                lower_limit=0.0,
                upper_limit=2.0,
            )
        ]
        chain = KinematicChain.from_joint_info(joints)
        T = forward_kinematics(chain, np.array([0.5]))
        np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, 0.5], atol=1e-12)

    def test_mixed_revolute_prismatic(self):
        """Chain with both revolute and prismatic joints."""
        joints = [
            JointInfo(
                name="j1",
                type="revolute",
                axis=np.array([0.0, 0.0, 1.0]),
                origin_xyz=np.zeros(3),
                origin_rpy=np.zeros(3),
            ),
            JointInfo(
                name="j2",
                type="prismatic",
                axis=np.array([1.0, 0.0, 0.0]),
                origin_xyz=np.array([1.0, 0.0, 0.0]),
                origin_rpy=np.zeros(3),
                lower_limit=0.0,
                upper_limit=1.0,
            ),
        ]
        chain = KinematicChain.from_joint_info(joints)
        # j1=0, j2=0.5: translate 0.5 along local X after link offset
        T = forward_kinematics(chain, np.array([0.0, 0.5]))
        np.testing.assert_allclose(T[:3, 3], [1.5, 0.0, 0.0], atol=1e-12)
        # j1=pi/2, j2=0.5: first link rotates, then translate along rotated X
        T2 = forward_kinematics(chain, np.array([np.pi / 2, 0.5]))
        np.testing.assert_allclose(T2[:3, 3], [0.0, 1.5, 0.0], atol=1e-10)

    def test_ik_result_dataclass(self):
        """IKResult should store all fields correctly."""
        result = IKResult(
            joint_angles=np.array([0.1, 0.2]),
            position_error=0.001,
            converged=True,
            iterations=15,
        )
        assert result.converged is True
        assert result.iterations == 15
        assert result.position_error == 0.001
        np.testing.assert_allclose(result.joint_angles, [0.1, 0.2])
