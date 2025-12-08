# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Tuple

import torch

import torchcontrol as toco
from torchcontrol.transform import Transformation as T
from torchcontrol.transform import Rotation as R
from torchcontrol.utils import to_tensor


class JointImpedanceControl(toco.PolicyModule):
    """
    Impedance control in joint space.
    """

    def __init__(
        self,
        joint_pos_current,
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in joint space
            Kd: D gains in joint space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.JointSpacePD(Kp, Kd)

        # Reference pose
        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            self.joint_vel_desired,
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}


class HybridJointImpedanceControl(toco.PolicyModule):
    """
    Impedance control in joint space, but with both fixed joint gains and adaptive operational space gains.
    """

    def __init__(
        self,
        joint_pos_current,
        Kq,
        Kqd,
        Kx,
        Kxd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in Cartesian space
            Kd: D gains in Cartesian space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.HybridJointSpacePD(Kq, Kqd, Kx, Kxd)

        # Reference pose
        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            self.joint_vel_desired,
            self.robot_model.compute_jacobian(joint_pos_current),
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}


class HybridJointImpedanceControlWithFF(toco.PolicyModule):
    """
    Hybrid impedance control that also accepts an end-effector wrench feedforward term.
    """

    def __init__(
        self,
        joint_pos_current,
        Kq,
        Kqd,
        Kx,
        Kxd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kq: P gains in joint space
            Kqd: D gains in joint space
            Kx: P gains in Cartesian space
            Kxd: D gains in Cartesian space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.HybridJointSpacePD(Kq, Kqd, Kx, Kxd)

        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)
        self.ee_wrench_ff = torch.nn.Parameter(torch.zeros(6))

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            self.joint_vel_desired,
            jacobian,
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )
        self.ee_wrench_ff[3:] = 0.0
        torque_ff_wrench = jacobian.T @ self.ee_wrench_ff

        torque_out = torque_feedback + torque_feedforward + torque_ff_wrench

        return {"joint_torques": torque_out}


class CartesianImpedanceControl(toco.PolicyModule):
    """
    Performs impedance control in Cartesian space.
    Errors and feedback are computed in Cartesian space, and the resulting forces are projected back into joint space.
    """

    def __init__(
        self,
        joint_pos_current,
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in Cartesian space
            Kd: D gains in Cartesian space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.pose_pd = toco.modules.feedback.CartesianSpacePDFast(Kp, Kd)

        # Reference pose
        joint_pos_current = to_tensor(joint_pos_current)
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
        self.ee_quat_desired = torch.nn.Parameter(ee_quat_current)
        self.ee_vel_desired = torch.nn.Parameter(torch.zeros(3))
        self.ee_rvel_desired = torch.nn.Parameter(torch.zeros(3))

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        ee_twist_current = jacobian @ joint_vel_current

        wrench_feedback = self.pose_pd(
            ee_pos_current,
            ee_quat_current,
            ee_twist_current,
            self.ee_pos_desired,
            self.ee_quat_desired,
            torch.cat([self.ee_vel_desired, self.ee_rvel_desired]),
        )
        torque_feedback = jacobian.T @ wrench_feedback

        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis

        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}





class CartesianAdmittanceControl(toco.PolicyModule):
    """
    Cartesian admittance controller.

    An admittance filter integrates external wrench into an inner desired
    pose/velocity, which is then tracked by a Cartesian impedance outer loop.
    
    Based on the C++ reference implementation (admittance.hpp):
    - Error is computed between inner_SE3 and desired SE3 (not current and inner)
    - Integration uses proper SE3 manifold integration via exp map
    - Wrench is transformed to world-aligned frame before use
    """

    def __init__(
        self,
        joint_pos_current,
        adm_mass: torch.Tensor,
        adm_damping: torch.Tensor,
        adm_stiffness: torch.Tensor,
        Kx: torch.Tensor,
        Kxd: torch.Tensor,
        robot_model: torch.nn.Module,
        dt: float,
        ignore_gravity=True,
        nullspace_damping: torch.Tensor = None,
    ):
        super().__init__()

        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.pose_pd = toco.modules.feedback.CartesianSpacePDFast(Kx, Kxd)

        joint_pos_current = to_tensor(joint_pos_current)
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(
            joint_pos_current
        )

        # Inner (integrated) state - corresponds to inner_SE3_ and inner_motion_ in C++
        self.register_buffer("inner_pos", ee_pos_current)
        self.register_buffer("inner_quat", ee_quat_current)
        self.register_buffer("inner_twist", torch.zeros(6))
        self._initialized: bool = False

        # Desired admittance set-points - corresponds to adm_ee_des_, adm_vee_des_, adm_aee_des_ in C++
        self.adm_pos_desired = torch.nn.Parameter(ee_pos_current.clone())
        self.adm_quat_desired = torch.nn.Parameter(ee_quat_current.clone())
        self.adm_twist_desired = torch.nn.Parameter(torch.zeros(6))
        self.adm_acc_desired = torch.nn.Parameter(torch.zeros(6))

        # Admittance gains - corresponds to aMd_, aKd_, aDd_ in C++
        self.register_buffer("adm_mass", adm_mass)
        self.register_buffer("adm_mass_inv", torch.inverse(adm_mass))
        self.register_buffer("adm_damping", adm_damping)
        self.register_buffer("adm_stiffness", adm_stiffness)

        # Nullspace damping for joint velocity damping in the nullspace
        if nullspace_damping is None:
            nullspace_damping = torch.zeros(7)  # Default: no nullspace damping
        self.register_buffer("nullspace_damping", nullspace_damping)

        self.dt = dt

    def _se3_error(
        self, pos_current: torch.Tensor, quat_current: torch.Tensor,
        pos_desired: torch.Tensor, quat_desired: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SE3 error from current to desired pose.
        Equivalent to adaptive::utils::SE3Error(current, desired) in C++.
        Returns 6D error vector [linear_error, angular_error] in world-aligned frame.
        """
        # Orientation error: log(quat_desired * quat_current^{-1})
        # This gives error in world-aligned frame (LOCAL_WORLD_ALIGNED in pinocchio)
        quat_current_inv = torch.ops.torchrot.invert_quaternion(quat_current)
        quat_error = torch.ops.torchrot.quaternion_multiply(quat_desired, quat_current_inv)
        ori_err = torch.ops.torchrot.quat2rotvec(quat_error)
        
        # Position error: desired - current
        pos_err = pos_desired - pos_current

        return torch.cat([pos_err, ori_err])

    def _motion_error(
        self, twist_current: torch.Tensor, twist_desired: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute motion (twist) error.
        Equivalent to adaptive::utils::motionError(current, desired) in C++.
        """
        return twist_desired - twist_current

    def _transform_wrench_to_world_aligned(
        self, wrench_local: torch.Tensor, quat_ee: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform wrench from LOCAL (end-effector) frame to LOCAL_WORLD_ALIGNED frame.
        Equivalent to pinocchio::changeReferenceFrame in C++.
        
        In LOCAL_WORLD_ALIGNED frame, the force/torque vectors are expressed in
        world-aligned coordinates but at the end-effector origin.
        """
        # Convert quaternion to rotation matrix
        rot_matrix = torch.ops.torchrot.quat2matrix(quat_ee)
        
        # Transform linear force and angular torque
        force_world = rot_matrix @ wrench_local[:3]
        torque_world = rot_matrix @ wrench_local[3:]
        
        return torch.cat([force_world, torque_world])

    def _se3_exp(self, twist: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SE3 exponential map: exp(twist * dt).
        Returns (translation_delta, quat_delta) representing the SE3 displacement.
        
        This is equivalent to pinocchio::exp6(motion * dt) in C++.
        """
        linear_vel = twist[:3]
        angular_vel = twist[3:]
        
        # Angular displacement
        rotvec = angular_vel * dt
        quat_delta = torch.ops.torchrot.rotvec2quat(rotvec)
        
        # Linear displacement
        # For small timesteps, use semi-implicit Euler: pos_delta = v * dt
        pos_delta = linear_vel * dt
        
        return pos_delta, quat_delta

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q = state_dict["joint_positions"]
        dq = state_dict["joint_velocities"]

        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(q)
        jacobian = self.robot_model.compute_jacobian(q)
        ee_twist_current = jacobian @ dq

        # One-time initialization of inner state to current EE pose
        if not self._initialized:
            self.inner_pos = ee_pos_current.clone()
            self.inner_quat = ee_quat_current.clone()
            self.inner_twist = ee_twist_current.clone()
            self._initialized = True

        # NOTE: Error computed with inner state (not current)
        se3_error = self._se3_error(
            self.inner_pos, self.inner_quat,
            self.adm_pos_desired, self.adm_quat_desired
        )
        
        # Velocity error between inner state and desired
        vel_error = self._motion_error(self.inner_twist, self.adm_twist_desired)

        # External wrench estimate from measured external torques
        # Use fast pseudo-inverse: (J^T)^+ = J (J^T J)^{-1} for 7x6 matrix J^T
        # This is faster than torch.pinverse() for real-time control (avoids SVD)
        # For overdetermined J^T (7x6), the least-squares solution is:
        # wrench = (J J^T)^{-1} J @ tau_ext
        JT = jacobian.T  # 7x6
        J = jacobian     # 6x7
        JJT = J @ JT     # 6x6
        JJT_inv = torch.linalg.solve(JJT, torch.eye(6, device=JJT.device, dtype=JJT.dtype))
        wrench_ext = JJT_inv @ (J @ state_dict["motor_torques_external"])
        wrench_ext[2] -= 0.3

        # Admittance dynamics: M * a = M * a_d + K * se3_error + D * vel_error + F_ext
        force_term = (
            self.adm_mass @ self.adm_acc_desired
            + self.adm_stiffness @ se3_error
            + self.adm_damping @ vel_error
            + wrench_ext
        )
        accel = self.adm_mass_inv @ force_term

        # Semi-implicit Euler integration on SE3 manifold
        inner_twist_next = self.inner_twist + accel * self.dt
        
        # SE3 integration: inner_SE3 = exp(inner_motion * dt) * inner_SE3
        pos_delta, quat_delta = self._se3_exp(inner_twist_next, self.dt)
        
        inner_pos_next = self.inner_pos + pos_delta
        # Quaternion composition: q_new = q_delta * q_current (right multiplication)
        inner_quat_next = torch.ops.torchrot.quaternion_multiply(quat_delta, self.inner_quat)
        inner_quat_next = torch.ops.torchrot.normalize_quaternion(inner_quat_next)

        self.inner_twist = inner_twist_next
        self.inner_pos = inner_pos_next
        self.inner_quat = inner_quat_next

        # Track inner state with outer Cartesian impedance (PD controller)
        wrench_feedback = self.pose_pd(
            ee_pos_current,
            ee_quat_current,
            ee_twist_current,
            self.inner_pos,
            self.inner_quat,
            self.inner_twist,
        )
        torque_feedback = jacobian.T @ wrench_feedback

        torque_feedforward = self.invdyn(
            q, dq, torch.zeros_like(q)
        )  # coriolis compensation

        # Nullspace damping: project joint velocity damping into nullspace of Jacobian
        # tau_null = (I - J^+ J) * (-Kd_null * dq)
        # Using dynamically consistent nullspace projector for better behavior
        JJT_inv = torch.linalg.solve(JJT, torch.eye(6, device=JJT.device, dtype=JJT.dtype))
        J_pinv = JT @ JJT_inv  # 7x6: right pseudo-inverse of J
        nullspace_projector = torch.eye(7, device=q.device, dtype=q.dtype) - J_pinv @ J
        torque_nullspace = nullspace_projector @ (-self.nullspace_damping * dq)

        torque_out = torque_feedback + torque_feedforward + torque_nullspace

        return {"joint_torques": torque_out}
