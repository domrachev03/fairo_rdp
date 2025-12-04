# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

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


class CartesianImpedanceControlWithFF(CartesianImpedanceControl):
    """
    Cartesian impedance controller with an end-effector wrench feedforward term.
    The feedforward wrench is treated as controller state (updated via `update`), not as a static parameter.
    """

    def __init__(
        self,
        joint_pos_current,
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        super().__init__(
            joint_pos_current=joint_pos_current,
            Kp=Kp,
            Kd=Kd,
            robot_model=robot_model,
            ignore_gravity=ignore_gravity,
        )
        # Feedforward wrench stored as buffer (controller state)
        self.register_buffer("ee_wrench_ff", torch.zeros(6))

    @torch.jit.export
    def update(self, update_dict: Dict[str, torch.Tensor]) -> None:
        # Allow updating feedforward wrench through controller updates
        if "ee_wrench_ff" in update_dict:
            self.ee_wrench_ff.copy_(update_dict["ee_wrench_ff"])
        # Update remaining parameters using base implementation
        remaining = {k: v for k, v in update_dict.items() if k != "ee_wrench_ff"}
        if len(remaining) > 0:
            super().update(remaining)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Same as base class but adds wrench feedforward
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

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

        # TODO: do we need to include force from FT sensor here?
        torque_ff_wrench = jacobian.T @ self.ee_wrench_ff

        torque_out = torque_feedback + torque_feedforward + torque_ff_wrench

        return {"joint_torques": torque_out}


class CartesianAdmittanceControl(toco.PolicyModule):
    """
    Cartesian admittance controller.

    An admittance filter integrates external wrench into an inner desired
    pose/velocity, which is then tracked by a Cartesian impedance outer loop.
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

        # Inner (integrated) state
        self.register_buffer("inner_pos", ee_pos_current)
        self.register_buffer("inner_quat", ee_quat_current)
        self.register_buffer("inner_twist", torch.zeros(6))
        self._initialized: bool = False

        # Desired admittance set-points
        self.adm_pos_desired = torch.nn.Parameter(ee_pos_current.clone())
        self.adm_quat_desired = torch.nn.Parameter(ee_quat_current.clone())
        self.adm_twist_desired = torch.nn.Parameter(torch.zeros(6))
        self.adm_acc_desired = torch.nn.Parameter(torch.zeros(6))

        # Admittance gains
        self.register_buffer("adm_mass", adm_mass)
        self.register_buffer("adm_mass_inv", torch.inverse(adm_mass))
        self.register_buffer("adm_damping", adm_damping)
        self.register_buffer("adm_stiffness", adm_stiffness)

        self.dt = dt

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q = state_dict["joint_positions"]
        dq = state_dict["joint_velocities"]

        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(q)
        jacobian = self.robot_model.compute_jacobian(q)
        ee_twist_current = jacobian @ dq

        # One-time initialization of inner state to avoid transient bump
        if not self._initialized:
            self.inner_pos = ee_pos_current
            self.inner_quat = ee_quat_current
            self.inner_twist = ee_twist_current
            self._initialized = True

        # Pose & velocity errors between inner state and admittance set-point
        pos_error = self.adm_pos_desired - self.inner_pos
        rel_quat = torch.ops.torchrot.quaternion_multiply(
            self.adm_quat_desired, torch.ops.torchrot.invert_quaternion(self.inner_quat)
        )
        rot_error = torch.ops.torchrot.quat2rotvec(rel_quat)
        se3_error = torch.cat([pos_error, rot_error])
        vel_error = self.adm_twist_desired - self.inner_twist

        # External wrench estimate from measured external torques
        wrench_ext = torch.matmul(
            torch.pinverse(jacobian.T), state_dict["motor_torques_external"]
        )

        # Admittance dynamics: M * a + D * (v - v_d) + K * (x - x_d) = F_ext + M * a_d
        force_term = (
            self.adm_mass @ self.adm_acc_desired
            + self.adm_stiffness @ se3_error
            + self.adm_damping @ vel_error
            + wrench_ext
        )
        accel = self.adm_mass_inv @ force_term

        # Integrate inner state (semi-implicit Euler)
        inner_twist_next = self.inner_twist + accel * self.dt
        inner_pos_next = self.inner_pos + inner_twist_next[:3] * self.dt
        inner_rotvec = torch.ops.torchrot.quat2rotvec(self.inner_quat)
        inner_rotvec_next = inner_rotvec + inner_twist_next[3:] * self.dt
        inner_quat_next = torch.ops.torchrot.rotvec2quat(inner_rotvec_next)

        self.inner_twist = inner_twist_next
        self.inner_pos = inner_pos_next
        self.inner_quat = inner_quat_next

        # Track inner state with outer Cartesian impedance
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

        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}
