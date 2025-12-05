# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List
import logging
import os

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from omegaconf import DictConfig

from polysim.envs import AbstractControlledEnv
from polymetis.utils.data_dir import get_full_path_to_urdf

log = logging.getLogger(__name__)


class BulletManipulatorEnv(AbstractControlledEnv):
    """A manipulator environment using PyBullet.

    Args:
        robot_model_cfg: A Hydra configuration file containing information needed for the
                        robot model, e.g. URDF. For an example, see
                        `polymetis/conf/robot_model/franka_panda.yaml`

        gui: Whether to initialize the PyBullet simulation in GUI mode.

        use_grav_comp: If True, adds gravity compensation torques to the input torques.
        
        simulation_urdf_path: Optional path to URDF for PyBullet simulation. If provided,
                              this URDF is used for visualization (can include gripper).
                              The robot_model_cfg.robot_description_path is still used for
                              Pinocchio (dynamics/kinematics calculations).
    """

    def __init__(
        self,
        robot_model_cfg: DictConfig,
        gui: bool,
        use_grav_comp: bool = True,
        gravity: float = 9.81,
        extract_config_from_rdf=False,
        simulation_urdf_path: str = None,
    ):
        self.robot_model_cfg = robot_model_cfg
        # robot_description_path is used for Pinocchio (dynamics model)
        self.robot_description_path = get_full_path_to_urdf(
            self.robot_model_cfg.robot_description_path
        )
        # simulation_urdf_path is used for PyBullet (can include gripper for visualization)
        if simulation_urdf_path is not None:
            self.simulation_urdf_path = get_full_path_to_urdf(simulation_urdf_path)
        else:
            self.simulation_urdf_path = self.robot_description_path

        self.gui = gui
        self.controlled_joints = self.robot_model_cfg.controlled_joints
        self.n_dofs = self.robot_model_cfg.num_dofs
        assert len(self.controlled_joints) == self.n_dofs
        self.ee_link_idx = self.robot_model_cfg.ee_link_idx
        self.ee_link_name = self.robot_model_cfg.ee_link_name
        self.rest_pose = self.robot_model_cfg.rest_pose
        self.joint_limits_low = np.array(self.robot_model_cfg.joint_limits_low)
        self.joint_limits_high = np.array(self.robot_model_cfg.joint_limits_high)
        if self.robot_model_cfg.joint_damping is None:
            self.joint_damping = None
        else:
            self.joint_damping = np.array(self.robot_model_cfg.joint_damping)
        if self.robot_model_cfg.torque_limits is None:
            self.torque_limits = np.inf * np.ones(self.n_dofs)
        else:
            self.torque_limits = np.array(self.robot_model_cfg.torque_limits)
        self.use_grav_comp = use_grav_comp

        # Initialize PyBullet simulation
        if self.gui:
            self.sim = BulletClient(connection_mode=pybullet.GUI)
        else:
            self.sim = BulletClient(connection_mode=pybullet.DIRECT)

        self.sim.setGravity(0, 0, -gravity)

        # Load robot using simulation_urdf_path (which may include gripper for visualization)
        ext = os.path.splitext(self.simulation_urdf_path)[-1][1:]
        if ext == "urdf":
            self.world_id, self.robot_id = self.load_robot_description_from_urdf(
                self.simulation_urdf_path, self.sim
            )
        elif ext == "sdf":
            self.world_id, self.robot_id = self.load_robot_description_from_sdf(
                self.simulation_urdf_path, self.sim
            )
        else:
            raise Exception(f"Unknown robot definition extension {ext}!")

        if extract_config_from_rdf:
            log.info("************ CONFIG INFO ************")
            num_joints = self.sim.getNumJoints(self.robot_id)
            for i in range(num_joints):
                # joint_info tuple structure described in PyBullet docs:
                # https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.la294ocbo43o
                joint_info = self.sim.getJointInfo(self.robot_id, i)
                log.info("Joint {}".format(joint_info[1].decode("utf-8")))  # jointName
                log.info("\tLimit low : {}".format(joint_info[8]))  # jointLowerLimit
                log.info("\tLimit High: {}".format(joint_info[9]))  # jointUpperLimit
                log.info("\tJoint Damping: {}".format(joint_info[6]))  # jointDamping
            log.info("*************************************")

        # Enable torque control
        self.sim.setJointMotorControlArray(
            self.robot_id,
            self.controlled_joints,
            pybullet.VELOCITY_CONTROL,
            forces=np.zeros(self.n_dofs),
        )

        # Get total number of joints in the robot (may include gripper/finger joints)
        self.num_all_joints = self.sim.getNumJoints(self.robot_id)
        
        # Get list of movable (non-fixed) joint indices for inverse dynamics
        # calculateInverseDynamics requires values only for movable joints
        self.movable_joint_indices = []
        for i in range(self.num_all_joints):
            joint_info = self.sim.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            if joint_type != pybullet.JOINT_FIXED:
                self.movable_joint_indices.append(i)
        self.num_movable_joints = len(self.movable_joint_indices)
        
        # Create mapping from movable joint index to position in the movable joints list
        self.movable_joint_to_idx = {j: i for i, j in enumerate(self.movable_joint_indices)}

        # Initialize variables
        self.prev_torques_commanded = np.zeros(self.n_dofs)
        self.prev_torques_applied = np.zeros(self.n_dofs)
        self.prev_torques_measured = np.zeros(self.n_dofs)
        self.prev_torques_external = np.zeros(self.n_dofs)

    @staticmethod
    def load_robot_description_from_urdf(abs_urdf_path: str, sim: BulletClient):
        """Loads a URDF file into the simulation."""
        log.info("loading urdf file: {}".format(abs_urdf_path))
        robot_id = sim.loadURDF(
            abs_urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
        )

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        world_id = pybullet.loadURDF("plane.urdf", [0.0, 0.0, 0.0])
        return world_id, robot_id

    @staticmethod
    def load_robot_description_from_sdf(abs_urdf_path: str, sim: BulletClient):
        """Loads a SDF file into the simulation."""
        log.info("loading sdf file: {}".format(abs_urdf_path))
        robot_id = sim.loadSDF(
            abs_urdf_path,
        )[0]

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        world_id = pybullet.loadURDF("plane.urdf", [0.0, 0.0, 0.0])
        return world_id, robot_id

    def reset(self, joint_pos: List[float] = None, joint_vel: List[float] = None):
        """Resets simulation to the given pose, or if not given, the default rest pose"""
        if joint_pos is None:
            joint_pos = self.rest_pose
        if joint_vel is None:
            joint_vel = [0 for _ in joint_pos]

        for i in range(self.n_dofs):
            self.sim.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=self.controlled_joints[i],
                targetValue=joint_pos[i],
                targetVelocity=joint_vel[i],
            )

    def get_num_dofs(self):
        """Return number of degrees of freedom for control"""
        return self.n_dofs

    def get_current_joint_pos_vel(self):
        """Returns (current joint position, current joint velocity) as a tuple of NumPy arrays"""
        joint_cur_states = self.sim.getJointStates(
            self.robot_id, self.controlled_joints
        )
        joint_cur_pos = [joint_cur_states[i][0] for i in range(self.n_dofs)]
        joint_cur_vel = [joint_cur_states[i][1] for i in range(self.n_dofs)]
        return np.array(joint_cur_pos), np.array(joint_cur_vel)

    def get_current_joint_pos(self):
        """Returns current joint position as a NumPy array."""
        return self.get_current_joint_pos_vel()[0]

    def get_current_joint_vel(self):
        """Returns current joint velocity as a NumPy array."""
        return self.get_current_joint_pos_vel()[1]

    def get_current_joint_torques(self):
        """Returns torques: [inputted, clipped, added with gravity compensation, and measured externally]"""
        return (
            self.prev_torques_commanded,
            self.prev_torques_applied,
            self.prev_torques_measured,
            self.prev_torques_external,
        )

    def apply_joint_torques(self, torque: np.ndarray):
        """Applies a NumPy array of torques and returns the final applied torque
        (after gravity compensation, if used)."""
        assert isinstance(torque, np.ndarray)
        self.prev_torques_commanded = torque

        # Do not mutate original array!
        applied_torque = np.clip(torque, -self.torque_limits, self.torque_limits)
        self.prev_torques_applied = applied_torque.copy()

        if self.use_grav_comp:
            joint_cur_pos = self.get_current_joint_pos()
            grav_comp_torques = self.compute_inverse_dynamics(
                joint_pos=joint_cur_pos,
                joint_vel=[0] * self.n_dofs,
                joint_acc=[0] * self.n_dofs,
            )
            applied_torque += grav_comp_torques
        self.prev_torques_measured = applied_torque.copy()

        self.sim.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.controlled_joints,
            controlMode=pybullet.TORQUE_CONTROL,
            forces=applied_torque,
        )

        self.sim.stepSimulation()

        return applied_torque

    def compute_forward_kinematics(self, joint_pos: List[float] = None):
        """Computes forward kinematics.

        Warning:
            Uses PyBullet forward kinematics by resetting to the given joint position (or, if not given,
            the rest position). Therefore, drastically modifies simulation state!

        Args:
            joint_pos: Joint positions for which to compute forward kinematics.

        Returns:
            np.ndarray: 3-dimensional end-effector position

            np.ndarray: 4-dimensional end-effector orientation as quaternion

        """
        if joint_pos != None:
            log.warning(
                "Resetting PyBullet simulation to given joint_pos to compute forward kinematics!"
            )
            self.reset(joint_pos)
        link_state = self.sim.getLinkState(
            self.robot_id,
            self.ee_link_idx,
            computeForwardKinematics=True,
        )
        ee_position = np.array(link_state[4])
        ee_orient_quaternion = np.array(link_state[5])

        return ee_position, ee_orient_quaternion

    def compute_inverse_kinematics(
        self, target_position: List[float], target_orientation: List[float] = None
    ):
        """Computes inverse kinematics.

        Uses PyBullet to compute inverse kinematics.

        Args:
            target_position: Desired end-effector position.
            target_orientation: Desired end-effector orientation as quaternion.

        Returns:
            np.ndarray: Joint position satisfying the given target end-effector position/orientation.

        """
        if isinstance(target_position, np.ndarray):
            target_position = target_position.tolist()
        if isinstance(target_orientation, np.ndarray):
            target_orientation = target_orientation.tolist()
        ik_kwargs = dict(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link_idx,
            targetPosition=target_position,
            targetOrientation=target_orientation,
            upperLimits=self.joint_limits_high.tolist(),
            lowerLimits=self.joint_limits_low.tolist(),
        )
        if self.joint_damping is not None:
            ik_kwargs["joint_damping"] = self.joint_damping.tolist()
        joint_des_pos = self.sim.calculateInverseKinematics(**ik_kwargs)
        return np.array(joint_des_pos)

    def _get_movable_joint_states(self):
        """Get positions and velocities for movable (non-fixed) joints only."""
        joint_states = self.sim.getJointStates(
            self.robot_id, self.movable_joint_indices
        )
        pos = [state[0] for state in joint_states]
        vel = [state[1] for state in joint_states]
        return pos, vel

    def compute_inverse_dynamics(
        self, joint_pos: np.ndarray, joint_vel: np.ndarray, joint_acc: np.ndarray
    ):
        """Computes inverse dynamics by returning the torques necessary to get the desired accelerations
        at the given joint position and velocity.
        
        Note: PyBullet's calculateInverseDynamics requires values for all MOVABLE (non-fixed) joints.
        This method handles that by getting the current state of movable joints and overriding
        only the controlled joint values.
        """
        # Get current state of all movable joints
        movable_pos, movable_vel = self._get_movable_joint_states()
        movable_acc = [0.0] * self.num_movable_joints
        
        # Override controlled joints with the provided values
        for i, joint_idx in enumerate(self.controlled_joints):
            if joint_idx in self.movable_joint_to_idx:
                idx = self.movable_joint_to_idx[joint_idx]
                movable_pos[idx] = joint_pos[i]
                movable_vel[idx] = joint_vel[i]
                movable_acc[idx] = joint_acc[i]
        
        # Calculate inverse dynamics for movable joints
        all_torques = self.sim.calculateInverseDynamics(
            self.robot_id, movable_pos, movable_vel, movable_acc
        )
        
        # Return only torques for controlled joints
        controlled_torques = []
        for joint_idx in self.controlled_joints:
            if joint_idx in self.movable_joint_to_idx:
                idx = self.movable_joint_to_idx[joint_idx]
                controlled_torques.append(all_torques[idx])
            else:
                controlled_torques.append(0.0)
        return np.asarray(controlled_torques)

    def set_robot_state(self, robot_state):
        req_joint_pos = robot_state.joint_positions
        req_joint_vel = robot_state.joint_velocities
        assert len(req_joint_pos) == len(req_joint_vel) == self.n_dofs
        for i in range(self.n_dofs):
            self.sim.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                targetValue=req_joint_pos[i],
                targetVelocity=req_joint_vel[i],
            )

        grav_comp_torques = self.compute_inverse_dynamics(
            joint_pos=req_joint_pos,
            joint_vel=[0] * self.n_dofs,
            joint_acc=[0] * self.n_dofs,
        )

        # bug in pybullet requires many steps after reset so that subsequent forward sim calls work correctly
        for _ in range(100):
            self.sim.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=self.controlled_joints,
                controlMode=pybullet.TORQUE_CONTROL,
                forces=grav_comp_torques,
            )
            self.sim.stepSimulation()

        joint_state = self.sim.getJointStates(
            bodyUniqueId=self.robot_id,
            jointIndices=self.controlled_joints,
        )

    def close(self):
        """Cleanly disconnect the PyBullet simulation to avoid leaving GL/X contexts open."""
        try:
            # step a few times to let any pending operations finish
            for _ in range(5):
                try:
                    self.sim.stepSimulation()
                except Exception:
                    break
            # disconnect the BulletClient which also closes any GUI windows
            try:
                self.sim.disconnect()
            except Exception:
                # fallback to pybullet.disconnect if needed
                try:
                    pybullet.disconnect()
                except Exception:
                    pass
        except Exception:
            log.exception("Exception during BulletManipulatorEnv.close()")
