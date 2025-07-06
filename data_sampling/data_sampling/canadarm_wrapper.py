import os
import numpy as np
from numpy.linalg import pinv

from rclpy.logging import get_logger

import pinocchio as pin
from pinocchio import RobotWrapper
from pinocchio.utils import *

from ament_index_python.packages import get_package_share_directory


class state():
    def __init__(self):
        self.q: np.array
        self.v: np.array
        self.a: np.array

        self.id: np.array
        self.G: np.array
        self.M: np.array
        self.J: np.array

        self.oMi : pin.SE3


class CanadarmWrapper(RobotWrapper):
    def __init__(self):
        package_name = "canadarm_description"
        urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "canadarm", "urdf", "Canadarm2_w_iss.urdf")

        # Build Robot Model 
        self.robot = self.BuildFromURDF(urdf_file_path)
        self.data, _, _ = \
            pin.createDatas(self.robot.model, self.robot.collision_model, self.robot.visual_model)
        self.model = self.robot.model

        self.state = state()

        self.ee_joint_name = "Wrist_Roll"
        self.state.id = self.index(self.ee_joint_name)
        
        # self.joint7 = self.model.getJointName(7)
        # joint_names = [joint.name for joint in self.model.joints]    

        self.joint_list = []
        for i in range(1, self.robot.nq+1):
            self.joint_list.append(self.model.names[i])



        self.state.q = zero(self.robot.nq)
        self.state.v = zero(self.robot.nv)
        self.state.a = zero(self.robot.nv)


        self.state.oMi = pin.SE3()
        self.eef_to_tip = pin.XYZQUATToSE3(np.array([0,0,-1.4, 1, 0, 0, 0]))


    def computeAllTerms(self, q, v):
        pin.computeAllTerms(self.model, self.data, q, v)
        self.computeJointJacobians(q)
        self.state.G = self.nle(q, v)     # NonLinearEffects
        self.state.M = self.mass(q)                  # Mass
        self.state.oMi = self.data.oMi[self.state.id]
        self.state.J = self.getJointJacobian(self.state.id)

    def get_all_link_position(self):
        link_pose_list = [pin.SE3(1)]
        for i in range(self.robot.nq):
            link_pose_list.append(self.data.oMi[self.model.getJointId(self.joint_list[i])])

        return link_pose_list

