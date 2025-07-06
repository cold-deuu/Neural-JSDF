# ROS2 - Python
import rclpy
from rclpy.logging import get_logger
from rclpy.node import Node

# Pinocchio (FK)
import pinocchio as pin
from pinocchio.utils import *
from pinocchio import RobotWrapper

# Mesh
import trimesh

# Custom - RobotWrapper
from data_sampling.canadarm_wrapper import CanadarmWrapper
from data_sampling.mesh_manager import MeshManager

class SamplingNode(Node):
    def __init__(self):
        super().__init__("canadarm_jsdf_sampling")

        self.robot = CanadarmWrapper()
        self.mesh_manager = MeshManager()
        
        self.q = np.zeros((self.robot.robot.nq))
        self.v = np.zeros((self.robot.robot.nv))


def main(args=None):
    rclpy.init(args=args)
    node = SamplingNode()
    try:
        while rclpy.ok():
            node.robot.computeAllTerms(node.q, node.v)
            link_list = node.robot.get_all_link_position()
            node.mesh_manager.update_all_meshes(link_list)
            
    except KeyboardInterrupt:
        node.mesh_manager.visualize_all_meshes()

            
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()