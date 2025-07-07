# ROS2 - Python
import rclpy
from rclpy.logging import get_logger
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

# Pinocchio (FK)
import pinocchio as pin
from pinocchio.utils import *
from pinocchio import RobotWrapper

# Mesh
import trimesh

# Custom - RobotWrapper
from data_sampling.canadarm_wrapper import CanadarmWrapper
from data_sampling.mesh_manager import MeshManager

# Python3
import os

class SamplingNode(Node):
    def __init__(self):
        super().__init__("canadarm_jsdf_sampling")

        package_name = "data_sampling"
        self.data_path = os.path.join(get_package_share_directory(package_name), "data")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)


        self.robot = CanadarmWrapper()
        self.mesh_manager = MeshManager()
        
        self.q = np.zeros((self.robot.robot.nq))
        self.v = np.zeros((self.robot.robot.nv))

        self.n_inner_pts = 25
        self.n_outer_pts = 25
        self.n_far_pts = 25
        self.n_close_pts = 25
        self.n_zero_pts = 25
        

        self.iter1 = 0
        self.iter2 = 0
        self.iter3 = 0
        self.iter4 = 0

        self.inner_pts_array = np.empty((3,0))
        self.outer_pts_array = np.empty((3,0))
        self.closed_pts_array = np.empty((3,0))
        self.far_pts_array = np.empty((3,0))
        self.total_pts_array = np.empty((3,0))

        self.inner_dist_array = np.empty((8,0))
        self.closed_dist_array = np.empty((8,0))
        self.far_dist_array = np.empty((8,0))
        self.outer_dist_array = np.empty((8,0))
        self.total_dist_array = np.empty((8,0))


        self.total_array = np.empty((11,0))


        self.is_sampling = False



    def compute_all_dist(self, mesh_list, points):
        dist = np.zeros((len(mesh_list)))
        for i, mesh in enumerate(mesh_list):
            dist[i] = mesh.calc_signed_dist(points)
        return dist



def main(args=None):
    rclpy.init(args=args)
    node = SamplingNode()
    try:
        while rclpy.ok():
            # Sampling Points
            
            node.robot.computeAllTerms(node.q, node.v)
            link_list = node.robot.get_all_link_position()
            node.mesh_manager.update_all_meshes(link_list)

            node.get_logger().info(f"Len : {len(node.mesh_manager.mesh_list)}")
            for i, mesh in enumerate(node.mesh_manager.mesh_list):
                while True:
                    point = np.random.rand(3)
                    sample_pts = mesh.far_box.sampling_in_box(point)
                    if mesh.box.check_in_box(sample_pts):
                        # Is in Mesh
                        if mesh.check_in_mesh(sample_pts):
                            if node.iter1 < node.n_inner_pts:
                                node.inner_pts_array = np.hstack((node.inner_pts_array, sample_pts.reshape(3,1)))
                                node.inner_dist_array = np.hstack((node.inner_dist_array, node.compute_all_dist(node.mesh_manager.mesh_list, sample_pts).reshape(8,1)))
                                node.iter1 +=1
                        # Is in bounding box of mesh but out of mesh
                        else:
                            if node.iter2 < node.n_outer_pts:
                                node.outer_pts_array = np.hstack((node.outer_pts_array, sample_pts.reshape(3,1)))
                                node.outer_dist_array = np.hstack((node.outer_dist_array, node.compute_all_dist(node.mesh_manager.mesh_list, sample_pts).reshape(8,1)))

                                node.iter2 +=1          

                    elif mesh.closed_box.check_in_box(sample_pts):
                        if node.iter3 < node.n_close_pts:
                            node.closed_pts_array = np.hstack((node.closed_pts_array, sample_pts.reshape(3,1)))
                            node.closed_dist_array = np.hstack((node.closed_dist_array, node.compute_all_dist(node.mesh_manager.mesh_list, sample_pts).reshape(8,1)))
                            node.iter3 +=1

                    else:
                        if node.iter4 < node.n_far_pts:
                            node.far_pts_array = np.hstack((node.far_pts_array, sample_pts.reshape(3,1)))
                            node.far_dist_array = np.hstack((node.far_dist_array, node.compute_all_dist(node.mesh_manager.mesh_list, sample_pts).reshape(8,1)))

                            node.iter4 +=1

                    if node.iter1 >= node.n_inner_pts and node.iter2 >= node.n_outer_pts and node.iter3 >= node.n_close_pts and node.iter4 >= node.n_far_pts:
                        # Total Points
                        node.total_pts_array = np.hstack((node.total_pts_array, node.inner_pts_array))
                        node.total_pts_array = np.hstack((node.total_pts_array, node.outer_pts_array))
                        node.total_pts_array = np.hstack((node.total_pts_array, node.closed_pts_array))
                        node.total_pts_array = np.hstack((node.total_pts_array, node.far_pts_array))

                        node.total_dist_array = np.hstack((node.total_dist_array, node.inner_dist_array))
                        node.total_dist_array = np.hstack((node.total_dist_array, node.outer_dist_array))
                        node.total_dist_array = np.hstack((node.total_dist_array, node.closed_dist_array))
                        node.total_dist_array = np.hstack((node.total_dist_array, node.far_dist_array))

                        # Initialize
                        node.iter1 = node.iter2 = node.iter3 = node.iter4 = 0
                        node.inner_pts_array = np.empty((3,0))
                        node.outer_pts_array = np.empty((3,0))
                        node.closed_pts_array = np.empty((3,0))
                        node.far_pts_array = np.empty((3,0))
                        
                        node.inner_dist_array = np.empty((8,0))
                        node.closed_dist_array = np.empty((8,0))
                        node.far_dist_array = np.empty((8,0))
                        node.outer_dist_array = np.empty((8,0))


                        if i == len(node.mesh_manager.mesh_list)-1:
                            node.is_sampling = True
                        break

            if node.is_sampling:
                node.total_array = np.vstack((node.total_pts_array, node.total_dist_array))
                np.savetxt(node.data_path + "/points.txt", node.total_array.T, fmt="%.6f")
                break

    except KeyboardInterrupt:
        node.get_logger().info("Node Cancled by User")

    finally:
        if node.is_sampling:
            points = np.loadtxt(node.data_path + "/points.txt")
            node.mesh_manager.visualize_all_meshes_with_sample(points[:, :3])
        else:
            node.mesh_manager.visualize_all_meshes()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()