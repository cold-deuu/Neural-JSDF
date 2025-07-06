# Path
import os

# Numpy
import numpy as np

# Mesh
import trimesh

# ROS2
from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory

class mesh:
    def __init__(self, link_id, link_name, F, V):
        self.link_id = link_id
        self.link_name = link_name
        self.init_V = V.copy()
        self.updated_V = self.init_V.copy()
        self.F = F
        

    def update_mesh(self, rot_mat, trans):
        for i in range(self.init_V.shape[0]):
            self.updated_V[i,:] = (rot_mat @ self.init_V[i,:].T).T + trans

    def return_mesh(self):
        return trimesh.Trimesh(vertices=self.updated_V, faces=self.F, process=False)

class MeshManager:
    def __init__(self):
        self.logger = get_logger("mesh_manager")

        package_name = "canadarm_description"
        mesh_folder_path = os.path.join(get_package_share_directory(package_name), "mesh_txt")
        self.mesh_list = []
        for i in range(8):
            mesh_path = mesh_folder_path + f"/mesh{i}.txt"
            with open(mesh_path, 'r') as f:
                lines = f.readlines()

            # joint index
            link_idx = int(lines[0].strip())

            # joint name
            link_name = lines[1].strip()

            v_index = lines.index('V\n')
            f_index = lines.index('F\n')

            vertices = np.array([
                list(map(float, line.strip().split()))
                for line in lines[v_index + 1: f_index]
            ])

            faces = np.array([
                list(map(int, line.strip().split()))
                for line in lines[f_index + 1:]
            ])

            mesh_info = mesh(link_idx, link_name, faces, vertices)
            self.mesh_list.append(mesh_info)

    def update_all_meshes(self, all_pose_list):
        for i in range(len(all_pose_list)):
            rot_mat = all_pose_list[i].rotation.copy()
            trans = all_pose_list[i].translation.copy()
            self.mesh_list[i].update_mesh(rot_mat, trans)

    def visualize_all_meshes(self):
        scene = trimesh.Scene()
        
        for m in self.mesh_list:
            mesh_obj = m.return_mesh()
            scene.add_geometry(mesh_obj, node_name=f"{m.link_id}_{m.link_name}")
        
        scene.show()

    