# Path
import os

# Numpy
import numpy as np

# Mesh
import trimesh

# ROS2
from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory

class box:
    def __init__(self, V, scale_factor):
        self.xmin = np.min(V[:,0])
        self.xmax = np.max(V[:,0])
        self.ymin = np.min(V[:,1])
        self.ymax = np.max(V[:,1])
        self.ymin = np.min(V[:,2])
        self.ymax = np.max(V[:,2])

        self.scale_factor = scale_factor
    
    def update_box(self, V):
        self.xmin = np.min(V[:,0])
        self.xmax = np.max(V[:,0])
        self.ymin = np.min(V[:,1])
        self.ymax = np.max(V[:,1])
        self.zmin = np.min(V[:,2])
        self.zmax = np.max(V[:,2])

    def scaling(self):
        del_x = self.xmax - self.xmin
        del_y = self.ymax - self.ymin
        del_z = self.zmax - self.zmin

        self.xmin = self.xmin - self.scale_factor * del_x
        self.xmax = self.xmax + self.scale_factor * del_x
        self.ymin = self.ymin - self.scale_factor * del_y
        self.ymax = self.ymax + self.scale_factor * del_y
        self.zmin = self.zmin - self.scale_factor * del_z
        self.zmax = self.zmax + self.scale_factor * del_z

    def sampling_in_box(self, random):
        x = self.xmin + (self.xmax - self.xmin) * random[0]
        y = self.ymin + (self.ymax - self.ymin) * random[1]
        z = self.zmin + (self.zmax - self.zmin) * random[2]

        return np.array([x, y, z])        

    def check_in_box(self, point):
        x, y, z = point
        return (self.xmin <= x <= self.xmax and self.ymin<=y<=self.ymax and self.zmin<=z<=self.zmax)
    


class mesh:
    def __init__(self, link_id, link_name, F, V):
        self.link_id = link_id
        self.link_name = link_name
        self.init_V = V.copy()
        self.updated_V = self.init_V.copy()
        self.F = F

        self.box = box(self.updated_V, 0.0)
        self.closed_box = box(self.updated_V, 0.1)
        self.far_box = box(self.updated_V, 1.0)
        

    def update_mesh(self, rot_mat, trans):
        for i in range(self.init_V.shape[0]):
            self.updated_V[i,:] = (rot_mat @ self.init_V[i,:].T).T + trans

        self.box.update_box(self.updated_V)
        self.closed_box.update_box(self.updated_V)
        self.far_box.update_box(self.updated_V)
        self.closed_box.scaling()
        self.far_box.scaling()
        

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

    def visualize_all_meshes_with_sample(self, points):
        scene = trimesh.Scene()

        # 포인트를 8개로 나누기
        N = points.shape[0]
        group_size = N // 8

        colors = np.zeros((N, 4), dtype=np.uint8)  # RGBA

        color_list = [
            [255, 0, 0, 255],     # 빨강
            [0, 255, 0, 255],     # 초록
            [0, 0, 255, 255],     # 파랑
            [255, 255, 0, 255],   # 노랑
            [0, 255, 255, 255],   # 시안
            [255, 0, 255, 255],   # 마젠타
            [128, 128, 128, 255], # 회색
            [255, 128, 0, 255]    # 주황
        ]

        for i in range(8):
            start = i * group_size
            end = (i + 1) * group_size if i < 7 else N  # 마지막 그룹은 나머지까지
            colors[start:end] = color_list[i]

        cloud = trimesh.points.PointCloud(points, colors=colors)

        scene.add_geometry(cloud)
        for m in self.mesh_list:
            mesh_obj = m.return_mesh()
            scene.add_geometry(mesh_obj, node_name=f"{m.link_id}_{m.link_name}")
        
        scene.show()

    