import os
import torch
import numpy as np
import argparse

import open3d as o3d
from plyfile import PlyData
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.abspath('./utils')))

from utils.path import get_shapenetcore_path, get_shapenetmesh_path, get_shapenetskel_path, get_shapenetmat_path
from utils.distfunc import point2sphere_distance_with_batch, sphere2point_distance_with_batch
from utils.filerw import load_data_id
from reconstruction.PoNQ import mesh_tools as mt
from method.MS3D.dataset import skeletonization_shapenet
from reconstruction.traditional.transformation import point_to_mesh



class ShapeNetMorphoSkel3D(Dataset):
    def __init__(self, data_list, data_folder, point_num, category, distance, evaluation, skel_points=1024, manifold_resolution=20000, num_vol_pts=1000000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.skel_points = skel_points
        self.manifold_resolution = manifold_resolution
        self.num_vol_pts = num_vol_pts
        self.data_id = data_list
        self.data_folder = data_folder
        self.point_num = point_num
        self.category = category
        self.distance = distance
        self.evaluation = evaluation

        self.categories = set(item.split('/')[0] for item in self.data_id)
        for category in self.categories:
            mesh_path = os.path.join(root_mesh, f"{category}")
            if not os.path.exists(mesh_path):
                os.makedirs(mesh_path)
            skel_path = os.path.join(root_skel, f"{category}")
            if not os.path.exists(skel_path):
                os.makedirs(skel_path)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.catfile = os.path.join(root, 'synsetoffset2category.txt')
        self.cat = {}
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        if not self.category is None:
            self.cat = {k: v for k, v in self.cat.items() if k in self.category}

        self.data_id = [item for item in self.data_id if item.split('/')[0] in self.cat.values()]

    def __getitem__(self, index):
        mesh_file = os.path.join(root_mesh, self.data_id[index] + '.obj')
        skel_file = os.path.join(root_skel, self.data_id[index] + '.txt')
        root_file = PlyData.read(os.path.join(self.data_folder, self.data_id[index] + '.ply'))
        if evaluation == 'mat':
            try:
                mat_file = PlyData.read(os.path.join(root_mat, self.data_id[index] + '_mat.ply'))
                vertex_data_mat = mat_file['vertex']
                x_mat, y_mat, z_mat = vertex_data_mat['x'], vertex_data_mat['y'], vertex_data_mat['z']
                if len(x_mat) < 2000:
                    print(self.data_id[index])
                input_points_mat = np.column_stack((x_mat, y_mat, z_mat))
            except FileNotFoundError:
                return -1
        vertex_data = root_file['vertex']
        x, y, z = vertex_data['x'], vertex_data['y'], vertex_data['z']
        nx, ny, nz = vertex_data['nx'], vertex_data['ny'], vertex_data['nz']
        input_points = np.column_stack((x, y, z))
        input_normals = np.column_stack((nx, ny, nz))

        if not os.path.exists(os.path.join(root_mesh, mesh_file)):
            trad_cloud = o3d.geometry.PointCloud()
            trad_cloud.points = o3d.utility.Vector3dVector(input_points)
            trad_cloud.normals = o3d.utility.Vector3dVector(input_normals)
            trad_mesh = point_to_mesh(trad_cloud, 'poisson')
            nv = np.asarray(trad_mesh.vertices)
            nf = np.asarray(trad_mesh.triangles)
            mt.export_obj(nv, nf, os.path.splitext(mesh_file)[0])
        else:
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            nv = np.asarray(mesh.vertices)
            nf = np.asarray(mesh.triangles)

        if not os.path.exists(os.path.join(root_skel, skel_file)):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(nv)
            mesh.triangles = o3d.utility.Vector3iVector(nf)
            p_skel, sdf_skel = skeletonization_shapenet(mesh,
                                                        skel_points=self.skel_points,
                                                        manifold_resolution=self.manifold_resolution,
                                                        num_vol_pts=self.num_vol_pts)
            skel_data = torch.cat((p_skel, sdf_skel.unsqueeze(1)), dim=1)
            np.savetxt(skel_file, skel_data.numpy(), fmt='%.6f')
        else:
            skel_data = np.loadtxt(skel_file)
            skel_data = skel_data[:self.skel_points, :4]
            # skel_data = farthest_point_sample(skel_data[:, :4], self.skel_points)
            if self.evaluation == 'mat':
                skel_data[:self.skel_points, 3] = np.zeros_like(skel_data[:self.skel_points, 3])

        if self.evaluation == 'recon':
            surface_points = torch.Tensor(input_points)
        elif self.evaluation == 'mat':
            surface_points = torch.Tensor(input_points_mat)
        else:
            raise ValueError(f'Invalid evaluation type: {self.evaluation}')

        d1 = point2sphere_distance_with_batch(surface_points, torch.Tensor(skel_data), self.distance)
        d2 = sphere2point_distance_with_batch(torch.Tensor(skel_data), surface_points, self.distance)
        if self.distance == 'chamfer':
            recon_err = d1 + d2
        elif self.distance == 'hausdorff':
            recon_err = max(d1, d2)
        else:
            raise ValueError(f'Invalid distance type: {self.distance}')

        return recon_err

    def __len__(self):
        return len(self.data_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=str, default='recon')
    parser.add_argument('--category', type=str, default='Airplane')
    parser.add_argument('--distance', type=str, default='chamfer')
    args = parser.parse_args()

    evaluation = args.evaluation
    category = args.category
    distance = args.distance
    skel_points = 1024

    root = get_shapenetcore_path()
    root_mesh = get_shapenetmesh_path()
    root_skel = get_shapenetskel_path()
    root_mat = get_shapenetmat_path()

    data_root = os.path.join(root, 'pointclouds/')
    train_list = load_data_id(os.path.join(root, 'data-split', f'all-train.txt'))
    test_list = load_data_id(os.path.join(root, 'data-split', f'all-test.txt'))
    # train_data = ShapeNetMorphoSkel3D(data_list=train_list, data_folder=data_root, category=category, distance=distance, evaluation=evaluation, point_num=2000, skel_points=skel_points, manifold_resolution=20000, num_vol_pts=10000000)
    test_data = ShapeNetMorphoSkel3D(data_list=test_list, data_folder=data_root, category=category, distance=distance, evaluation=evaluation,
                                     point_num=2000, skel_points=skel_points, manifold_resolution=20000, num_vol_pts=10000000)
    # train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, drop_last=False)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=False)
    index, all_recon_err = 1, 0
    for idx, data in enumerate(test_loader):
        if data == -1:
            continue
        index += 1
        loss_recon = data
        all_recon_err += loss_recon
        print(all_recon_err / index)