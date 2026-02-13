import os
import torch
import numpy as np
import time

import argparse
import open3d as o3d
from plyfile import PlyData
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.abspath('./utils')))

from utils.path import get_shapenetcore_path, get_shapenetmesh_path, get_shapenetwatertight_path, get_shapenetskel_path
from utils.filerw import load_data_id
from reconstruction.PoNQ import mesh_tools as mt
from method.MS3D.dataset import skeletonization_shapenet
from reconstruction.traditional.transformation import point_to_mesh, mesh_to_watertight


class ShapeNetMorphoSkel3D(Dataset):
    def __init__(self, data_list, data_folder, point_num, category, skel_points=1024, manifold_resolution=20000, num_vol_pts=1000000, manifold='skeleton'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.skel_points = skel_points
        self.manifold_resolution = manifold_resolution
        self.num_vol_pts = num_vol_pts
        self.data_id = data_list
        self.data_folder = data_folder
        self.point_num = point_num
        self.category = category
        self.manifold = manifold

        self.categories = set(item.split('/')[0] for item in self.data_id)
        for category in self.categories:
            mesh_path = os.path.join(root_mesh, f"{category}")
            if not os.path.exists(mesh_path):
                os.makedirs(mesh_path)
            skel_path = os.path.join(root_skel, f"{category}")
            if not os.path.exists(skel_path):
                os.makedirs(skel_path)
            watertight_path = os.path.join(root_watertight, f"{category}")
            if not os.path.exists(watertight_path):
                os.makedirs(watertight_path)

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
        watertight_file = os.path.join(root_watertight, self.data_id[index] + '.obj')
        skel_file = os.path.join(root_skel, self.data_id[index] + '.txt')
        root_file = PlyData.read(os.path.join(self.data_folder, self.data_id[index] + '.ply'))
        vertex_data = root_file['vertex']
        x, y, z = vertex_data['x'], vertex_data['y'], vertex_data['z']
        nx, ny, nz = vertex_data['nx'], vertex_data['ny'], vertex_data['nz']
        input_points = np.column_stack((x, y, z))
        input_normals = np.column_stack((nx, ny, nz))

        if manifold == 'mesh':
            if not os.path.exists(os.path.join(root_mesh, mesh_file)):
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(input_points)
                cloud.normals = o3d.utility.Vector3dVector(input_normals)
                mesh = point_to_mesh(cloud, 'poisson')
                nv = np.asarray(mesh.vertices)
                nf = np.asarray(mesh.triangles)
                mt.export_obj(nv, nf, os.path.splitext(mesh_file)[0])

        elif manifold == 'watertight':
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            if not os.path.exists(os.path.join(root_skel, watertight_file)):
                mesh_watertight = mesh_to_watertight(mesh, self.manifold_resolution)
                mt.export_obj(np.asarray(mesh_watertight.vertices), np.asarray(mesh_watertight.triangles), os.path.splitext(watertight_file)[0])

        elif manifold == 'skeleton':
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            if not os.path.exists(os.path.join(root_skel, skel_file)):
                p_skel, sdf_skel = skeletonization_shapenet(mesh,
                                                            skel_points=self.skel_points,
                                                            manifold_resolution=self.manifold_resolution,
                                                            num_vol_pts=self.num_vol_pts)
                skel_data = torch.cat((p_skel, sdf_skel.unsqueeze(1)), dim=1)
                np.savetxt(skel_file, skel_data.numpy(), fmt='%.6f')

        return input_points

    def __len__(self):
        return len(self.data_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='Airplane', help='ShapeNet category')
    parser.add_argument('--tag', type=str, default='Demo', help='Tag name to append to root_skel path')
    args = parser.parse_args()

    root = get_shapenetcore_path()
    root_mesh = get_shapenetmesh_path()
    root_watertight = get_shapenetwatertight_path()
    root_skel = get_shapenetskel_path()
    if args.tag:
        root_skel = str(root_skel) + args.tag

    category = args.category
    manifold = 'skeleton'
    data_root = os.path.join(root, 'pointclouds/')
    test_list = load_data_id(os.path.join(root, 'data-split', f'all-test.txt'))
    test_data = ShapeNetMorphoSkel3D(data_list=test_list, data_folder=data_root, category=category, point_num=2000, skel_points=1024, manifold_resolution=20000, num_vol_pts=10000000, manifold='skeleton')
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=False)

    start_time = time.time()
    for idx, data in enumerate(test_loader):
        pass
    test_duration = time.time() - start_time
    print(f'Total time for test loader: {test_duration} seconds')
