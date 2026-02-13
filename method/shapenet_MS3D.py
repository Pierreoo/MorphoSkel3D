import os
import glob
import torch
import numpy as np
import json
import open3d as o3d

import sys
sys.path.append(os.path.dirname(os.path.abspath('./utils')))

from utils.path import get_shapenetcore_path, get_shapenetmesh_path, get_shapenetskel_path
from utils import mesh_tools as mt
from method.MS3D.dataset import skeletonization_shapenet
from reconstruction.traditional.transformation import point_to_mesh


class ShapeNetMorphoSkel3D:
    def __init__(self, split='train', category=None, skel_points=1024, manifold_resolution=20000, num_vol_pts=1000000):
        self.root = root
        self.root_mesh = root_mesh
        self.root_skel = root_skel
        self.category = category
        self.skel_points = skel_points
        self.manifold_resolution = manifold_resolution
        self.num_vol_pts = num_vol_pts
        self.cache = {}
        self.cache_size = 20000

        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not category is None:
            self.cat = {k: v for k, v in self.cat.items() if k in category}

        self.core_meta = {}
        self.mesh_meta = {}
        self.skel_meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.core_meta[item] = []
            self.mesh_meta[item] = []
            self.skel_meta[item] = []
            dir_core_point = os.path.join(self.root, self.cat[item])
            dir_mesh_point = os.path.join(self.root_mesh, self.cat[item])
            dir_skel_point = os.path.join(self.root_skel, self.cat[item])
            if not os.path.exists(dir_skel_point):
                os.makedirs(dir_skel_point)
            fns = sorted(os.listdir(dir_core_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.core_meta[item].append(os.path.join(dir_mesh_point, token + '.txt'))
                self.mesh_meta[item].append(os.path.join(dir_mesh_point, token + '.obj'))
                self.skel_meta[item].append(os.path.join(dir_skel_point, token + '.txt'))

        self.core_datapath = []
        for item in self.cat:
            for fn in self.core_meta[item]:
                self.core_datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.files = []
        self.class_numbers = []
        for category, category_number in self.cat.items():
            category_files = glob.glob(os.path.join(self.root, f"{category_number}/*.txt"))
            category_files = sorted([os.path.relpath(fname, self.root) for fname in category_files])
            self.files.extend(category_files)
            self.class_numbers.extend([self.classes[category]] * len(category_files))

            mesh_path = os.path.join(self.root_mesh, f"{category_number}")
            if not os.path.exists(mesh_path):
                os.makedirs(mesh_path)

            skel_path = os.path.join(self.root_skel, f"{category_number}")
            if not os.path.exists(skel_path):
                os.makedirs(skel_path)

            for file_path in category_files:
                file_base = os.path.basename(file_path)
                file_name = os.path.splitext(file_base)[0]
                mesh_file = os.path.join(mesh_path, f"{file_name}.obj")
                skel_file = os.path.join(skel_path, f"{file_name}.txt")
                if not os.path.exists(os.path.join(self.root_mesh, mesh_file)):
                    with open(os.path.join(self.root, file_path), 'r') as f:
                        loaded_data = np.loadtxt(f, delimiter=' ')
                        input_points = loaded_data[:, :3]
                        input_normals = loaded_data[:, 3:6]
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

                if not os.path.exists(os.path.join(self.root_skel, skel_file)):
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(nv)
                    mesh.triangles = o3d.utility.Vector3iVector(nf)
                    p_skel, sdf_skel = skeletonization_shapenet(mesh,
                                                                skel_points=self.skel_points,
                                                                manifold_resolution=self.manifold_resolution,
                                                                num_vol_pts=self.num_vol_pts)
                    skel_data = torch.cat((p_skel, sdf_skel.unsqueeze(1)), dim=1)
                    np.savetxt(skel_file, skel_data.numpy(), fmt='%.6f')


if __name__ == "__main__":
    root = get_shapenetcore_path()
    root_mesh = get_shapenetmesh_path()
    root_skel = get_shapenetskel_path()

    ShapeNetMorphoSkel3D(split='train', skel_points=1024, manifold_resolution=20000, num_vol_pts=10000000)
    ShapeNetMorphoSkel3D(split='test', skel_points=1024, manifold_resolution=20000, num_vol_pts=10000000)