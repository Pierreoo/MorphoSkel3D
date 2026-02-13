import os
import glob
import torch
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath('./utils')))

from utils.path import get_modelnetcore_path, get_modelnetmesh_path, get_modelnetskel_path
from utils import mesh_tools as mt
from method.MS3D.dataset import skeletonization_modelnet
if torch.cuda.is_available():
    from reconstruction.PoNQ.PoNQ import PoNQ
    from reconstruction.PoNQ.optimization import train_simple
from reconstruction.traditional.transformation import point_to_mesh



class ModelNetMorphoSkel3D:
    def __init__(self, split='train', skel_points=1024, manifold_resolution=20000, num_vol_pts=1000000, reconstruction='poisson'):
        self.root = root
        self.root_mesh = root_mesh
        self.root_skel = root_skel
        self.skel_points = skel_points
        self.manifold_resolution = manifold_resolution
        self.num_vol_pts = num_vol_pts
        self.reconstruciton = reconstruction

        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i in range(len(shape_ids[split]))]

        self.files = []
        self.class_numbers = []
        for category in self.cat:
            category_files = glob.glob(os.path.join(self.root, f"{category}/*.txt"))
            category_files = sorted([os.path.relpath(fname, self.root) for fname in category_files])
            self.files.extend(category_files)
            self.class_numbers.extend([self.classes[category]] * len(category_files))

            mesh_path = os.path.join(self.root_mesh, f"{category}")
            if not os.path.exists(mesh_path):
                os.makedirs(mesh_path)

            skel_path = os.path.join(self.root_skel, f"{category}")
            if not os.path.exists(skel_path):
                os.makedirs(skel_path)

            for file_path in category_files:
                file_base = os.path.basename(file_path)
                file_name = os.path.splitext(file_base)[0]
                mesh_file = os.path.join(mesh_path, f"{file_name}.obj")
                skel_file = os.path.join(skel_path, f"{file_name}.txt")
                if not os.path.exists(os.path.join(self.root_mesh, mesh_file)):
                    with open(os.path.join(self.root, file_path), 'r') as f:
                        loaded_data = np.loadtxt(f, delimiter=',')
                        input_points = loaded_data[:, :3]
                        input_normals = loaded_data[:, 3:6]
                        if reconstruction == 'ponq':
                            N_PONQ_POINTS = int(5e4)
                            ponq_points = input_points[np.random.choice(len(input_points), N_PONQ_POINTS)] + 1e-5
                            V = PoNQ(ponq_points, device)
                            optimizer = torch.optim.Adam([V.points], 1e-3)
                            tensor_surface = torch.tensor(input_points, dtype=torch.float32).to(device)
                            tensor_normals = torch.tensor(input_normals, dtype=torch.float32).to(device)
                            for i in tqdm(range(200)):
                                train_simple(V, optimizer, tensor_surface, repulsion_fac=0, sample_fac=.1)
                            V.cluster_samples_quadrics_normals(tensor_surface, tensor_normals)
                            nv, nf = V.min_cut_surface(128)
                        elif reconstruction == 'poisson':
                            trad_cloud = o3d.geometry.PointCloud()
                            trad_cloud.points = o3d.utility.Vector3dVector(input_points)
                            trad_cloud.normals = o3d.utility.Vector3dVector(input_normals)
                            trad_mesh = point_to_mesh(trad_cloud, 'poisson')
                            nv = np.asarray(trad_mesh.vertices)
                            nf = np.asarray(trad_mesh.triangles)
                        else:
                            print('Unknown reconstruction method: %s' % (reconstruction))
                        mt.export_obj(nv, nf, os.path.splitext(mesh_file)[0])
                else:
                    mesh = o3d.io.read_triangle_mesh(mesh_file)
                    nv = np.asarray(mesh.vertices)
                    nf = np.asarray(mesh.triangles)

                if not os.path.exists(os.path.join(self.root_skel, skel_file)):
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(nv)
                    mesh.triangles = o3d.utility.Vector3iVector(nf)
                    p_skel, sdf_skel = skeletonization_modelnet(mesh,
                                                                skel_points=self.skel_points,
                                                                manifold_resolution=self.manifold_resolution,
                                                                num_vol_pts=self.num_vol_pts)
                    skel_data = torch.cat((p_skel, sdf_skel.unsqueeze(1)), dim=1)
                    np.savetxt(skel_file, skel_data.numpy(), fmt='%.6f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test', help='ModelNet split (train or test)')
    parser.add_argument('--tag', type=str, default='Demo', help='Tag name to append to root_skel path')
    args = parser.parse_args()

    root = get_modelnetcore_path()
    root_mesh = get_modelnetmesh_path()
    root_skel = get_modelnetskel_path()
    if args.tag:
        root_skel = str(root_skel) + args.tag

    split = args.split
    ModelNetMorphoSkel3D(split=split, skel_points=1024, manifold_resolution=20000, num_vol_pts=10000000, reconstruction='poisson')