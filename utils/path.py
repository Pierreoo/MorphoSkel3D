import torch
import numpy as np
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_data_path() -> Path:
    proj_root = get_project_root()
    return proj_root / 'data'

def get_modelnet_path() -> Path:
    data_path = get_data_path()
    return data_path / 'modelnet'

def get_shapenet_path() -> Path:
    data_path = get_data_path()
    return data_path / 'shapenet'

def get_modelnetcore_path() -> Path:
    modelnet_path = get_modelnet_path()
    return modelnet_path / 'ModelNet'

def get_modelnetmesh_path() -> Path:
    modelnet_path = get_modelnet_path()
    return modelnet_path / 'ModelNetMeshFinal'

def get_modelnetskel_path() -> Path:
    modelnet_path = get_modelnet_path()
    return modelnet_path / 'ModelNetSkelFinal'

def get_shapenetcore_path() -> Path:
    shapenet_path = get_shapenet_path()
    return shapenet_path / 'ShapeNet'

def get_shapenetmesh_path() -> Path:
    shapenet_path = get_shapenet_path()
    return shapenet_path / 'ShapeNetMeshFinal'

def get_shapenetwatertight_path() -> Path:
    shapenet_path = get_shapenet_path()
    return shapenet_path / 'ShapeNetWatertightFinal'

def get_shapenetskel_path() -> Path:
    shapenet_path = get_shapenet_path()
    return shapenet_path / 'ShapeNetSkelFinal'

def get_shapenetmat_path() -> Path:
    shapenet_path = get_shapenet_path()
    return shapenet_path / 'ShapeNetMAT'

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def remap_labels(y_true):
    y_remap = torch.zeros_like(y_true)
    for i, l in enumerate(torch.unique(y_true)):
        y_remap[y_true==l] = i
    return y_remap

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud dataloaders, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def read_off(file):
    off_header = file.readline().strip()
    if 'OFF' == off_header:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces
