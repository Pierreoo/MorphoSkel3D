import torch
import numpy as np
import open3d as o3d
import open3d.ml.torch as ml3d


def minimal_distance(point_cloud1, point_cloud2):
    knn_search = ml3d.layers.KNNSearch(return_distances=True)
    knn_search_results = knn_search(point_cloud2, point_cloud1, 1)
    min_distances = knn_search_results.neighbors_distance

    return min_distances


def udf_mesh(mesh, num_vol_pts, grid='meshgrid'):
    legacy_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(legacy_mesh)
    min_bound = legacy_mesh.vertex.positions.min(0).numpy()
    max_bound = legacy_mesh.vertex.positions.max(0).numpy()
    if grid == 'random':
        p_random = np.random.uniform(low=min_bound, high=max_bound, size=[num_vol_pts, 3]).astype(np.float32)
    elif grid == 'meshgrid':
        num_points_per_dim = int(np.cbrt(num_vol_pts))
        x = np.linspace(min_bound[0], max_bound[0], num_points_per_dim)
        y = np.linspace(min_bound[1], max_bound[1], num_points_per_dim)
        z = np.linspace(min_bound[2], max_bound[2], num_points_per_dim)
        X, Y, Z = np.meshgrid(x, y, z)
        p_random = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T.astype(np.float32)
    else:
        raise ValueError('Invalid grid type')
    occupancy = scene.compute_occupancy(p_random)
    p_inside = p_random[occupancy.numpy() == 1]
    sdf_inside = scene.compute_distance(p_inside).numpy()
    p_inside = torch.tensor(p_inside, dtype=torch.float32)
    usdf_inside = torch.tensor(sdf_inside, dtype=torch.float32)

    return p_inside, usdf_inside


def udf_cloud(surface_cloud, num_vol_pts):
    min_bound = surface_cloud.get_min_bound()
    max_bound = surface_cloud.get_max_bound()
    p_random = np.random.uniform(low=min_bound, high=max_bound, size=[num_vol_pts, 3]).astype(np.float32)
    udf = minimal_distance(torch.Tensor(p_random), torch.Tensor(surface_cloud.points))
    p_inside = torch.tensor(p_random, dtype=torch.float32)
    usdf_inside = torch.tensor(udf, dtype=torch.float32)

    return p_inside, usdf_inside


def erosion(sdf, nearest_neighbors):
    nearest_sdf_values = sdf[nearest_neighbors]
    sdf_eroded, nearest_min_indices = torch.min(nearest_sdf_values, dim=1)

    return sdf_eroded


def dilation(sdf, nearest_neighbors):
    nearest_sdf_values = sdf[nearest_neighbors]
    sdf_dilated, nearest_max_indices = torch.max(nearest_sdf_values, dim=1)

    return sdf_dilated


def opening(sdf, nearest_neighbors):
    sdf_eroded = erosion(sdf, nearest_neighbors)
    sdf_opened = dilation(sdf_eroded, nearest_neighbors)

    return sdf_opened


def maximal_balls(p_vol, sdf_vol, nearest_neighbors, skel_points):
    sdf_dilate = dilation(sdf_vol, nearest_neighbors)
    sdf_diff = sdf_dilate - sdf_vol
    if skel_points > len(sdf_diff):
        remainder = skel_points - len(sdf_diff)
        skeleton = torch.topk(sdf_diff, len(sdf_diff), largest=False).indices
        repetitions_needed = (remainder // len(skeleton)) + 2
        skeleton = torch.cat([skeleton] * repetitions_needed, dim=0)
        skeleton = skeleton[:skel_points]
    else:
        skeleton = torch.topk(sdf_diff, skel_points, largest=False).indices
    p_skel, sdf_skel, sdf_diff_skel = p_vol[skeleton], sdf_vol[skeleton], sdf_diff[skeleton]
    return p_skel, sdf_skel, sdf_diff_skel, sdf_dilate, sdf_diff
