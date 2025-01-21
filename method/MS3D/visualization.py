import os
import numpy as np

import torch
import open3d as o3d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from plyfile import PlyData

from neighbor import knn_neighbors
from reconstruction.traditional.transformation import mesh_to_watertight
from morphology import udf_mesh, maximal_balls
from reconstruction.traditional.transformation import point_to_mesh


def skeletonization(dataset, manifold_resolution=20000, num_vol_pts=100000, skel_points=1024):
    shapenet_category_path = "../../data/shapenet/ShapeNet/pointclouds/02691156"
    modelnet_category_path = "../../data/modelnet/ModelNetMesh/airplane"
    modelnet_surface_category_path = "../../data/modelnet/ModelNet/airplane"
    if dataset == 'shapenet':
        for model_path in os.listdir(shapenet_category_path):
            if model_path == '1c93b0eb9c313f5d9a6e43b878d5b335.ply':
                surface_data = PlyData.read(f'{shapenet_category_path}/{model_path}')
                vertex_data = surface_data['vertex']
                x, y, z = vertex_data['x'], vertex_data['y'], vertex_data['z']
                nx, ny, nz = vertex_data['nx'], vertex_data['ny'], vertex_data['nz']
                input_points = np.column_stack((x, y, z))
                input_normals = np.column_stack((nx, ny, nz))
                surface_cloud = o3d.geometry.PointCloud()
                surface_cloud.points = o3d.utility.Vector3dVector(input_points)
                surface_cloud.normals = o3d.utility.Vector3dVector(input_normals)
                trad_cloud = o3d.geometry.PointCloud()
                trad_cloud.points = o3d.utility.Vector3dVector(input_points)
                trad_cloud.normals = o3d.utility.Vector3dVector(input_normals)
                mesh = point_to_mesh(trad_cloud, 'poisson')

    elif dataset == 'modelnet':
        for model_path in os.listdir(modelnet_category_path):
            model_name, extension = os.path.splitext(model_path)
            if model_name == 'airplane_0088':
                mesh = o3d.io.read_triangle_mesh(f'{modelnet_category_path}/{model_name}.obj')
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                with open(f'{modelnet_surface_category_path}/{model_name}.txt', 'r') as f:
                    loaded_data = np.loadtxt(f, delimiter=',')
                    p_surface = torch.Tensor(loaded_data[:, :3])

    elif dataset == 'dragon':
        mesh = o3d.io.read_triangle_mesh(f'../../data/{dataset}.obj')
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=10000)
        vertex_normals = np.asarray(mesh.vertex_normals)
        kdtree = o3d.geometry.KDTreeFlann(mesh)
        sampled_points = np.asarray(pcd.points)
        sampled_normals = np.zeros((sampled_points.shape[0], 3))
        for i, point in enumerate(sampled_points):
            [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)
            sampled_normals[i] = vertex_normals[idx[0]]
        pcd.normals = o3d.utility.Vector3dVector(sampled_normals)
        pcd.paint_uniform_color([0.2, 0.2, 0.2])
        o3d.visualization.draw_geometries([pcd], point_show_normal=False)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        mesh.paint_uniform_color([0.2, 0.2, 0.2])

    elif dataset == 'rationale':
        mesh = o3d.geometry.TriangleMesh.create_box()
        # mesh = o3d.geometry.TriangleMesh.create_torus()

    mesh_watertight = mesh_to_watertight(mesh, manifold_resolution)
    mesh_watertight.paint_uniform_color([0.2, 0.2, 0.2])
    p_vol, sdf_vol = udf_mesh(mesh_watertight, num_vol_pts)
    nearest_neighbors = knn_neighbors(p_vol, k=20)
    p_skel, sdf_skel, sdf_diff_skel, sdf_dilate, sdf_diff = maximal_balls(p_vol, sdf_vol, nearest_neighbors, skel_points)

    distance_field_visualization(p_vol, sdf_vol)
    distance_field_visualization(p_vol, sdf_dilate)
    distance_field_visualization(p_vol, sdf_diff)
    distance_field_visualization(p_skel, sdf_skel)
    skel_density_visualization(p_skel, sdf_skel, sdf_diff_skel)
    skel_surface_visualization(p_skel, p_surface)


def distance_field_visualization(points, udf):
    inner_cloud = o3d.geometry.PointCloud()
    inner_cloud.points = o3d.utility.Vector3dVector(points)
    udf = udf.numpy()
    sdf_min, sdf_max = np.min(udf), np.max(udf)
    colors = plt.colormaps.get_cmap('viridis')((udf - sdf_min) / (sdf_max - sdf_min))[:, :3]
    inner_cloud.colors = o3d.utility.Vector3dVector(colors)
    min_bound = np.min(np.asarray(inner_cloud.points), axis=0)
    max_bound = np.max(np.asarray(inner_cloud.points), axis=0)
    mid_z = (min_bound[2] + max_bound[2]) / 2.0
    max_bound[2] = mid_z
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_cloud = inner_cloud.crop(bounding_box)

    fig = plt.figure(figsize=(6, 1))
    ax = fig.add_axes([0.05, 0.5, 0.9, 0.15])
    cmap = cm.viridis
    norm = plt.Normalize(vmin=sdf_min, vmax=sdf_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.ax.tick_params(axis='x', rotation=45)
    plt.show()
    o3d.visualization.draw_geometries([cropped_cloud])


def skel_density_visualization(p_skel, sdf_skel, sdf_diff_skel):
    skel_cloud = o3d.geometry.PointCloud()
    skel_cloud.points = o3d.utility.Vector3dVector(p_skel)

    udf = sdf_skel.numpy()
    sdf_min, sdf_max = np.min(udf), np.max(udf)
    colors = plt.colormaps.get_cmap('viridis')((udf - sdf_min) / (sdf_max - sdf_min))[:, :3]
    skel_cloud.colors = o3d.utility.Vector3dVector(colors)

    fig = plt.figure(figsize=(6, 1))
    ax = fig.add_axes([0.05, 0.5, 0.9, 0.15])
    cmap = cm.viridis
    norm = plt.Normalize(vmin=sdf_min, vmax=sdf_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.ax.tick_params(axis='x', rotation=45)
    plt.show()
    o3d.visualization.draw_geometries([skel_cloud])
    plt.hist(sdf_diff_skel, bins=20)
    plt.xlabel('MS3D(x)')
    plt.ylabel('points')
    plt.show()


def skel_surface_visualization(p_skel, p_surface):
    skel_cloud = o3d.geometry.PointCloud()
    skel_cloud.points = o3d.utility.Vector3dVector(p_skel)

    surface_cloud = o3d.geometry.PointCloud()
    surface_cloud.points = o3d.utility.Vector3dVector(p_surface)

    surface_cloud.paint_uniform_color([1, 0.9, 0.7])
    skel_cloud.paint_uniform_color([0.2, 0.2, 0.4])

    o3d.visualization.draw_geometries([skel_cloud, surface_cloud])


if __name__ == "__main__":
    p_skel = skeletonization(dataset='dragon', manifold_resolution=20000, num_vol_pts=10000000, skel_points=1024)
