from reconstruction.traditional.transformation import mesh_to_watertight
from .morphology import udf_mesh, maximal_balls
from .neighbor import knn_neighbors


def skeletonization_modelnet(mesh, manifold_resolution=20000, num_vol_pts=1000000, skel_points=1024):
    mesh_watertight = mesh_to_watertight(mesh, manifold_resolution)
    p_vol, sdf_vol = udf_mesh(mesh_watertight, num_vol_pts)
    nearest_neighbors = knn_neighbors(p_vol)
    p_skel, sdf_skel, sdf_diff_skel, sdf_dilate, sdf_diff = maximal_balls(p_vol, sdf_vol, nearest_neighbors, skel_points)

    return p_skel, sdf_skel


def skeletonization_shapenet(mesh, manifold_resolution=20000, num_vol_pts=1000000, skel_points=1024):
    mesh_watertight = mesh_to_watertight(mesh, manifold_resolution)
    p_vol, sdf_vol = udf_mesh(mesh_watertight, num_vol_pts)
    nearest_neighbors = knn_neighbors(p_vol)
    p_skel, sdf_skel, sdf_diff_skel, sdf_dilate, sdf_diff = maximal_balls(p_vol, sdf_vol, nearest_neighbors, skel_points)

    return p_skel, sdf_skel
