import numpy as np
import open3d as o3d
import point_cloud_utils as pcu


def point_to_mesh(point_cloud, strategy='alpha'):
    mesh = o3d.geometry.TriangleMesh()
    if strategy == 'bpa':
        distances = point_cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        radii = o3d.utility.DoubleVector([radius, radius * 2])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=point_cloud, radii=radii)
        mesh = mesh.simplify_quadric_decimation(100000)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
    elif strategy == 'poisson':
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=point_cloud, depth=8)[0]
    elif strategy == 'alpha':
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd=point_cloud, alpha=0.09)
        mesh.compute_vertex_normals()

    return mesh


def mesh_normalization(mesh):
    center = mesh.get_center()
    mesh.translate(-center)
    max_range = np.max(np.abs(np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound())))
    scale_factor = 2.0 / max_range
    mesh.scale(scale_factor, center=(0, 0, 0))

    return mesh


def mesh_to_watertight(mesh, manifold_resolution=20000):
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    vm, fm = pcu.make_mesh_watertight(v, f, manifold_resolution)
    watertight_mesh = o3d.geometry.TriangleMesh()
    watertight_mesh.vertices = o3d.utility.Vector3dVector(vm)
    watertight_mesh.triangles = o3d.utility.Vector3iVector(fm)

    return watertight_mesh
