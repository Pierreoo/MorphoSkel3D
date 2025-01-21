import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points


def train_simple(V, optimizer, tensor_surface, repulsion_fac=0, sample_fac=1):
    optimizer.zero_grad()
    masks = torch.rand_like(tensor_surface[:, 0]) < sample_fac
    loss = chamfer_distance(
        tensor_surface[masks][None, :], V.points[None, :])[0].mean()
    if repulsion_fac > 0:
        min_dist = knn_points(V.points[None, :], V.points[None, :], K=2).dists[0, :, 1] ** 2
        loss += -repulsion_fac * min_dist.mean()
    x = loss.item()
    loss.backward()
    optimizer.step()
    return x