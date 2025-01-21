import torch
import numpy as np
import open3d.ml.torch as ml3d
from scipy.sparse import coo_matrix


def knn_neighbors(p_vol, k=20, symmetric=False):
    knn_search = ml3d.layers.KNNSearch()
    knn_search_results = knn_search(p_vol, p_vol, k)
    neighbors_index = knn_search_results.neighbors_index.reshape(len(p_vol), k)

    if symmetric:
        rows = np.repeat(np.arange(len(p_vol)), k)
        cols = neighbors_index.flatten()
        data = np.ones_like(rows)
        adjacency_matrix = coo_matrix((data, (rows, cols)), shape=(len(p_vol), len(p_vol)))
        adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.transpose())
        neighbors_index = [adjacency_matrix.indices[start:end] for start, end in zip(adjacency_matrix.indptr[:-1], adjacency_matrix.indptr[1:])]
        neighbors_index = [torch.tensor(index) for index in neighbors_index]
        max_length = max(len(index) for index in neighbors_index)
        neighbors_index = [torch.nn.functional.pad(index, (0, max_length - len(index)), value=index[0]) for index in neighbors_index]
        neighbors_index = torch.stack(neighbors_index)

    return neighbors_index


def rnn_neighbors(p_vol, radius=0.01):
    radius_search = ml3d.layers.RadiusSearch()
    knn_search = ml3d.layers.KNNSearch()
    radius_search_results = radius_search(p_vol, p_vol, torch.full((len(p_vol),), radius))
    knn_search_results = knn_search(p_vol, p_vol, 1)
    closest_neighbor = knn_search_results.neighbors_index.reshape(len(p_vol), 1)
    row_splits = radius_search_results.neighbors_row_splits

    all_neighbor_indices = []
    for i in range(len(row_splits) - 1):
        start_idx = row_splits[i]
        end_idx = row_splits[i + 1]
        neighbor_indices = radius_search_results.neighbors_index[start_idx:end_idx]
        if len(neighbor_indices) == 0:
            neighbor_indices = closest_neighbor[i]
        else:
            neighbor_indices = radius_search_results.neighbors_index[start_idx:end_idx]
        all_neighbor_indices.append(neighbor_indices)

    all_neighbor_indices = [torch.tensor(index) for index in all_neighbor_indices]
    max_length = max(len(index) for index in all_neighbor_indices)
    all_neighbor_indices = [torch.nn.functional.pad(index, (0, max_length - len(index)), value=index[0]) for index in all_neighbor_indices]
    all_neighbor_indices = torch.stack(all_neighbor_indices)

    return all_neighbor_indices
