import numpy as np

import pandas as pd
from scipy import sparse as sp

import accmp.base_features._core as core
import accmp.base_features._core_f32 as core32
import accmp.base_features._core_f64 as core64


def edges_weights_to_df(edge_index: np.ndarray, weights: np.ndarray = None):
    edge_data = pd.DataFrame(
        {"src": edge_index[0], "dst": edge_index[1]}
    )

    weight_cols = []
    if weights is not None:
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        weight_cols = [f"w{i}" for i in range(weights.shape[1])]
        weights_data = pd.DataFrame(weights, columns=weight_cols)
        edge_data = pd.concat((edge_data, weights_data), axis=1, ignore_index=False)
    return edge_data, weight_cols


def rm_self_edges(edge_data: pd.DataFrame):
    edge_data = edge_data.loc[edge_data["src"] != edge_data["dst"], :].copy()
    return edge_data


def sum_repeated_edges(edge_data: pd.DataFrame):
    edge_data = edge_data.groupby(["src", "dst"], as_index=False).agg("sum")
    return edge_data


def make_symmetric(edge_data: pd.DataFrame):
    flipped_edge_data = edge_data.copy()
    flipped_edge_data["src"] = edge_data["dst"].to_numpy()
    flipped_edge_data["dst"] = edge_data["src"].to_numpy()
    edge_data = pd.concat((edge_data, flipped_edge_data), axis=0, ignore_index=True)
    edge_data = edge_data.groupby(["src", "dst"], as_index=False).agg("mean")
    return edge_data


def preprocess_edge_data(edge_index: np.ndarray, as_symmetric: bool = False, weights: np.ndarray = None,
                         dtype: type = np.float64):
    edge_index = edge_index.astype(np.int64, copy=False)
    edge_data, weight_cols = edges_weights_to_df(edge_index=edge_index, weights=weights)
    edge_data = rm_self_edges(edge_data)
    edge_data = sum_repeated_edges(edge_data)
    if as_symmetric:
        edge_data = make_symmetric(edge_data)
    edge_index = np.stack((edge_data["src"].to_numpy(), edge_data["dst"].to_numpy()), axis=0)
    # edge_index = edge_data.loc[:, ["src", "dst"]].to_numpy().T
    if weights is not None:
        weights = edge_data.loc[:, weight_cols].to_numpy().astype(dtype, copy=False)
    return edge_index, weights


def degree_features(edge_index: np.ndarray, num_nodes: int, as_undirected: bool = False, weights: np.ndarray = None,
                    dtype: type = np.float64):
    feature_names = []
    edge_index, weights = preprocess_edge_data(
        edge_index=edge_index,
        weights=weights,
        as_symmetric=as_undirected,
        dtype=dtype
    )
    out_degrees, in_degrees, out_weights_sum, in_weights_sum = _compute_degree_features(
        edge_index=edge_index,
        num_nodes=num_nodes,
        weights=weights,
        dtype=dtype
    )

    if as_undirected:
        deg_features = out_degrees[..., np.newaxis].astype(dtype, copy=False)
        feature_names.append('deg')
    else:
        deg_features = np.stack((out_degrees, in_degrees), axis=1).astype(dtype, copy=False)
        feature_names.extend(['out_deg', 'in_deg'])

    if weights is not None:
        if as_undirected:
            w_deg_features = out_weights_sum.astype(dtype, copy=False)
            w_feat_names = [f'w{i}_deg' for i in range(weights.shape[1])]
        else:
            w_deg_features = np.concatenate((out_weights_sum, in_weights_sum), axis=1).astype(dtype, copy=False)
            w_feat_names = [f'w{i}_out_deg' for i in range(weights.shape[1])] + [f'w{i}_in_deg' for i in
                                                                                 range(weights.shape[1])]

        deg_features = np.concatenate((deg_features, w_deg_features), axis=1)
        feature_names.extend(w_feat_names)
    return deg_features, feature_names


def _compute_degree_features(edge_index: np.ndarray, num_nodes: int, weights: np.ndarray = None,
                             dtype: type = np.float64):
    """ Extract node features for each node in the graph. Node features are in and out degrees, and their weighted
    counterparts if `weights` are provided.

    Args:
        edge_index: Graph edges in shape [2, num_edges]
        num_nodes: Number of nodes in the graph
        weights: Edge weights with shape [num_edges, num_weight_types]
    Returns:
        out_degrees: Out degrees for each node [num_nodes]
        in_degrees: In degrees for each node [num_nodes]
        out_weights_sum: Sum of out edge weights for each node and weight type [num_nodes, num_weight_types]
        in_weights_sum: Sum of in edge weights for each node and weight type [num_nodes, num_weight_types]
    """

    edge_data = pd.DataFrame(
        {"src": edge_index[0], "dst": edge_index[1]}
    )

    weight_cols = []
    if weights is not None:
        weight_cols = [f"w{i}" for i in range(weights.shape[1])]
        weights_data = pd.DataFrame(np.atleast_2d(weights), columns=weight_cols)
        edge_data = pd.concat((edge_data, weights_data), axis=1, ignore_index=False)

    out_agg = {"dst": "count"}
    out_agg.update({w_name: "sum" for w_name in weight_cols})
    in_agg = {"src": "count"}
    in_agg.update({w_name: "sum" for w_name in weight_cols})

    out_features = edge_data.groupby("src").agg(out_agg)
    in_features = edge_data.groupby("dst").agg(in_agg)

    out_degrees = np.zeros(num_nodes, dtype=dtype)
    out_degrees[out_features.index] = out_features['dst'].to_numpy()

    in_degrees = np.zeros(num_nodes, dtype=dtype)
    in_degrees[in_features.index] = in_features['src'].to_numpy()

    if weights is not None:
        out_weights_sum = np.zeros((num_nodes, len(weight_cols)), dtype=dtype)
        out_weights_sum[out_features.index] = out_features.loc[:, weight_cols].to_numpy()
        in_weights_sum = np.zeros((num_nodes, len(weight_cols)), dtype=dtype)
        in_weights_sum[in_features.index] = in_features.loc[:, weight_cols].to_numpy()
    else:
        out_weights_sum = in_weights_sum = None
    return out_degrees, in_degrees, out_weights_sum, in_weights_sum


def local_clustering_coefficients_features(
        edge_index: np.ndarray,
        num_nodes: int,
        weights: np.ndarray = None,
        as_undirected: bool = False,
        dtype: type = np.float64
):
    """ Compute local clustering coefficients for each node in an undirected graph.

    Args:
        edge_index: Graph edges in shape [2, num_edges]
        num_nodes: Number of nodes in the graph
        weights: Edge weights with shape [num_edges, num_weight_types]
        as_undirected: Calculate lcc of undirected version of graph
        dtype: np.float32 or np.float64
    Returns:
        out_degrees: Out degrees for each node [num_nodes]
        in_degrees: In degrees for each node [num_nodes]
        out_weights_sum: Sum of out edge weights for each node and weight type [num_nodes, num_weight_types]
        in_weights_sum: Sum of in edge weights for each node and weight type [num_nodes, num_weight_types]
    """
    edge_index, weights = preprocess_edge_data(
        edge_index=edge_index,
        weights=weights,
        as_symmetric=as_undirected,
        dtype=dtype
    )

    if as_undirected:
        feature_names = ['lcc'] + ([f'w{i}_lcc' for i in range(weights.shape[1])] if weights is not None else [])
    else:
        lcc_types = ['out', 'in', 'cycle', 'mid']
        feature_names = [f'{s}_lcc' for s in lcc_types]
        if weights is not None:
            for lcc_type in lcc_types:
                feature_names += [f'w{i}_{lcc_type}_lcc' for i in range(weights.shape[1])]

    lcc, w_lcc = _compute_local_clustering_coefficients(
        edge_index=edge_index,
        num_nodes=num_nodes,
        as_undirected=as_undirected,
        weights=weights,
        dtype=dtype
    )
    if w_lcc is not None:
        lcc_features = np.concatenate((lcc, w_lcc), axis=1)
    else:
        lcc_features = lcc

    return lcc_features, feature_names


def _compute_num_triplets(adj_out_csc: sp.csc_array, adj_in_csr: sp.csr_array):
    adj_out_csc = adj_out_csc.copy()
    adj_out_csc.data = np.ones_like(adj_out_csc.data)
    adj_in_csr = adj_in_csr.copy()
    adj_in_csr.data = np.ones_like(adj_in_csr.data)
    in_degrees = adj_in_csr.sum(axis=1)
    out_degrees = adj_out_csc.sum(axis=0)
    reciprocal_degree = (adj_in_csr.T * adj_out_csc).sum(axis=0)
    num_out_triplets = out_degrees * (out_degrees - 1)
    num_in_triplets = in_degrees * (in_degrees - 1)
    num_cycle_triplets = in_degrees * out_degrees - reciprocal_degree
    num_middle_triplets = num_cycle_triplets
    return num_out_triplets, num_in_triplets, num_cycle_triplets, num_middle_triplets


def _compute_local_clustering_coefficients(
        edge_index: np.ndarray,
        num_nodes: int,
        as_undirected: bool = False,
        weights: np.ndarray = None,
        dtype: type = np.float64
):
    edge_1d_indices = np.arange(edge_index.shape[1], dtype=np.int64)
    adj = sp.coo_array((edge_1d_indices, (edge_index[1], edge_index[0])), shape=(num_nodes, num_nodes))
    core_f = core32 if dtype == np.float32 else core64
    adj_out_csc = adj.tocsc()
    adj_in_csr = adj.tocsr()

    (num_out_triplets,
     num_in_triplets,
     num_cycle_triplets,
     num_middle_triplets
     ) = _compute_num_triplets(adj_out_csc, adj_in_csr)

    if weights is None:
        (
            num_out_triangles,
            num_in_triangles,
            num_cycle_triangles,
            num_middle_triangles
        ) = core._count_number_triangles_directed(
            out_indices=adj_out_csc.indices.astype(np.int64, copy=False),
            out_indptr=adj_out_csc.indptr.astype(np.int64, copy=False),
            in_indices=adj_in_csr.indices.astype(np.int64, copy=False),
            in_indptr=adj_in_csr.indptr.astype(np.int64, copy=False),
        )
        weights_out_triangles = weights_in_triangles = weights_cycle_triangles = weights_middle_triangles = None

    else:
        weights = weights / np.max(weights, axis=0, keepdims=True)
        (
            num_out_triangles,
            num_in_triangles,
            num_cycle_triangles,
            num_middle_triangles,
            weights_out_triangles,
            weights_in_triangles,
            weights_cycle_triangles,
            weights_middle_triangles
        ) = core_f._count_number_triangles_directed_weighted(
            out_indices=adj_out_csc.indices.astype(np.int64, copy=False),
            out_indptr=adj_out_csc.indptr.astype(np.int64, copy=False),
            in_indices=adj_in_csr.indices.astype(np.int64, copy=False),
            in_indptr=adj_in_csr.indptr.astype(np.int64, copy=False),
            out_edge_index=adj_out_csc.data.astype(np.int64, copy=False),
            in_edge_index=adj_in_csr.data.astype(np.int64, copy=False),
            weights=weights.astype(dtype, copy=False)
        )

    out_lcc = num_out_triangles / np.maximum(num_out_triplets.astype(dtype, copy=False), 1e-6)
    in_lcc = num_in_triangles / np.maximum(num_in_triplets.astype(dtype, copy=False), 1e-6)
    cycle_lcc = num_cycle_triangles / np.maximum(num_cycle_triplets.astype(dtype, copy=False), 1e-6)
    mid_lcc = num_middle_triangles / np.maximum(num_middle_triplets.astype(dtype, copy=False), 1e-6)
    lcc = np.stack((out_lcc, in_lcc, cycle_lcc, mid_lcc), axis=1).astype(dtype=dtype, copy=False)
    if as_undirected:
        lcc = np.mean(lcc, axis=1, keepdims=True)
    if weights is not None:
        w_out_lcc = weights_out_triangles / np.maximum(num_out_triplets.astype(dtype, copy=False), 1e-6).reshape(-1, 1)
        w_in_lcc = weights_in_triangles / np.maximum(num_in_triplets.astype(dtype, copy=False), 1e-6).reshape(-1, 1)
        w_cycle_lcc = weights_cycle_triangles / np.maximum(num_cycle_triplets.astype(dtype, copy=False), 1e-6).reshape(
            -1, 1)
        w_mid_lcc = weights_middle_triangles / np.maximum(num_middle_triplets.astype(dtype, copy=False), 1e-6).reshape(
            -1, 1)
        if as_undirected:
            w_lcc = (w_out_lcc + w_in_lcc + w_cycle_lcc + w_mid_lcc) / 4.
        else:
            w_lcc = np.concatenate((w_out_lcc, w_in_lcc, w_cycle_lcc, w_mid_lcc), axis=1).astype(dtype=dtype,
                                                                                                 copy=False)
    else:
        w_lcc = None
    return lcc, w_lcc


def legacy_egonet_edge_features(
        edge_index: np.ndarray,
        num_nodes: int,
        weights: np.ndarray = None,
        as_undirected: bool = False,
        dtype: type = np.float64
):
    """ Compute local clustering coefficients for each node in an undirected graph.

    Args:
        edge_index: Graph edges in shape [2, num_edges]
        num_nodes: Number of nodes in the graph
        weights: Edge weights with shape [num_edges, num_weight_types]
        as_undirected: Calculate lcc of undirected version of graph
        dtype: np.float32 or np.float64
    Returns:
        out_degrees: Out degrees for each node [num_nodes]
        in_degrees: In degrees for each node [num_nodes]
        out_weights_sum: Sum of out edge weights for each node and weight type [num_nodes, num_weight_types]
        in_weights_sum: Sum of in edge weights for each node and weight type [num_nodes, num_weight_types]
    """
    edge_index, weights = preprocess_edge_data(
        edge_index=edge_index,
        weights=weights,
        as_symmetric=as_undirected,
        dtype=dtype
    )
    if weights is not None:
        assert weights.shape[1] == 1  # Currently only single weights supported
        data = weights.ravel()
    else:
        data = np.ones(edge_index.shape[1], dtype=dtype)
    adj = sp.coo_array((data, (edge_index[1], edge_index[0])), shape=(num_nodes, num_nodes))
    core_f = core32 if dtype == np.float32 else core64
    adj_out_csc = adj.tocsc()
    adj_in_csr = adj.tocsr()

    if as_undirected:
        feature_names = ['lcy_en_edges', 'lcy_en_src', 'lcy_en_dst']
        if weights is not None:
            feature_names += ['w_lcy_en_edges', 'w_lcy_en_src', 'w_lcy_en_dst']
    else:
        feature_names = ['lcy_en_edges', 'lcy_en_src', 'lcy_en_dst']
        if weights is not None:
            feature_names += ['w_lcy_en_edges', 'w_lcy_en_src', 'w_lcy_en_dst']

    if weights is not None:
        features = core_f._extract_egonet_features(
            out_indices=adj_out_csc.indices.astype(np.int64, copy=False),
            out_indptr=adj_out_csc.indptr.astype(np.int64, copy=False),
            out_weights=adj_out_csc.data,
            in_indices=adj_in_csr.indices.astype(np.int64, copy=False),
            in_indptr=adj_in_csr.indptr.astype(np.int64, copy=False),
            in_weights=adj_in_csr.data,
        )
    else:
        features = core_f._extract_egonet_features_no_weights(
            out_indices=adj_out_csc.indices.astype(np.int64, copy=False),
            out_indptr=adj_out_csc.indptr.astype(np.int64, copy=False),
            in_indices=adj_in_csr.indices.astype(np.int64, copy=False),
            in_indptr=adj_in_csr.indptr.astype(np.int64, copy=False),
        )
    return features, feature_names
