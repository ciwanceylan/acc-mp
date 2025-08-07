import dataclasses as dc
from functools import cached_property
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch_sparse as tsp

import accmp.base_features.features as bf_core


@dc.dataclass(frozen=True)
class InitFeaturesWeightsParams:
    use_degree: bool
    use_log1p_degree: bool
    use_lcc: bool
    use_weights: bool
    use_node_attributes: bool
    as_undirected: bool
    dtype: type = np.float32


class FeaturesAndWeightsFactory:

    def __init__(
            self,
            *,
            edge_index: np.ndarray,
            num_nodes: int,
            weights: np.ndarray = None,
            weight_names: Tuple[str] = None,
            node_attributes: np.ndarray = None,
            init_params: InitFeaturesWeightsParams,
    ):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.weights = weights
        self.node_attributes = node_attributes
        self.init_params = init_params
        if weights is not None:
            assert len(weights.shape) > 1  # Weights should be [num_edges x num_weightings]
            if weight_names is None:
                weight_names = [f"w{i}" for i in range(weights.shape[1])]
        self.weight_names = weight_names

    def get_initial_features(self):
        initial_features = []
        initial_feature_names = []

        if self.init_params.use_degree:
            deg_features, feature_names = self.degree_features
            initial_features.append(deg_features)
            initial_feature_names.extend(feature_names)

        if self.init_params.use_log1p_degree:
            log1p_deg_features, feature_names = self.log_degree_features
            initial_features.append(log1p_deg_features)
            initial_feature_names.extend(feature_names)

        if self.init_params.use_lcc:
            lcc_features, feature_names = self.lcc_features
            initial_features.append(lcc_features)
            initial_feature_names.extend(feature_names)

        if self.node_attributes is not None:
            initial_features.append(self.node_attributes)
            initial_feature_names.extend([f"na{i}" for i in range(self.node_attributes.shape[1])])

        initial_features = np.concatenate(initial_features, axis=1)
        return initial_features.astype(self.init_params.dtype), initial_feature_names

    def get_weights(self, normalized: bool):
        weight_names = []
        s2t_weights = [np.ones((self.edge_index.shape[1], 1), dtype=self.init_params.dtype)]
        t2s_weights = [np.ones((self.edge_index.shape[1], 1), dtype=self.init_params.dtype)]

        if self.weights is not None:
            s2t_weights.append(self.weights)
            t2s_weights.append(self.weights)
            weight_names.extend(list(self.weight_names))

        s2t_weights = np.concatenate(s2t_weights, axis=1)
        t2s_weights = np.concatenate(t2s_weights, axis=1)

        if normalized:
            s2t_weights = self.normalize_weights(
                edge_ends=self.edge_index[1],
                weights=s2t_weights
            )
            t2s_weights = self.normalize_weights(
                edge_ends=self.edge_index[0],
                weights=t2s_weights
            )
            weight_names = ["mean"] + weight_names
        else:
            weight_names = ["sum"] + weight_names

        return s2t_weights.astype(self.init_params.dtype), t2s_weights.astype(self.init_params.dtype), weight_names

    @staticmethod
    def normalize_weights(edge_ends: np.ndarray, weights: np.ndarray):
        df = pd.DataFrame(weights, dtype=np.float64)
        df["edge_ends"] = edge_ends
        weight_sums = df.groupby("edge_ends").transform("sum")
        weight_sums.replace(to_replace=0., value=1., inplace=True)
        df = df.drop(columns=["edge_ends"])
        norm_weights = df / weight_sums
        return norm_weights.to_numpy()

    @cached_property
    def degree_features(self):
        deg_features, feature_names = bf_core.degree_features(
            edge_index=self.edge_index, num_nodes=self.num_nodes,
            as_undirected=self.init_params.as_undirected,
            weights=self.weights, dtype=self.init_params.dtype)
        return deg_features, feature_names

    @property
    def log_degree_features(self):
        deg_features, feature_names = self.degree_features
        feature_names = [name + "_log" for name in feature_names]
        return np.log1p(deg_features), feature_names

    @cached_property
    def lcc_features(self):
        lcc_features, feature_names = bf_core.local_clustering_coefficients_features(
            edge_index=self.edge_index, num_nodes=self.num_nodes,
            as_undirected=self.init_params.as_undirected,
            weights=self.weights, dtype=self.init_params.dtype
        )
        return lcc_features, feature_names

    @cached_property
    def undir_degree(self):
        deg_features, feature_names = self.degree_features
        if feature_names[0] == "deg":
            degrees = deg_features[:, 0]
        elif set(feature_names[:2]) == {'out_deg', 'in_deg'}:
            degrees = deg_features[:, 0] + deg_features[:, 1]
        else:
            raise NotImplementedError("Degree not found. This hacky implementation should be fixed.")
        return degrees

    @cached_property
    def undir_log1p_degree(self):
        deg_features, feature_names = self.degree_features
        if feature_names[0] == "deg":
            degrees = np.log1p(deg_features[:, 0])
        elif set(feature_names[:2]) == {'out_deg', 'in_deg'}:
            degrees = np.log1p(deg_features[:, 0] + deg_features[:, 1])
        else:
            raise NotImplementedError("Degree not found. This hacky implementation should be fixed.")
        return degrees

    @cached_property
    def undir_lcc(self):
        lcc_features, feature_names = self.lcc_features
        if feature_names[0] == "lcc":
            lcc = np.log1p(lcc_features[:, 0])
        elif set(feature_names[:4]) == {'out_lcc', 'in_lcc', 'cycle_lcc', 'mid_lcc'}:
            lcc = np.mean(lcc_features[:, :4], axis=1)
        else:
            raise NotImplementedError("LCCs not found. This hacky implementation should be fixed.")
        return lcc


def prepare_inputs(edge_index: np.ndarray, num_nodes: int,
                   params: InitFeaturesWeightsParams,
                   weights: np.ndarray = None,
                   node_attributes: np.ndarray = None):
    edge_index = edge_index.astype(np.int64, copy=False)
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T

    if params.use_node_attributes and node_attributes is not None:
        assert num_nodes == node_attributes.shape[0]
        if node_attributes.ndim == 1:
            node_attributes = node_attributes.reshape(-1, 1)
        node_attributes = node_attributes.astype(params.dtype, copy=False)
    else:
        node_attributes = None

    if params.use_weights and weights is not None:
        assert edge_index.shape[1] == weights.shape[0]
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
    else:
        weights = None
    return edge_index, weights, node_attributes


def to_undirected(edge_index: np.ndarray, weights: Optional[np.ndarray]):
    edges = np.concatenate((edge_index.T, np.flip(edge_index, 0).T), axis=0)
    df = pd.DataFrame({"src": edges[:, 0], "dst": edges[:, 1]})
    names = None
    if weights is not None:
        names = [f"w{i}" for i in range(weights.shape[1])]
        df_w = pd.DataFrame(np.concatenate((weights, weights), axis=0), columns=names)
        df = pd.concat((df, df_w), axis=1)
    undir_df = df.groupby(["src", "dst"], as_index=False).agg("sum")
    edge_index = undir_df.loc[:, ["src", "dst"]].to_numpy().T
    if names is not None:
        weights = undir_df.loc[:, names].to_numpy()
    return edge_index, weights


def create_adj_t_weights_and_initial_states(
        edge_index: np.ndarray,
        num_nodes: int,
        init_params: InitFeaturesWeightsParams,
        normalized_weights: bool,
        weights: np.ndarray = None,
        node_attributes: np.ndarray = None,
        verbose: bool = False
):
    if verbose:
        print("Preparing inputs...")
    edge_index, weights, node_attributes = prepare_inputs(edge_index=edge_index, num_nodes=num_nodes,
                                                          params=init_params, weights=weights,
                                                          node_attributes=node_attributes)
    if init_params.as_undirected:
        if verbose:
            print("Convert to undirected...")
        edge_index, weights = to_undirected(edge_index=edge_index, weights=weights)

    init_factory = FeaturesAndWeightsFactory(
        edge_index=edge_index,
        num_nodes=num_nodes,
        weights=weights,
        node_attributes=node_attributes,
        init_params=init_params
    )
    if verbose:
        print("Compute initial features...")
    init_features, initial_feature_names = init_factory.get_initial_features()
    if verbose:
        print("Compute weights...")
    s2t_weights, t2s_weights, weight_names = init_factory.get_weights(normalized=normalized_weights)

    if verbose:
        print("Convert from numpy to torch...")
    initial_features = torch.from_numpy(init_features)
    adj_wt2s = tsp.SparseTensor.from_edge_index(
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(t2s_weights),
        sparse_sizes=(num_nodes, num_nodes)
    )

    adj_ws2t_t = tsp.SparseTensor.from_edge_index(
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(s2t_weights),
        sparse_sizes=(num_nodes, num_nodes)
    ).t()
    if verbose:
        print("Initial features and adjacency matrix setup done!")
    return adj_ws2t_t, adj_wt2s, initial_features, initial_feature_names, weight_names


def _compute_initial_features(
        edge_index: np.ndarray,
        out_degrees: torch.Tensor,
        in_degrees: torch.Tensor,
        num_nodes: int,
        init_params: InitFeaturesWeightsParams,
        node_attributes: np.ndarray = None,
        verbose: bool = False
):
    initial_features = []

    if init_params.use_degree:
        if init_params.as_undirected:
            deg_features = out_degrees
        else:
            deg_features = torch.cat((out_degrees, in_degrees), dim=1)
        initial_features.append(deg_features)

    if init_params.use_log1p_degree:
        if init_params.as_undirected:
            log1p_deg_features = torch.log1p(out_degrees)
        else:
            log1p_deg_features = torch.log1p(torch.cat((out_degrees, in_degrees), dim=1))
        initial_features.append(log1p_deg_features)

    if init_params.use_lcc:
        if verbose:
            print("Computing LCC...")
        lcc_features, _ = bf_core._compute_local_clustering_coefficients(
            edge_index=edge_index,
            num_nodes=num_nodes,
            as_undirected=init_params.as_undirected,
            weights=None,
            dtype=init_params.dtype
        )
        initial_features.append(torch.from_numpy(lcc_features))

    if node_attributes is not None:
        initial_features.append(torch.from_numpy(node_attributes))

    initial_features = torch.cat(initial_features, dim=1)
    return initial_features


def create_adj_t_weights_and_initial_states_no_weights(
        edge_index: np.ndarray,
        num_nodes: int,
        init_params: InitFeaturesWeightsParams,
        normalized_weights: bool,
        node_attributes: np.ndarray = None,
        verbose: bool = False
):
    npdtype2thdtype = {np.float16: torch.float16,
                       np.float32: torch.float32,
                       np.float64: torch.float64}
    assert edge_index.shape[0] == 2

    if verbose:
        print("Removing self-loops...")
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # remove self-loops

    if init_params.as_undirected:
        if verbose:
            print("Convert to undirected...")
        edge_index = np.concatenate((edge_index, np.flip(edge_index, 0)), axis=1)

    if verbose:
        print("Removing duplicate edges...")
    edge_index = pd.DataFrame(edge_index.T).drop_duplicates().to_numpy().T

    if verbose:
        print("Creating adjacency matrices..")

    adj_wt2s = tsp.SparseTensor.from_edge_index(
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.ones((edge_index.shape[1], 1), dtype=npdtype2thdtype[init_params.dtype]),
        sparse_sizes=(num_nodes, num_nodes)
    )

    adj_ws2t_t = adj_wt2s.t()

    if verbose:
        print("Computing degrees...")
    in_degrees = adj_ws2t_t.sum(dim=1)
    out_degrees = adj_wt2s.sum(dim=1)

    if normalized_weights:
        if verbose:
            print("Normalizing adjacency matrices...")
        adj_ws2t_t = adj_ws2t_t * (1. / torch.maximum(in_degrees.view(-1, 1, 1), torch.ones(1, dtype=in_degrees.dtype)))
        adj_wt2s = adj_wt2s * (1. / torch.maximum(out_degrees.view(-1, 1, 1), torch.ones(1, dtype=out_degrees.dtype)))

    if verbose:
        print("Creating initial embeddings...")

    initial_features = _compute_initial_features(
        edge_index=edge_index,
        num_nodes=num_nodes,
        out_degrees=out_degrees,
        in_degrees=in_degrees,
        init_params=init_params,
        node_attributes=node_attributes,
        verbose=verbose
    )

    if verbose:
        print("Initial features and adjacency matrix setup done!")
    return adj_ws2t_t, adj_wt2s, initial_features


def add_self_loops_to_zero_deg_nodes(adj: tsp.SparseTensor, degrees: torch.Tensor):
    if adj is None:
        return adj
    zero_deg_nodes = torch.nonzero(~(degrees.view(-1) > 0)).view(-1)
    rows = zero_deg_nodes
    cols = zero_deg_nodes
    values = torch.ones((len(zero_deg_nodes)), dtype=adj.dtype(), device=adj.device())
    new_diag = tsp.SparseTensor(row=rows, col=cols, value=values, sparse_sizes=tuple(adj.sizes()))
    adj = adj + new_diag
    return adj


def rw_adj_normalization(adj: tsp.SparseTensor, degrees: torch.Tensor):
    if adj is None:
        return adj
    adj = adj * (1. / torch.maximum(degrees.view(-1, 1), torch.ones(1, dtype=degrees.dtype)))
    return adj


def create_rw_normalized_adj_matrices(
        edge_index: np.ndarray,
        num_nodes: int,
        init_params: InitFeaturesWeightsParams,
        node_attributes: np.ndarray = None,
        add_self_loops_to_sinks: bool = True,
        verbose: bool = False
):
    npdtype2thdtype = {np.float16: torch.float16,
                       np.float32: torch.float32,
                       np.float64: torch.float64}
    assert edge_index.shape[0] == 2

    if verbose:
        print("Removing self-loops...")
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # remove self-loops

    if init_params.as_undirected:
        if verbose:
            print("Convert to undirected...")
        edge_index = np.concatenate((edge_index, np.flip(edge_index, 0)), axis=1)

    if verbose:
        print("Removing duplicate edges...")
    edge_index = pd.DataFrame(edge_index.T).drop_duplicates().to_numpy().T

    if verbose:
        print("Creating adjacency matrices..")

    if init_params.as_undirected:
        adj_undir = tsp.SparseTensor.from_edge_index(
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.ones((edge_index.shape[1]), dtype=npdtype2thdtype[init_params.dtype]),
            sparse_sizes=(num_nodes, num_nodes)
        )
        adj_wt2s = None
        adj_ws2t_t = None

    else:
        adj_wt2s = tsp.SparseTensor.from_edge_index(
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.ones((edge_index.shape[1]), dtype=npdtype2thdtype[init_params.dtype]),
            sparse_sizes=(num_nodes, num_nodes)
        )

        adj_ws2t_t = adj_wt2s.t()
        adj_undir = adj_wt2s + adj_ws2t_t

    if verbose:
        print("Computing degrees...")
    degrees = adj_undir.sum(dim=1)
    if adj_ws2t_t is not None and adj_wt2s is not None:
        in_degrees = adj_ws2t_t.sum(dim=1)
        out_degrees = adj_wt2s.sum(dim=1)
    else:
        in_degrees = degrees
        out_degrees = degrees


    if add_self_loops_to_sinks:
        adj_ws2t_t = add_self_loops_to_zero_deg_nodes(adj_ws2t_t, degrees=in_degrees)
        adj_wt2s = add_self_loops_to_zero_deg_nodes(adj_wt2s, degrees=out_degrees)
        adj_undir = add_self_loops_to_zero_deg_nodes(adj_undir, degrees=degrees)

    adj_ws2t_t = rw_adj_normalization(adj_ws2t_t, in_degrees)
    adj_wt2s = rw_adj_normalization(adj_wt2s, out_degrees)
    adj_undir = rw_adj_normalization(adj_undir, degrees)

    if verbose:
        print("Creating initial embeddings...")

    initial_features = _compute_initial_features(
        edge_index=edge_index,
        num_nodes=num_nodes,
        out_degrees=out_degrees.view(-1, 1),
        in_degrees=in_degrees.view(-1, 1),
        init_params=init_params,
        node_attributes=node_attributes,
        verbose=verbose
    )

    return adj_undir, adj_ws2t_t, adj_wt2s, initial_features
