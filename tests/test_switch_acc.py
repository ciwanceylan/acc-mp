import pytest

import numpy as np
import torch
import accmp.preprocessing as preproc
import accmp.acc as acc
import accmp.transforms as accmptrns

from tests.alignment import load_graph


def test_rw_normalized_adj_dir():
    num_steps = 4
    dtype_np = np.float32
    dtype_torch = torch.float32
    as_undirected = False
    graph = load_graph("usairport")
    node_attr = np.random.randn(graph.num_nodes, 3).astype(dtype=dtype_np)

    edge_index = graph.edges.T

    init_params = preproc.InitFeaturesWeightsParams(
        use_degree=True,
        use_log1p_degree=False,
        use_lcc=True,
        use_weights=False,
        use_node_attributes=True,
        as_undirected=False,
        dtype=dtype_np
    )

    normalization = accmptrns.FeatureNormalization(
        mode=None,
        subtract_mean=True,
        before_prune=True,
        before_propagate=False
    )

    params = acc.ACCParams(
        max_steps=num_steps,
        initial_feature_standardization=accmptrns.FeatureNormalization(mode='std', subtract_mean=True,
                                                                       before_prune=False, before_propagate=False),
        mp_feature_normalization=normalization,
        init_params=init_params,
        decomposed_layers=1,
        normalized_weights=True,
        min_add_dim=2,
        max_dim=512,
        return_us=False,
        sv_thresholding='rtol',
        theta=1e-8,
        use_rsvd=False,
    )

    switch_embeddings = acc.switch_agg_compress_cat_embeddings(
        edge_index=edge_index, num_nodes=graph.num_nodes, num_dir_steps=num_steps, params=params,
        node_attributes=node_attr, device=torch.device('cpu'), add_self_loops_to_sinks=False
    )

    acc_embeddings = acc.agg_compress_cat_embeddings(
        edge_index=edge_index, num_nodes=graph.num_nodes, directed_conv=True, params=params,
        node_attributes=node_attr, device=torch.device('cpu')
    )

    np.testing.assert_allclose(switch_embeddings, acc_embeddings)
