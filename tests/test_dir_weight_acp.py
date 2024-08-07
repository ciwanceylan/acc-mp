import pytest

import numpy as np
import torch
import torch_sparse as tsp
# import unstructnes.aggcap.agg_cat_prune as acp
import accmp.transforms
import accmp.layers as layers
import accmp.preprocessing as preproc

from tests.alignment import load_graph


def get_og_initial_features(edge_index, num_nodes, weights, init_params):
    bf_params = preproc.InitFeaturesWeightsParams(
        use_weights=init_params.use_weights,
        use_node_attributes=False,
        as_undirected=init_params.as_undirected,
        use_degree=init_params.use_degree,
        use_lcc=init_params.use_lcc,
        use_log1p_degree=False,
        dtype=init_params.dtype
    )
    if bf_params.as_undirected:
        edge_index, weights = preproc.to_undirected(edge_index=edge_index, weights=weights)

    feat_factory = preproc.FeaturesAndWeightsFactory(edge_index=edge_index, num_nodes=num_nodes, weights=weights,
                                                     init_params=init_params)
    features, _ = feat_factory.get_initial_features()
    return features


@pytest.mark.parametrize("graph_name", ["arenas", "usairport"])
@pytest.mark.parametrize("as_undirected", [True])
def test_get_adj_and_initial_features_no_weights(graph_name: str, as_undirected: bool):
    dtype_np = np.float32
    graph = load_graph(graph_name)
    node_attr = np.random.randn(graph.num_nodes, 3).astype(dtype=dtype_np)

    edge_index = graph.edges.T

    init_params = preproc.InitFeaturesWeightsParams(
        use_weights=True,
        use_node_attributes=True,
        as_undirected=as_undirected,
        use_degree=True,
        use_lcc=True,
        use_log1p_degree=True,
        dtype=dtype_np
    )

    adj_ws2t_t_W, adj_wt2s_W, initial_features_W, _, _ = preproc.create_adj_t_weights_and_initial_states(
        edge_index=edge_index,
        num_nodes=graph.num_nodes,
        init_params=init_params,
        normalized_weights=True,
        weights=None,
        node_attributes=node_attr
    )

    adj_ws2t_t, adj_wt2s, initial_features = preproc.create_adj_t_weights_and_initial_states_no_weights(
        edge_index=edge_index,
        num_nodes=graph.num_nodes,
        init_params=init_params,
        normalized_weights=True,
        node_attributes=node_attr
    )

    torch.testing.assert_close(initial_features_W, initial_features)
    torch.testing.assert_close(adj_ws2t_t_W.to_dense(), adj_ws2t_t.to_dense())
    torch.testing.assert_close(adj_wt2s_W.to_dense(), adj_wt2s.to_dense())


def manual_dir_conv(graph, init_params, feat_norm: accmp.transforms.FeatureNormalization):
    adj_ws2t_t, adj_wt2s, initial_features, _, _ = preproc.create_adj_t_weights_and_initial_states(
        edge_index=graph.edges.T,
        num_nodes=graph.num_nodes,
        init_params=init_params,
        normalized_weights=True,
        weights=None,
        node_attributes=None
    )

    adj_ws2t_t = adj_ws2t_t.set_value(adj_ws2t_t.storage.value().view(-1))
    adj_wt2s = adj_wt2s.set_value(adj_wt2s.storage.value().view(-1))

    x = accmp.transforms.normalize_features(initial_features, feat_norm)
    new_x = tsp.matmul(adj_ws2t_t, x)
    new_x = accmp.transforms.normalize_features(new_x, feat_norm)

    new_x_t = tsp.matmul(adj_wt2s, x)
    new_x_t = accmp.transforms.normalize_features(new_x_t, feat_norm)

    x = torch.cat((new_x, new_x_t), dim=1)
    return x


def test_directed_convs():
    dtype = np.float32
    graph = load_graph("usairport")
    feat_norm = accmp.transforms.FeatureNormalization(mode='sphere', subtract_mean=False,
                                                      before_prune=True, before_propagate=True)

    init_params = preproc.InitFeaturesWeightsParams(
        use_weights=False,
        use_node_attributes=False,
        as_undirected=True,
        use_degree=True,
        use_lcc=True,
        use_log1p_degree=False,
        dtype=dtype
    )

    adj_ws2t_t, adj_wt2s, initial_features, _, _ = preproc.create_adj_t_weights_and_initial_states(
        edge_index=graph.edges.T,
        num_nodes=graph.num_nodes,
        init_params=init_params,
        normalized_weights=True,
        weights=None,
        node_attributes=None
    )
    agg_layer = layers.MsgAggStep(use_dir_conv=True, feature_normalization=feat_norm)
    dir_weight_embeddings = agg_layer(initial_features, adj_ws2t_t, adj_wt2s)

    ref_embeddings = manual_dir_conv(graph, init_params=init_params, feat_norm=feat_norm)
    np.testing.assert_allclose(ref_embeddings.numpy(), dir_weight_embeddings.numpy(), rtol=1e-5)
