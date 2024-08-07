import warnings

import pytest
import numpy as np
import networkx as nx

import accmp.base_features.features as uns_bf
import accmp.base_features._core as core
import accmp.base_features._core_f32 as core32
import accmp.base_features._core_f64 as core64
from tests.utils import get_nx_degrees_as_array, add_weights_and_edge_index_to_nx_graph, get_nx_lcc_as_array


@pytest.fixture
def random_grpah_gmn(directed: bool) -> nx.Graph:
    n = 500
    m = 5000
    graph = nx.gnm_random_graph(n=n, m=m, directed=directed)
    return graph


@pytest.fixture
def graph_with_sl(directed: bool) -> nx.Graph:
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    nx.add_path(graph, [0, 1, 2, 3, 4])
    graph.add_edge(0, 0)
    graph.add_edge(1, 1)
    graph.add_edge(2, 2)
    graph.add_edge(1, 2)
    graph.add_edge(2, 1)
    graph.add_edge(4, 2)
    graph.add_node(5)
    graph.add_node(6)
    return graph


@pytest.fixture()
def five_node_graph_adj():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    graph.add_edge(1, 2)
    graph.add_edge(2, 1)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(2, 4)
    graph.add_edge(1, 4)
    graph.add_edge(4, 3)
    graph.add_edge(4, 2)
    graph.add_edge(4, 1)
    return nx.adjacency_matrix(graph).T.tocoo()


@pytest.mark.parametrize('directed', [True, False])
def test_triangle_counts(random_grpah_gmn, directed):
    """ Test _count_number_triangles_directed against formulas in eq (13) in https://arxiv.org/pdf/physics/0612169.pdf.
    """
    assert random_grpah_gmn.is_directed() == directed

    adj = nx.adjacency_matrix(random_grpah_gmn)
    adj_csc = adj.tocsc()
    adj_csr = adj.tocsr()
    num_out_mm = (adj @ (adj @ adj.T)).diagonal()
    num_in_mm = (adj.T @ (adj @ adj)).diagonal()
    num_cycle_mm = (adj @ (adj @ adj)).diagonal()
    num_mid_mm = (adj @ (adj.T @ adj)).diagonal()

    (num_out_triangles,
     num_in_triangles,
     num_cycle_triangles,
     num_middle_triangles) = core._count_number_triangles_directed(adj_csr.indices.astype(np.int64),
                                                                   adj_csr.indptr.astype(np.int64),
                                                                   adj_csc.indices.astype(np.int64),
                                                                   adj_csc.indptr.astype(np.int64),
                                                                   )

    assert np.allclose(num_out_triangles, num_out_mm)
    assert np.allclose(num_in_triangles, num_in_mm)
    assert np.allclose(num_cycle_triangles, num_cycle_mm)
    assert np.allclose(num_middle_triangles, num_mid_mm)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
@pytest.mark.parametrize('directed', [True, False])
def test_triangle_counts_weighted(random_grpah_gmn, directed, dtype):
    num_weights = 3
    weights = np.random.rand(random_grpah_gmn.number_of_edges(), num_weights).astype(dtype)
    random_grpah_gmn = add_weights_and_edge_index_to_nx_graph(random_grpah_gmn, weights)

    adj = nx.adjacency_matrix(random_grpah_gmn, weight='edge_index').T
    adj_out_csc = adj.tocsc()
    adj_in_csr = adj.tocsr()
    coref = core32 if dtype == np.float32 else core64

    (num_out_triangles,
     num_in_triangles,
     num_cycle_triangles,
     num_middle_triangles,
     weights_out_triangles,
     weights_in_triangles,
     weights_cycle_triangles,
     weights_middle_triangles
     ) = coref._count_number_triangles_directed_weighted(
        out_indices=adj_out_csc.indices.astype(np.int64),
        out_indptr=adj_out_csc.indptr.astype(np.int64),
        in_indices=adj_in_csr.indices.astype(np.int64),
        in_indptr=adj_in_csr.indptr.astype(np.int64),
        out_edge_index=adj_out_csc.data.astype(np.int64),
        in_edge_index=adj_in_csr.data.astype(np.int64),
        weights=weights
    )

    adj = nx.adjacency_matrix(random_grpah_gmn, weight=None)
    num_out_mm = (adj @ (adj @ adj.T)).diagonal()
    num_in_mm = (adj.T @ (adj @ adj)).diagonal()
    num_cycle_mm = (adj @ (adj @ adj)).diagonal()
    num_mid_mm = (adj @ (adj.T @ adj)).diagonal()
    assert np.allclose(num_out_triangles, num_out_mm)
    assert np.allclose(num_in_triangles, num_in_mm)
    assert np.allclose(num_cycle_triangles, num_cycle_mm)
    assert np.allclose(num_middle_triangles, num_mid_mm)

    for col in range(num_weights):
        adj_w = nx.adjacency_matrix(random_grpah_gmn, weight=f'weight{col}')
        adj_w.data = np.power(adj_w.data, 1. / 3.).astype(dtype)
        w_out_mm = (adj_w @ (adj_w @ adj_w.T)).diagonal()
        w_in_mm = (adj_w.T @ (adj_w @ adj_w)).diagonal()
        w_cycle_mm = (adj_w @ (adj_w @ adj_w)).diagonal()
        w_mid_mm = (adj_w @ (adj_w.T @ adj_w)).diagonal()

        assert np.allclose(weights_out_triangles[:, col], w_out_mm)
        assert np.allclose(weights_in_triangles[:, col], w_in_mm)
        assert np.allclose(weights_cycle_triangles[:, col], w_cycle_mm)
        assert np.allclose(weights_middle_triangles[:, col], w_mid_mm)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
@pytest.mark.parametrize('directed', [True, False])
def test_lcc(random_grpah_gmn, directed, dtype):
    adj = nx.adjacency_matrix(random_grpah_gmn).T
    adj_out_csc = adj.tocsc()
    adj_in_csr = adj.tocsr()

    (num_out_triplets,
     num_in_triplets,
     num_cycle_triplets,
     num_middle_triplets
     ) = uns_bf._compute_num_triplets(adj_out_csc, adj_in_csr)

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

    lcc_t = (num_out_triangles + num_in_triangles + num_cycle_triangles + num_middle_triangles)
    lcc_T = (num_out_triplets + num_in_triplets + num_cycle_triplets + num_middle_triplets)
    lcc = lcc_t / np.maximum(lcc_T, 1e-6).astype(dtype, copy=False)
    nx_lcc = nx.clustering(random_grpah_gmn)
    nx_lcc = np.asarray([nx_lcc[i] for i in range(random_grpah_gmn.number_of_nodes())], dtype=dtype)
    assert np.allclose(lcc, nx_lcc)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
@pytest.mark.parametrize('directed', [True, False])
def test_lcc_weighted(random_grpah_gmn, directed: bool, dtype: type):
    num_weights = 3
    weights = np.random.rand(random_grpah_gmn.number_of_edges(), num_weights).astype(dtype)
    random_grpah_gmn = add_weights_and_edge_index_to_nx_graph(random_grpah_gmn, weights)

    adj = nx.adjacency_matrix(random_grpah_gmn, weight='edge_index').T
    adj_out_csc = adj.tocsc()
    adj_in_csr = adj.tocsr()
    coref = core32 if dtype == np.float32 else core64

    (num_out_triplets,
     num_in_triplets,
     num_cycle_triplets,
     num_middle_triplets
     ) = uns_bf._compute_num_triplets(adj_out_csc, adj_in_csr)

    weights = weights / np.max(weights, axis=0, keepdims=True)
    (num_out_triangles,
     num_in_triangles,
     num_cycle_triangles,
     num_middle_triangles,
     weights_out_triangles,
     weights_in_triangles,
     weights_cycle_triangles,
     weights_middle_triangles
     ) = coref._count_number_triangles_directed_weighted(
        out_indices=adj_out_csc.indices.astype(np.int64, copy=False),
        out_indptr=adj_out_csc.indptr.astype(np.int64, copy=False),
        in_indices=adj_in_csr.indices.astype(np.int64, copy=False),
        in_indptr=adj_in_csr.indptr.astype(np.int64, copy=False),
        out_edge_index=adj_out_csc.data.astype(np.int64, copy=False),
        in_edge_index=adj_in_csr.data.astype(np.int64, copy=False),
        weights=weights
    )

    lcc_t = (num_out_triangles + num_in_triangles + num_cycle_triangles + num_middle_triangles)
    lcc_T = (num_out_triplets + num_in_triplets + num_cycle_triplets + num_middle_triplets)
    lcc = lcc_t / np.maximum(lcc_T, 1e-6).astype(dtype, copy=False)

    nx_lcc = get_nx_lcc_as_array(random_grpah_gmn, weight=None, dtype=dtype)
    assert np.allclose(lcc, nx_lcc)

    for col in range(num_weights):
        lcc_t_w = (weights_out_triangles[:, col] + weights_in_triangles[:, col] +
                   weights_cycle_triangles[:, col] + weights_middle_triangles[:, col])
        lcc_w = lcc_t_w / np.maximum(lcc_T, 1e-6).astype(dtype, copy=False)
        nx_lcc_w = get_nx_lcc_as_array(random_grpah_gmn, weight=f'weight{col}', dtype=dtype)

        assert np.allclose(lcc_w, nx_lcc_w)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
@pytest.mark.parametrize('directed', [True, False])
def test_degree_feature_extraction(graph_with_sl: nx.Graph, directed: bool, dtype: type):
    edge_index = np.asarray([(s, t) for s, t in graph_with_sl.edges], dtype=np.int64).T
    num_nodes = graph_with_sl.number_of_nodes()

    num_weights = 3
    weights = np.random.rand(graph_with_sl.number_of_edges(), num_weights).astype(dtype)
    graph_with_sl = add_weights_and_edge_index_to_nx_graph(graph_with_sl, weights)

    degree_feat, feature_names = uns_bf.degree_features(
        edge_index=edge_index,
        num_nodes=num_nodes,
        as_undirected=not directed,
        weights=weights,
        dtype=dtype
    )

    if directed:
        correct_feature_names = ["out_deg", "in_deg",
                                 "w0_out_deg", "w1_out_deg", "w2_out_deg",
                                 "w0_in_deg", "w1_in_deg", "w2_in_deg"]
    else:
        correct_feature_names = ["deg", "w0_deg", "w1_deg", "w2_deg"]
    assert len(feature_names) == len(correct_feature_names)
    assert all([name == corr_name for name, corr_name in zip(feature_names, correct_feature_names)])

    graph_with_sl.remove_edges_from(nx.selfloop_edges(graph_with_sl))
    nx_degs = get_nx_degrees_as_array(graph_with_sl, directed=directed, weight=None, dtype=dtype)
    if directed:
        assert np.allclose(degree_feat[:, :2], nx_degs)
    else:
        assert np.allclose(degree_feat[:, [0]], nx_degs)

    for col in range(num_weights):
        nx_w_degs = get_nx_degrees_as_array(graph_with_sl, directed=directed, weight=f'weight{col}', dtype=dtype)

        if directed:
            feature_slice = [2 + col, 2 + 3 + col]
            assert np.allclose(degree_feat[:, feature_slice], nx_w_degs)
        else:
            feature_slice = [1 + col]
            assert np.allclose(degree_feat[:, feature_slice], nx_w_degs)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
@pytest.mark.parametrize('directed', [True, False])
def test_lcc_feature_extraction(graph_with_sl: nx.Graph, directed: bool, dtype: type):
    edge_index = np.asarray([(s, t) for s, t in graph_with_sl.edges], dtype=np.int64).T
    num_nodes = graph_with_sl.number_of_nodes()

    num_weights = 3
    weights = np.random.rand(graph_with_sl.number_of_edges(), num_weights).astype(dtype)
    graph_with_sl = add_weights_and_edge_index_to_nx_graph(graph_with_sl, weights)

    lcc_feat, feature_names = uns_bf.local_clustering_coefficients_features(
        edge_index=edge_index,
        num_nodes=num_nodes,
        as_undirected=not directed,
        weights=weights,
        dtype=dtype
    )

    if directed:
        correct_feature_names = ["out_lcc", "in_lcc", "cycle_lcc", "mid_lcc",
                                 "w0_out_lcc", "w1_out_lcc", "w2_out_lcc",
                                 "w0_in_lcc", "w1_in_lcc", "w2_in_lcc",
                                 "w0_cycle_lcc", "w1_cycle_lcc", "w2_cycle_lcc",
                                 "w0_mid_lcc", "w1_mid_lcc", "w2_mid_lcc"]
    else:
        correct_feature_names = ["lcc", "w0_lcc", "w1_lcc", "w2_lcc"]
    assert len(feature_names) == len(correct_feature_names)
    assert all([name == corr_name for name, corr_name in zip(feature_names, correct_feature_names)])

    graph_with_sl.remove_edges_from(nx.selfloop_edges(graph_with_sl))
    nx_lcc = get_nx_lcc_as_array(graph_with_sl, weight=None, dtype=dtype)
    if directed:
        warnings.warn("No reference implementation for directed lcc available.")
    else:
        assert np.allclose(lcc_feat[:, 0], nx_lcc)

    for col in range(num_weights):
        nx_w_lcc = get_nx_lcc_as_array(graph_with_sl, weight=f'weight{col}', dtype=dtype)

        if directed:
            warnings.warn("No reference implementation for directed lcc available.")
        else:
            assert np.allclose(lcc_feat[:, col + 1], nx_w_lcc)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
@pytest.mark.parametrize('directed', [True])
@pytest.mark.parametrize('use_weights', [True, False])
def test_egonet_features(five_node_graph_adj, directed, use_weights, dtype: type):
    edge_index = np.stack((five_node_graph_adj.col, five_node_graph_adj.row), axis=0)
    num_nodes = five_node_graph_adj.shape[0]
    if not use_weights:
        weights = None
    else:
        weights = np.ones(edge_index.shape[1], dtype=dtype)
    ego_features, feature_names = uns_bf.legacy_egonet_edge_features(
        edge_index=edge_index,
        num_nodes=num_nodes,
        weights=weights,
        as_undirected=not directed,
        dtype=dtype
    )
    answer = np.asarray([[6., 3., 3., 6., 3., 3.],
                         [8., 3., 1., 8., 3., 1.],
                         [12., 0., 0., 12., 0., 0.],
                         [7., 3., 2., 7., 3., 2.],
                         [9., 0., 3., 9., 0., 3.]])
    if use_weights:
        assert np.allclose(ego_features, answer)
    else:
        assert np.allclose(ego_features, answer[:, :3])
