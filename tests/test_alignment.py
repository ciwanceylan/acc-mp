import pytest

import numpy as np
import torch

from sklearn.decomposition import PCA

import accmp.acc as acc
import accmp.pcapass as pcapass
import accmp.transforms
import accmp.preprocessing as preproc

from tests.alignment import load_alignment_problem, eval_topk_sim

USAIRPORT_ALIGNMENT_ACC_GOALS = {
    '0': {1: 0.88, 5: 0.97, 10: 0.995},
    '0.05': {1: 0.71, 5: 0.86, 10: 0.89, },
}

ARENAS_ALIGNMENT_ACC_GOALS = {
    '0': {1: 0.97, 5: 0.9999, 10: 0.9999},
    '0.05': {1: 0.74, 5: 0.87, 10: 0.9, },
}


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("noise_p", ['0', '0.05'])
def test_pcapass_us(dtype: type, noise_p: float):
    seed = 4042

    merged_graph, alignment_obj = load_alignment_problem("usairport", noise_p=float(noise_p), seed=seed)
    params = pcapass.PCAPassParams(
        max_steps=8,
        initial_feature_standardization=accmp.transforms.FeatureNormalization(mode='std', subtract_mean=True),
        mp_feature_normalization=accmp.transforms.FeatureNormalization(mode=None, subtract_mean=True),
        mp_instance_normalization=accmp.transforms.FeatureNormalization(mode=None, subtract_mean=False,
                                                                        before_prune=False, before_propagate=False),
        init_params=preproc.InitFeaturesWeightsParams(
            use_weights=False,
            use_node_attributes=False,
            as_undirected=False,
            use_degree=True,
            use_lcc=True,
            use_log1p_degree=False,
            dtype=dtype
        ),
        max_dim=64,
        decomposed_layers=1,
    )

    embeddings = pcapass.pcapass_embeddings(
        directed_conv=True,
        edge_index=merged_graph.edges.T,
        num_nodes=merged_graph.num_nodes,
        params=params,
        device=torch.device('cpu'),
        weights=None,
        node_attributes=None
    )
    assert embeddings.dtype == dtype
    embeddings = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    embeddings = embeddings / np.std(embeddings, axis=0, keepdims=True)
    # embeddings, _ = whitning.WhiteningTransform.fit_transform_whiten(embeddings)

    res = eval_topk_sim(embeddings, alignment_obj)
    for k, v in USAIRPORT_ALIGNMENT_ACC_GOALS[noise_p].items():
        assert res[k] >= v


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("noise_p", ['0', '0.05'])
def test_acc_us(dtype: type, noise_p: float):
    seed = 4042

    merged_graph, alignment_obj = load_alignment_problem("usairport", noise_p=float(noise_p), seed=seed)
    params = acc.ACCParams(
        max_steps=8,
        initial_feature_standardization=accmp.transforms.FeatureNormalization(mode='std', subtract_mean=True),
        mp_feature_normalization=accmp.transforms.FeatureNormalization(mode=None, subtract_mean=False),
        init_params=preproc.InitFeaturesWeightsParams(
            use_weights=False,
            use_node_attributes=False,
            as_undirected=False,
            use_degree=True,
            use_lcc=True,
            use_log1p_degree=False,
            dtype=dtype
        ),
        max_dim=64,
        decomposed_layers=1,
    )

    embeddings = acc.agg_compress_cat_embeddings(
        directed_conv=True,
        edge_index=merged_graph.edges.T,
        num_nodes=merged_graph.num_nodes,
        params=params,
        device=torch.device('cpu'),
        weights=None,
        node_attributes=None
    )
    assert embeddings.dtype == dtype
    embeddings = PCA(whiten=True).fit_transform(embeddings)

    res = eval_topk_sim(embeddings, alignment_obj)
    for k, v in USAIRPORT_ALIGNMENT_ACC_GOALS[noise_p].items():
        assert res[k] >= v


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("noise_p", ['0', '0.05'])
def test_acc_arenas(dtype: type, noise_p: float):
    seed = 4042

    merged_graph, alignment_obj = load_alignment_problem("arenas", noise_p=float(noise_p), seed=seed)
    params = acc.ACCParams(
        max_steps=8,
        initial_feature_standardization=accmp.transforms.FeatureNormalization(mode='std', subtract_mean=True),
        mp_feature_normalization=accmp.transforms.FeatureNormalization(mode=None, subtract_mean=False),
        init_params=preproc.InitFeaturesWeightsParams(
            use_weights=False,
            use_node_attributes=False,
            as_undirected=False,
            use_degree=True,
            use_lcc=True,
            use_log1p_degree=False,
            dtype=dtype
        ),
        max_dim=64,
        decomposed_layers=1,
    )

    embeddings = acc.agg_compress_cat_embeddings(
        directed_conv=False,
        edge_index=merged_graph.edges.T,
        num_nodes=merged_graph.num_nodes,
        params=params,
        device=torch.device('cpu'),
        weights=None,
        node_attributes=None
    )
    assert embeddings.dtype == dtype
    embeddings = PCA(whiten=True).fit_transform(embeddings)

    res = eval_topk_sim(embeddings, alignment_obj)
    for k, v in ARENAS_ALIGNMENT_ACC_GOALS[noise_p].items():
        assert res[k] >= v


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("noise_p", ['0', '0.05'])
def test_acc_arenas(dtype: type, noise_p: str):
    seed = 4042

    merged_graph, alignment_obj = load_alignment_problem("arenas", noise_p=float(noise_p), seed=seed)
    params = acc.ACCParams(
        max_steps=8,
        initial_feature_standardization=accmp.transforms.FeatureNormalization(mode='std', subtract_mean=True),
        mp_feature_normalization=accmp.transforms.FeatureNormalization(mode=None, subtract_mean=True),
        init_params=preproc.InitFeaturesWeightsParams(
            use_weights=False,
            use_node_attributes=False,
            as_undirected=True,
            use_degree=True,
            use_lcc=True,
            use_log1p_degree=False,
            dtype=dtype
        ),
        max_dim=64,
        decomposed_layers=1,
    )

    embeddings = acc.switch_agg_compress_cat_embeddings(
        num_dir_steps=0,
        dim_factor_for_directed=1,
        edge_index=merged_graph.edges.T,
        num_nodes=merged_graph.num_nodes,
        params=params,
        device=torch.device('cpu'),
        node_attributes=None
    )
    assert embeddings.dtype == dtype
    embeddings = PCA(whiten=True).fit_transform(embeddings)

    res = eval_topk_sim(embeddings, alignment_obj)
    for k, v in ARENAS_ALIGNMENT_ACC_GOALS[noise_p].items():
        assert res[k] >= v
