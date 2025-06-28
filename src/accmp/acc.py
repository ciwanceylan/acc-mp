import dataclasses as dc
import numpy as np
import torch

import accmp.models as aggcap_models
import accmp.preprocessing as preproc
import accmp.compression as comprs
from accmp.transforms import FeatureNormalization


@dc.dataclass(frozen=True)
class ACCParams:
    max_steps: int
    max_dim: int
    initial_feature_standardization: FeatureNormalization
    mp_feature_normalization: FeatureNormalization
    init_params: preproc.InitFeaturesWeightsParams
    min_add_dim: int = 2
    sv_thresholding: comprs.SV_THRESHOLDING = 'rtol'
    theta: float = 1e-8
    return_us: bool = False
    use_rsvd: bool = False
    normalized_weights: bool = True
    decomposed_layers: int = 1


def agg_compress_cat_embeddings(edge_index: np.ndarray, num_nodes: int, directed_conv: bool,
                                params: ACCParams, device: torch.device,
                                weights: np.ndarray = None, node_attributes: np.ndarray = None,
                                return_np: bool = True, verbose: bool = False):
    model = aggcap_models.ACC(
        use_dir_conv=directed_conv,
        max_dim=params.max_dim,
        min_add_dim=params.min_add_dim,
        mp_feature_normalization=params.mp_feature_normalization,
        initial_feature_standardization=params.initial_feature_standardization,
        return_us=params.return_us,
        use_rsvd=params.use_rsvd,
        sv_thresholding=params.sv_thresholding,
        theta=params.theta,
        decomposed_layers=params.decomposed_layers,
        verbose=verbose
    )
    if verbose:
        print("Building adjacency matrices and initial embeddings...")
    if weights is None:
        adj_ws2t_t, adj_wt2s, initial_features = preproc.create_adj_t_weights_and_initial_states_no_weights(
            edge_index=edge_index,
            num_nodes=num_nodes,
            init_params=params.init_params,
            normalized_weights=params.normalized_weights,
            node_attributes=node_attributes,
            verbose=verbose
        )
    else:
        adj_ws2t_t, adj_wt2s, initial_features, _, _ = preproc.create_adj_t_weights_and_initial_states(
            edge_index=edge_index,
            num_nodes=num_nodes,
            init_params=params.init_params,
            normalized_weights=params.normalized_weights,
            weights=weights,
            node_attributes=node_attributes,
            verbose=verbose
        )

    if verbose:
        print("Calling ACC model...")
    embeddings = model(
        initial_features.to(device),
        adj_ws2t_t=adj_ws2t_t.to(device),
        adj_wt2s=adj_wt2s.to(device) if directed_conv else None,
        num_steps=params.max_steps,
    )
    if return_np and isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    return embeddings


def switch_agg_compress_cat_embeddings(edge_index: np.ndarray, num_nodes: int, num_dir_steps: int,
                                       dim_factor_for_directed: int, params: ACCParams,
                                       device: torch.device, node_attributes: np.ndarray = None,
                                       return_np: bool = True, add_self_loops_to_sinks: bool = True,
                                       verbose: bool = False):
    model = aggcap_models.SwitchACC(
        num_dir_steps=num_dir_steps,
        max_dim=params.max_dim,
        min_add_dim=params.min_add_dim,
        mp_feature_normalization=params.mp_feature_normalization,
        initial_feature_standardization=params.initial_feature_standardization,
        dim_factor_for_directed=dim_factor_for_directed,
        return_us=params.return_us,
        use_rsvd=params.use_rsvd,
        sv_thresholding=params.sv_thresholding,
        theta=params.theta,
        decomposed_layers=params.decomposed_layers,
        verbose=verbose
    )
    if verbose:
        print("Building adjacency matrices and initial embeddings...")

    adj_undir, adj_ws2t_t, adj_wt2s, initial_features = preproc.create_rw_normalized_adj_matrices(
        edge_index=edge_index,
        num_nodes=num_nodes,
        init_params=params.init_params,
        node_attributes=node_attributes,
        add_self_loops_to_sinks=add_self_loops_to_sinks,
        verbose=verbose
    )

    if verbose:
        print("Calling ACC model...")
    embeddings = model(
        initial_features.to(device),
        adj_undir=adj_undir.to(device),
        adj_ws2t_t=adj_ws2t_t.to(device) if adj_ws2t_t is not None else None,
        adj_wt2s=adj_wt2s.to(device) if adj_wt2s is not None else None,
        num_steps=params.max_steps,
    )
    if return_np and isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    return embeddings
