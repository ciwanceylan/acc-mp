import dataclasses as dc
import numpy as np
import torch

import accmp.models as aggcap_models
import accmp.preprocessing as preproc
import accmp.compression as comprs
from accmp.transforms import FeatureNormalization
from accmp.preprocessing import InitFeaturesWeightsParams


@dc.dataclass(frozen=True)
class PCAPassParams:
    max_steps: int
    max_dim: int
    initial_feature_standardization: FeatureNormalization
    mp_feature_normalization: FeatureNormalization
    mp_instance_normalization: FeatureNormalization
    init_params: InitFeaturesWeightsParams
    return_us: bool
    sv_prune: comprs.SV_THRESHOLDING = 'none'
    sv_tol: float = 0.0
    return_us: bool = False
    use_rsvd: bool = False
    normalized_weights: bool = True
    decomposed_layers: int = 1


def pcapass_embeddings(edge_index: np.ndarray, num_nodes: int, directed_conv: bool, params: PCAPassParams,
                       device: torch.device, weights: np.ndarray = None, node_attributes: np.ndarray = None,
                       return_np: bool = True):
    model = aggcap_models.PCAPass(
        use_dir_conv=directed_conv,
        max_dim=params.max_dim,
        initial_feature_standardization=params.initial_feature_standardization,
        mp_feature_normalization=params.mp_feature_normalization,
        mp_instance_normalization=params.mp_instance_normalization,
        return_us=params.return_us,
        sv_prune=params.sv_prune,
        sv_tol=params.sv_tol,
        use_rsvd=params.use_rsvd,
        decomposed_layers=params.decomposed_layers,
    )

    adj_ws2t_t, adj_wt2s, initial_features, _, _ = preproc.create_adj_t_weights_and_initial_states(
        edge_index=edge_index,
        num_nodes=num_nodes,
        init_params=params.init_params,
        normalized_weights=params.normalized_weights,
        weights=weights,
        node_attributes=node_attributes
    )
    embeddings = model(
        initial_features.to(device),
        adj_ws2t_t=adj_ws2t_t.to(device),
        adj_wt2s=adj_wt2s.to(device) if directed_conv else None,
        num_steps=params.max_steps,
    )
    if return_np and isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    return embeddings
