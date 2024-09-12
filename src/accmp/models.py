from typing import Optional
import torch
import torch_sparse as tsp

import accmp.compression as feat_prune
from accmp.layers import ConcatMP, MsgAggStep
from accmp.transforms import FeatureNormalization, normalize_features


def vprint(s: str, verbose: bool):
    if verbose:
        print(s)


class ACC(torch.nn.Module):
    agg_step: MsgAggStep

    def __init__(
            self,
            *,
            use_dir_conv: bool,
            max_dim: int,
            min_add_dim: int,
            initial_feature_standardization: FeatureNormalization,
            mp_feature_normalization: FeatureNormalization,
            return_us: bool = False,
            use_rsvd: bool = False,
            sv_thresholding: feat_prune.SV_THRESHOLDING,
            theta: float = 0.0,
            decomposed_layers: int = 1,
            verbose: bool = False
    ):
        super().__init__()
        self.agg_step = MsgAggStep(
            use_dir_conv=use_dir_conv,
            feature_normalization=mp_feature_normalization,
            decomposed_layers=decomposed_layers
        )
        self.feat_norm = mp_feature_normalization
        self.base_transform = initial_feature_standardization
        self.compressor = feat_prune.ACCCompressor(return_us=return_us, sv_thresholding=sv_thresholding, theta=theta,
                                                   use_rsvd=use_rsvd)
        self.max_dim = max_dim
        self.min_add_dim = min_add_dim
        self.verbose = verbose

    def get_emb_dims(self, num_nodes: int, num_steps: int):
        dim_per_step = min(max(self.max_dim // (num_steps + 1), self.min_add_dim), num_nodes)
        dims = num_steps * [dim_per_step]
        return dim_per_step, dims

    def forward(
            self,
            x: torch.Tensor,
            adj_ws2t_t: tsp.SparseTensor,
            adj_wt2s: Optional[tsp.SparseTensor],
            num_steps: int,
    ):
        assert isinstance(adj_ws2t_t, tsp.SparseTensor)  # To avoid silent errors
        assert adj_wt2s is None or isinstance(adj_wt2s, tsp.SparseTensor)  # To avoid silent errors

        init_dim, emb_dims = self.get_emb_dims(num_nodes=x.shape[0], num_steps=num_steps)

        vprint("Applying base transforms...", self.verbose)
        x = normalize_features(x, self.base_transform)

        vprint("Applying SVD to initial features...", self.verbose)
        x, _ = self.compressor.compress(x, k=init_dim)
        x_new = x
        embeddings = [x]

        vprint("Starting message-passing...", self.verbose)
        for step, k in enumerate(emb_dims):
            vprint(f"Aggegation {step}...", self.verbose)
            x_new = self.agg_step(
                x_prop=x_new,
                adj_ws2t_t=adj_ws2t_t,
                adj_wt2s=adj_wt2s,
            )
            vprint(f"SVD {step}...", self.verbose)
            x_new, components = self.compressor.compress(x_new, k=k)
            embeddings.append(x_new)

        vprint(f"Embedding computation done!", self.verbose)
        embeddings = torch.cat(embeddings, dim=1)
        return embeddings


class PCAPass(torch.nn.Module):
    acp_step: ConcatMP

    def __init__(
            self,
            *,
            use_dir_conv: bool,
            max_dim: int,
            initial_feature_standardization: FeatureNormalization,
            mp_feature_normalization: FeatureNormalization,
            mp_instance_normalization: FeatureNormalization,
            sv_thresholding: feat_prune.SV_THRESHOLDING,
            theta: float = 0.0,
            return_us: bool = False,
            use_rsvd: bool = False,
            decomposed_layers: int = 1
    ):
        super().__init__()

        self.acp_step = ConcatMP(
            use_dir_conv=use_dir_conv,
            compressor=feat_prune.PCAPassCompressor(sv_thresholding=sv_thresholding, theta=theta, max_dim=max_dim,
                                                    return_us=return_us, use_rsvd=use_rsvd),
            feature_normalization=mp_feature_normalization,
            instance_normalization=mp_instance_normalization,
            decomposed_layers=decomposed_layers
        )
        self.base_transform = initial_feature_standardization

    def forward(
            self,
            x: torch.Tensor,
            adj_ws2t_t: tsp.SparseTensor,
            adj_wt2s: Optional[tsp.SparseTensor],
            num_steps: int,
    ):
        assert isinstance(adj_ws2t_t, tsp.SparseTensor)  # To avoid silent errors
        assert adj_wt2s is None or isinstance(adj_wt2s, tsp.SparseTensor)  # To avoid silent errors
        x = normalize_features(x, self.base_transform)

        for step in range(num_steps):
            x, components = self.acp_step(x=x, adj_ws2t_t=adj_ws2t_t, adj_wt2s=adj_wt2s)

        return x
