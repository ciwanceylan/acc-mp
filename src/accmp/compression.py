from typing import Tuple, Optional
from typing_extensions import Literal
import warnings
import abc
import numpy as np
import torch

try:
    import torch_sprsvd.sprsvd as sprsvd

    RSVD_AVAILABLE = True
except ImportError:
    warnings.warn(f"Package `torch_sprsvd` could not be imported, so rSVD is not available.")
    RSVD_AVAILABLE = False

SV_THRESHOLDING = Literal['none', 'tol', 'rtol', 'rank', 'stwhp']


class FixedCompressor(abc.ABC):

    @abc.abstractmethod
    def compress(self, features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass


class VariableCompressor(abc.ABC):

    @abc.abstractmethod
    def compress(self, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass


def _get_sv_threshold(sv_prune: SV_THRESHOLDING, tol: float,
                      singular_values: torch.Tensor, features: torch.Tensor):
    if sv_prune == 'rank':
        sv_max = singular_values.max()
        sv_threshold = sv_max * max(*features.shape) * torch.finfo(features.dtype).eps
    elif sv_prune == 'rtol':
        sv_max = singular_values.max()
        sv_threshold = tol * sv_max
    elif sv_prune == 'tol':
        sv_threshold = tol
    elif sv_prune == 'stwhp':
        sv_max = singular_values.max()
        sv_threshold = sv_max * 0.5 * np.sqrt(features.shape[0] + features.shape[1] + 1.) * torch.finfo(
            features.dtype).eps
    elif sv_prune == 'none':
        sv_threshold = -1.
    else:
        raise NotImplementedError(f"Singular value threshold '{sv_prune}' not implemented.")
    return sv_threshold


class ACCCompressor(VariableCompressor):

    def __init__(self, sv_prune: SV_THRESHOLDING = 'none', sv_tol: float = 0.0,
                 return_us: bool = False, use_rsvd: bool = False):
        self.return_us = return_us
        self.sv_prune = sv_prune
        self.tol = sv_tol
        if use_rsvd and not RSVD_AVAILABLE:
            raise ImportError("Could not import package `torch_sprsvd` required to use rSVD.")
        self.use_rsvd = use_rsvd

    def compress(self, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_rsvd and features.shape[1] > k:
            U, singular_values, Vh = sprsvd.multi_pass_rsvd(
                input_matrix=features,
                k=k,
                num_oversampling=12,
                num_iter=6
            )
        else:
            U, singular_values, Vh = torch.linalg.svd(
                A=features,
                full_matrices=False
            )
        sv_threshold = _get_sv_threshold(sv_prune=self.sv_prune, tol=self.tol,
                                         singular_values=singular_values, features=features)
        rank = (singular_values >= sv_threshold).sum().item()
        k = max(min(rank, k), 1)
        V = Vh.t()
        out = features @ V[:, :k]
        return out, Vh


class PCAPassCompressor(FixedCompressor):

    def __init__(self, *, sv_prune: SV_THRESHOLDING, sv_tol: float = 0., max_dim: int = 64,
                 return_us: bool = False, use_rsvd: bool = False):
        self.max_dim = max_dim
        self.sv_prune = sv_prune
        self.tol = sv_tol
        self.return_us = return_us
        self.keep_all = sv_prune == 'none'
        if use_rsvd and not RSVD_AVAILABLE:
            raise ImportError("Could not import package `torch_sprsvd` required to use rSVD.")
        self.use_rsvd = use_rsvd

    def compress(self, features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_rsvd and features.shape[1] > self.max_dim:
            U, singular_values, Vh = sprsvd.multi_pass_rsvd(
                input_matrix=features,
                k=self.max_dim,
                num_oversampling=12,
                num_iter=6
            )
        else:
            U, singular_values, Vh = torch.linalg.svd(
                A=features,
                full_matrices=False
            )
        sv_threshold = _get_sv_threshold(sv_prune=self.sv_prune, tol=self.tol,
                                         singular_values=singular_values, features=features)
        rank = (singular_values >= sv_threshold).sum().item()
        k = max(min(rank, self.max_dim), 1)
        V = Vh.t()
        out = features @ V[:, :k]
        return out, Vh
