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

try:
    import qmrfs.qmr_feature_selection as qmrfs

    QMR_AVAILABLE = True
except ImportError:
    warnings.warn(f"Package `qmrfs` could not be imported, so QMR feature selection is not available.")
    QMR_AVAILABLE = False

SV_THRESHOLDING = Literal['none', 'tol', 'rtol', 'rank', 'stwhp']


class FixedCompressor(abc.ABC):

    @abc.abstractmethod
    def compress(self, features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass


class VariableCompressor(abc.ABC):

    @abc.abstractmethod
    def compress(self, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

class IncrementalCompressor(abc.ABC):

    @abc.abstractmethod
    def compress(self, features: torch.Tensor, num_new_dims: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass


def _get_sv_threshold(sv_thresholding: SV_THRESHOLDING, tol: float,
                      singular_values: torch.Tensor, features: torch.Tensor):
    if sv_thresholding == 'rank':
        sv_max = singular_values.max()
        sv_threshold = sv_max * max(*features.shape) * torch.finfo(features.dtype).eps
    elif sv_thresholding == 'rtol':
        sv_max = singular_values.max()
        sv_threshold = tol * sv_max
    elif sv_thresholding == 'tol':
        sv_threshold = tol
    elif sv_thresholding == 'stwhp':
        sv_max = singular_values.max()
        sv_threshold = sv_max * 0.5 * np.sqrt(features.shape[0] + features.shape[1] + 1.) * torch.finfo(
            features.dtype).eps
    elif sv_thresholding == 'none':
        sv_threshold = -1.
    else:
        raise NotImplementedError(f"Singular value threshold '{sv_thresholding}' not implemented.")
    return sv_threshold


class ACCCompressor(VariableCompressor):

    def __init__(self, sv_thresholding: SV_THRESHOLDING = 'none', theta: float = 0.0,
                 return_us: bool = False, use_rsvd: bool = False):
        self.return_us = return_us
        self.sv_thresholding = sv_thresholding
        self.tol = theta
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
        sv_threshold = _get_sv_threshold(sv_thresholding=self.sv_thresholding, tol=self.tol,
                                         singular_values=singular_values, features=features)
        rank = (singular_values >= sv_threshold).sum().item()
        k = max(min(rank, k), 1)
        V = Vh.t()
        out = features @ V[:, :k]
        return out, Vh


def qmr_fs_torch(features: torch.Tensor, tolerance: float):
    selected_columns = torch.arange(features.shape[1], dtype=torch.int64, device=features.device)
    with_variance = features.std(dim=0) > 0
    features = features[:, with_variance]
    selected_columns = selected_columns[with_variance]

    selector = qmrfs.QMRFeatureSelector(tolerance=tolerance)
    selected_features, columns_to_keep_mask = selector.fit_transform(features)
    selected_columns = selected_columns[columns_to_keep_mask]
    return selected_features, selected_columns


class ACCQMRCompressor(IncrementalCompressor):

    def __init__(self, theta: float = 0.0):
        self.tol = theta
        if not QMR_AVAILABLE:
            raise ImportError("Could not import package `qmrfs` required to use QMR feature selection.")

    def compress(self, new_features: torch.Tensor, old_features: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_new_dims = new_features.shape[1]
        if old_features is None:
            new_selected_features, selected_columns = qmr_fs_torch(new_features, tolerance=self.tol)
        else:
            num_old_dims = old_features.shape[1]
            combined = torch.concatenate((old_features, new_features), dim=1)
            selected_features, selected_columns = qmr_fs_torch(combined, tolerance=self.tol)
            selected_columns = selected_columns[selected_columns > num_old_dims]
            new_selected_features = combined[:, selected_columns]
            selected_columns = selected_columns - num_old_dims  # selected_columns should be indexed for the new_features input
            # assert torch.allclose(new_selected_features, new_features[:, selected_columns])  # Sainity check
        return new_selected_features, selected_columns


class PCAPassCompressor(FixedCompressor):

    def __init__(self, *, sv_thresholding: SV_THRESHOLDING, theta: float = 0., max_dim: int = 64,
                 return_us: bool = False, use_rsvd: bool = False):
        self.max_dim = max_dim
        self.sv_thresholding = sv_thresholding
        self.tol = theta
        self.return_us = return_us
        self.keep_all = sv_thresholding == 'none'
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
        sv_threshold = _get_sv_threshold(sv_thresholding=self.sv_thresholding, tol=self.tol,
                                         singular_values=singular_values, features=features)
        rank = (singular_values >= sv_threshold).sum().item()
        k = max(min(rank, self.max_dim), 1)
        V = Vh.t()
        out = features @ V[:, :k]
        return out, Vh
