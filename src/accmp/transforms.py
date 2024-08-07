import dataclasses as dc
from typing import Optional, Literal

import torch


@dc.dataclass(frozen=True)
class FeatureNormalization:
    mode: Optional[Literal['sphere', 'std']]
    subtract_mean: bool
    before_propagate: bool = False
    before_prune: bool = True


def normalize_features(x: torch.Tensor, feature_normalization: FeatureNormalization):
    if feature_normalization.subtract_mean:
        x = x - torch.mean(x, dim=-2, keepdim=True)

    if feature_normalization.mode == 'sphere':
        norm = torch.linalg.norm(x, dim=-2, keepdim=True)
        norm = torch.maximum(norm, torch.tensor(1e-12, dtype=x.dtype))
        x = x / norm
    elif feature_normalization.mode == 'std':
        std = torch.std(x, dim=-2, keepdim=True)
        # change 0 std to 1 to resolve NaN issue (same approach as in sklearn)
        std[std == 0] = 1
        x = x / std
    elif feature_normalization.mode is None:
        pass
    else:
        raise NotImplementedError
    return x


def normalize_instances(x: torch.Tensor, feature_normalization: FeatureNormalization):
    if feature_normalization.subtract_mean:
        x = x - torch.mean(x, dim=-1, keepdim=True)

    if feature_normalization.mode == 'sphere':
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        norm = torch.maximum(norm, torch.tensor(1e-12, dtype=x.dtype))
        x = x / norm
    elif feature_normalization.mode == 'std':
        std = torch.std(x, dim=-1, keepdim=True)
        # change 0 std to 1 to resolve NaN issue (same approach as in sklearn)
        std[std == 0] = 1
        x = x / std
    elif feature_normalization.mode is None:
        pass
    else:
        raise NotImplementedError
    return x
