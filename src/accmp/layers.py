from typing import Tuple

import torch
import torch_sparse as tsp
from torch_geometric.nn import MessagePassing

import accmp.compression as comprs
from accmp.transforms import FeatureNormalization, normalize_features, normalize_instances


class MsgAggStep(MessagePassing):

    def __init__(self,
                 *,
                 use_dir_conv: bool,
                 feature_normalization: FeatureNormalization,
                 decomposed_layers: int = 1
                 ):
        self.use_dir_conv = use_dir_conv
        super().__init__(aggr="sum", decomposed_layers=decomposed_layers)
        self._feature_normalization = feature_normalization

    def forward(
            self,
            x_prop: torch.Tensor,
            adj_ws2t_t: tsp.SparseTensor,
            adj_wt2s: tsp.SparseTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._feature_normalization.before_propagate:
            x_prop = normalize_features(x_prop, self._feature_normalization)

        if self.use_dir_conv:
            new_x_src_dst = self.propagate(adj_ws2t_t, x=x_prop)
            new_x_dst_src = self.propagate(adj_wt2s, x=x_prop)
            new_x = torch.cat((new_x_src_dst, new_x_dst_src), dim=1)
        else:
            new_x = self.propagate(adj_ws2t_t, x=x_prop)

        if self._feature_normalization.before_prune:
            new_x = normalize_features(new_x, self._feature_normalization)
        return new_x

    def message_and_aggregate(self, adj_t: tsp.SparseTensor, x: torch.Tensor):
        x_new = []
        for i in range(adj_t.size(-1)):
            adj_t_ = adj_t.set_value(adj_t.storage.value()[:, i], layout='coo')
            x_new.append(
                tsp.matmul(adj_t_, x, reduce=self.aggr)
            )
        x_new = torch.cat(x_new, dim=1)
        return x_new


class ConcatMP(MessagePassing):

    def __init__(self,
                 *,
                 use_dir_conv: bool,
                 compressor: comprs.FixedCompressor,
                 feature_normalization: FeatureNormalization,
                 instance_normalization: FeatureNormalization,
                 decomposed_layers: int = 1
                 ):
        self.use_dir_conv = use_dir_conv
        super().__init__(aggr="sum", decomposed_layers=decomposed_layers)
        self._feature_normalization = feature_normalization
        self._instance_normalization = instance_normalization
        self._compressor = compressor

    def forward(
            self,
            x: torch.Tensor,
            adj_ws2t_t: tsp.SparseTensor,
            adj_wt2s: tsp.SparseTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self._instance_normalization.before_propagate:
            x = normalize_instances(x, self._instance_normalization)

        if self._feature_normalization.before_propagate:
            x = normalize_features(x, self._feature_normalization)

        if self.use_dir_conv:
            new_x_src_dst = self.propagate(adj_ws2t_t, x=x)
            new_x_dst_src = self.propagate(adj_wt2s, x=x)
            new_x = torch.cat((new_x_src_dst, new_x_dst_src), dim=1)
        else:
            new_x = self.propagate(adj_ws2t_t, x=x)

        x = torch.cat((x, new_x), dim=-1)

        if self._instance_normalization.before_prune:
            x = normalize_instances(x, self._instance_normalization)

        if self._feature_normalization.before_prune:
            x = normalize_features(x, self._feature_normalization)

        x, components = self._compressor.compress(x)
        return x, components

    def message_and_aggregate(self, adj_t: tsp.SparseTensor, x: torch.Tensor):
        x_new = []
        for i in range(adj_t.size(-1)):
            adj_t_ = adj_t.set_value(adj_t.storage.value()[:, i], layout='coo')
            x_new.append(
                tsp.matmul(adj_t_, x, reduce=self.aggr)
            )
        x_new = torch.cat(x_new, dim=1)
        return x_new
