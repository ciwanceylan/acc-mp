import dataclasses as dc
from typing import Union, Iterable, Tuple, Sequence, Literal, List
import itertools

from unstructnes.aggcap.feature_pruning.pruning import ReFeXPruningResult, PruningResult

AGG_TYPE = Literal['O', 'I', 'U']


@dc.dataclass(frozen=True)
class FeatureDescriptor:
    origin_feature: str
    creation_step: int = 0
    aggregations: tuple[AGG_TYPE] = ()

    def create_new_from_agg(self, agg: str):
        new_feature = FeatureDescriptor(origin_feature=self.origin_feature,
                                        aggregations=self.aggregations + (agg,),
                                        creation_step=self.creation_step + 1,
                                        )
        return new_feature

    def __str__(self):
        return f"({self.origin_feature};{self.creation_step};{'::'.join(self.aggregations)})"


@dc.dataclass(frozen=True)
class QMREmbeddingDescriptor:
    features: Tuple[FeatureDescriptor]

    @classmethod
    def create_from_origin_features(cls, origin_features: tuple[str]):
        features = tuple(FeatureDescriptor(origin_feature=of) for of in origin_features)
        return cls(features)

    def to_str_list(self):
        features = [str(feat) for feat in self.features]
        return features

    def __add__(self, other):
        return QMREmbeddingDescriptor(self.features + other.features)

    def select(self, selected_columns: List[int]):
        new_features = [self.features[i] for i in selected_columns]
        return QMREmbeddingDescriptor(new_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item: int):
        return self.features[item]

    def __repr__(self):
        s = ' | '.join(self.to_str_list())
        return s


def message_passing(features: QMREmbeddingDescriptor, agg: AGG_TYPE):
    features = tuple([feat.create_new_from_agg(agg=agg) for feat in features.features])
    return QMREmbeddingDescriptor(features)


def default_directed_msg_passing(features: QMREmbeddingDescriptor):
    featuresO = message_passing(features, agg='O')
    featuresI = message_passing(features, agg='I')
    return featuresO + featuresI

def default_undirected_msg_passing(features: QMREmbeddingDescriptor):
    featuresU = message_passing(features, agg='U')
    return featuresU


def accqmr_step(start_features: QMREmbeddingDescriptor, selected_columns, directed: bool):
    if directed:
        new_features = default_directed_msg_passing(start_features)
    else:
        new_features = default_undirected_msg_passing(start_features)
    new_features = new_features.select(selected_columns)
    return new_features
