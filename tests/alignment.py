import os
from typing import Mapping
import dataclasses as dc
import numpy as np
import pandas as pd
import scipy.sparse as sp

from numpy.random import default_rng

from sklearn.neighbors import KDTree

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


@dc.dataclass(frozen=True)
class SimpleGraph:
    num_nodes: int
    edges: np.ndarray
    weights: np.ndarray = None

    @classmethod
    def union(cls, g1: 'SimpleGraph', g2: 'SimpleGraph'):
        g2_to_merged = AlignedGraphs.g2_to_merged(g1.num_nodes, g2.num_nodes)
        # new_sources = g2_to_merged[g2.edges[:, 0]]
        # new_targets = g2_to_merged[g2.edges[:, 1]]
        new_edges = np.stack((g2_to_merged[g2.edges[:, 0]], g2_to_merged[g2.edges[:, 1]]), axis=1)
        edges = np.concatenate((g1.edges, new_edges), axis=0)
        weights = None
        if g1.weights is not None and g2.weights is not None:
            weights = np.concatenate((g1.weights, g2.weights), axis=0)
        return cls(g1.num_nodes + g2.num_nodes, edges, weights)

    @classmethod
    def remove_edges_noise(cls, g: 'SimpleGraph', p: float, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        num_existing_edges = g.edges.shape[0]
        num_noise_edges = int(p * num_existing_edges)
        edges_to_remove = rng.choice(num_existing_edges, size=num_noise_edges, replace=False)
        mask = np.ones((num_existing_edges,), dtype=bool)
        mask[edges_to_remove] = 0
        edges = g.edges[mask, :]
        weights = g.weights[mask] if g.weights is not None else None
        actual_p = edges_to_remove.shape[0] / num_existing_edges
        return cls(g.num_nodes, edges, weights), actual_p

    @classmethod
    def read_file(cls, path: str, is_weighted: bool, directed: bool, remove_self_loops: bool = True,
                  num_nodes=None, **pd_kwargs):
        filetype = path.split('.')[-1]
        num_nodes_ = None
        comment_char = None
        if filetype in {"csv", "tsv", "edgelist"}:
            num_nodes_, comment_char = try_read_num_nodes(path)
        if comment_char is not None:
            pd_kwargs["comment"] = comment_char
        num_nodes = num_nodes_ if num_nodes_ is not None else num_nodes
        edges, weights = read_edges(path, filetype, is_weighted=is_weighted, is_undirected=not directed,
                                    remove_self_loops=remove_self_loops, **pd_kwargs)
        if num_nodes is None:
            num_nodes = np.max(edges) + 1
        return cls(num_nodes=num_nodes, edges=edges, weights=weights)

    def to_csc(self, directed: bool):
        return edges2spmat(self.edges, self.weights, num_nodes=self.num_nodes, directed=directed)


class AlignedGraphs:
    g1_num_nodes: int
    g2_num_nodes: int
    # g2_to_merged: pd.Series
    # merged_to_g2: pd.Series
    g1_to_g2: pd.Series
    g2_to_g1: pd.Series

    def __init__(self, g1_num_nodes, g2_num_nodes, g2_to_g1):
        assert g1_num_nodes >= g2_num_nodes
        assert len(g2_to_g1) == g2_num_nodes
        # assert len(perm_backward) == g2_num_nodes

        self.g1_num_nodes = g1_num_nodes
        self.g2_num_nodes = g2_num_nodes

        self.g2_to_g1 = g2_to_g1
        self.g1_to_g2 = pd.Series(g2_to_g1.index, index=g2_to_g1)
        # self.g2_to_merged = pd.Series(np.arange(g2_num_nodes, dtype=np.int64) + self.g1_num_nodes)
        # self.merged_to_g2 = pd.Series(range(g2_num_nodes), index=self.g2_to_merged)

    @staticmethod
    def g2_to_merged(g1_num_nodes: int, g2_num_nodes: int):
        return pd.Series(np.arange(g2_num_nodes, dtype=np.int64) + g1_num_nodes)

    @staticmethod
    def merged_to_g2(g1_num_nodes: int, g2_num_nodes: int):
        return pd.Series(np.arange(g2_num_nodes, dtype=np.int64),
                         index=np.arange(g2_num_nodes, dtype=np.int64) + g1_num_nodes)

    @classmethod
    def load_from_file(cls, path: str):
        with open(path, "r") as fp:
            first_line = fp.readline()
        g1_num_nodes, g2_num_nodes = first_line.strip("%\n ").split("::")
        g1_num_nodes = int(g1_num_nodes)
        g2_num_nodes = int(g2_num_nodes)
        g2_to_g1 = pd.read_csv(path, index_col=0, names=["g2_to_g1"], header=None, comment="%")["g2_to_g1"]
        return cls(g1_num_nodes, g2_num_nodes, g2_to_g1)

    def save2file(self, fp):
        fp.write("%" + str(self.g1_num_nodes) + "::" + str(self.g2_num_nodes) + "\n")
        pd.Series(self.g2_to_g1).to_csv(fp, index=True, header=False)


def try_read_num_nodes(filename):
    num_nodes = None
    comment_char = None
    with open(filename) as f:
        first_line = f.readline()
    if first_line[0] in "#%":
        comment_char = first_line[0]
        try:
            num_nodes = int(first_line.strip(comment_char + "\n "))
        except ValueError:
            pass
    return num_nodes, comment_char


def read_edges(filename, filetype, is_weighted, is_undirected, remove_self_loops=True, **pd_kwargs):
    if "index_col" not in pd_kwargs:
        pd_kwargs["index_col"] = False

    if "header" not in pd_kwargs:
        pd_kwargs["header"] = None

    if "comment" not in pd_kwargs:
        pd_kwargs["comment"] = '%'

    if filetype in {'csv', 'tsv', 'edgelist', 'txt'}:
        if filetype == "csv":
            pd_kwargs["sep"] = ","
        elif filetype == "tsv" or filetype == "edgelist" or filetype == "twitter_txt":
            pd_kwargs["sep"] = r"\s+"

        edges, weights = _read_tex_to_edges(filename, is_weighted=is_weighted, as_undirected=is_undirected,
                                            remove_self_loops=remove_self_loops, **pd_kwargs)
    else:
        raise NotImplementedError
    return edges, weights


def edges2spmat(edges, weights, num_nodes, directed):
    mat = sp.coo_matrix((weights, (edges[:, 1], edges[:, 0])), shape=[num_nodes, num_nodes]).tocsc()

    if not directed:
        mat = mat.maximum(mat.T)

    return mat


def _read_tex_to_edges(filename, is_weighted, as_undirected, remove_self_loops=True, **pd_kwargs):
    df = pd.read_csv(filename, **pd_kwargs)
    return _post_load(df, is_weighted=is_weighted, as_undirected=as_undirected, remove_self_loops=remove_self_loops)


def _post_load(df, is_weighted, as_undirected, remove_self_loops=True):
    if remove_self_loops:
        df = df.loc[df.iloc[:, 0] != df.iloc[:, 1], :]
    edges = df.iloc[:, [0, 1]].to_numpy().astype(np.int64)
    if is_weighted and df.shape[1] > 2:
        weights = df.iloc[:, 2].to_numpy().astype(np.float64)
    else:
        weights = np.ones(edges.shape[0], dtype=np.float64)
    if as_undirected:
        edges = np.sort(edges, axis=1)
        df = pd.DataFrame({'source': edges[:, 0], 'target': edges[:, 1], 'weight': weights})
        df = df.groupby(['source', 'target'], as_index=False).agg('mean')
        edges = df.loc[:, ['source', 'target']].to_numpy().astype(np.int64)
        weights = df.loc[:, 'weight'].to_numpy().astype(np.float64)
    return edges, weights


def create_permuted(g: SimpleGraph):
    rng = default_rng()
    g2_to_g1 = pd.Series(rng.permutation(g.num_nodes))
    g1_to_g2 = pd.Series(g2_to_g1.index, index=g2_to_g1)
    new_sources = g1_to_g2[g.edges[:, 0]]
    new_targets = g1_to_g2[g.edges[:, 1]]
    g2_edges = np.stack((new_sources, new_targets), axis=1)
    new_g = SimpleGraph(num_nodes=g.num_nodes, edges=g2_edges, weights=g.weights)
    return new_g, AlignedGraphs(g.num_nodes, g.num_nodes, g2_to_g1)


def split_embeddings(embeddings: np.ndarray, g2_to_merged: pd.Series):
    g1_num_nodes = embeddings.shape[0] - len(g2_to_merged)
    g1_embeddings = embeddings[:g1_num_nodes, :]
    g2_merged = g2_to_merged[np.arange(len(g2_to_merged), dtype=np.int64)]
    g2_embeddings = embeddings[g2_merged, :]
    return g1_embeddings, g2_embeddings


def calc_topk_similarties(g1_embeddings, g2_embeddings, alpha=10):
    kd_tree = KDTree(g1_embeddings, metric="euclidean")

    dist, ind = kd_tree.query(g2_embeddings, k=alpha)
    similarity = np.exp(-dist)
    return similarity, ind


def get_top_sim(embeddings: np.ndarray, g2_to_merged: pd.Series, alpha=50):
    num_g2_nodes = len(g2_to_merged)
    g1_embeddings, g2_embeddings = split_embeddings(embeddings, g2_to_merged)
    similarity, ind = calc_topk_similarties(g1_embeddings, g2_embeddings, alpha=alpha)
    assert similarity.shape[0] == num_g2_nodes and ind.shape[0] == num_g2_nodes

    res = dict()
    for i in range(num_g2_nodes):
        res[i] = [(s, node) for s, node in zip(similarity[i, :], ind[i, :])]

    return res


def calc_topk_acc_score(y_true: Mapping, top_sim: dict):
    topk_vals = np.asarray([1, 5, 10])
    scores = np.zeros(len(topk_vals), dtype=np.float64)

    for g2_node, y in y_true.items():
        vals = sorted(top_sim[g2_node], key=lambda x: -x[0])
        the_k = np.inf
        for k_, (sim, val) in enumerate(vals):
            if y == val:
                the_k = k_
        scores[the_k < topk_vals] += 1.
    scores = scores / len(y_true)
    return dict(zip(topk_vals, scores))


def eval_topk_sim(embeddings: np.ndarray, alignment: AlignedGraphs):
    top_sim = get_top_sim(embeddings, alignment.g2_to_merged(alignment.g1_num_nodes, alignment.g2_num_nodes))
    res = calc_topk_acc_score(alignment.g2_to_g1, top_sim)
    return res


def load_graph(dataset_name: str):
    if dataset_name.lower() == "arenas":
        graph = SimpleGraph.read_file(os.path.join(ROOT_DIR, "datasets/arenas_clean.edgelist"), is_weighted=False,
                                      directed=False)
    elif dataset_name.lower() == 'usairport':
        graph = SimpleGraph.read_file(os.path.join(ROOT_DIR, "datasets/usairport.edgelist"), is_weighted=True,
                                      directed=True)
    else:
        raise NotImplementedError(f"Cannot load dataset {dataset_name}")
    return graph


def load_alignment_problem(dataset_name: str, noise_p: float, seed: int):
    rng = np.random.default_rng(seed=seed)
    graph = load_graph(dataset_name)
    permuted_graph, alignment_obj = create_permuted(graph)
    merged_graph = merge_and_remove_edges_noise(graph, permuted_graph, noise_p=noise_p, rng=rng)
    return merged_graph, alignment_obj


def merge_and_remove_edges_noise(g1: SimpleGraph, g2: SimpleGraph, noise_p: float, rng=None):
    if noise_p > 0.:
        g1, actual_p = g1.remove_edges_noise(g1, noise_p, rng=rng)
    merged_g = SimpleGraph.union(g1, g2)
    return merged_g
