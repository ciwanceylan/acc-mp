from typing import Union
import numpy as np
import networkx as nx


def get_nx_degrees_as_array(graph: Union[nx.Graph, nx.DiGraph], directed: bool, weight: str = None,
                            dtype: type = np.float64):
    if directed:
        nx_degs = np.asarray([
            (graph.out_degree(nbunch=i, weight=weight),
             graph.in_degree(nbunch=i, weight=weight))
            for i in range(graph.number_of_nodes())
        ], dtype=dtype)
    else:
        nx_degs = np.asarray([
            (graph.degree(nbunch=i, weight=weight),) for i in range(graph.number_of_nodes())
        ], dtype=dtype)
    return nx_degs


def get_nx_lcc_as_array(graph: Union[nx.Graph, nx.DiGraph], weight: str = None, dtype: type = np.float64):
    nx_lcc = nx.clustering(graph, weight=weight)
    nx_lcc = np.asarray([nx_lcc[i] for i in range(graph.number_of_nodes())], dtype=dtype)
    return nx_lcc


def add_weights_and_edge_index_to_nx_graph(graph: nx.Graph, weights: np.ndarray):
    for e_idx, e in enumerate(graph.edges):
        graph[e[0]][e[1]]['edge_index'] = e_idx
        for col in range(weights.shape[1]):
            graph[e[0]][e[1]][f'weight{col}'] = weights[e_idx, col]
    return graph
