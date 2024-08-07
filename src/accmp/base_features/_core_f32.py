import numba as nb
import numpy as np

from accmp.base_features._core import get_neigh, get_data, USE_CACHE


@nb.jit(nb.float32[:](nb.float32[:, :]), nopython=True, nogil=True, parallel=True, cache=USE_CACHE)
def nb_max(array):
    out = np.empty(array.shape[0], dtype=np.float32)
    for i in nb.prange(array.shape[0]):
        out[i] = np.max(array[i])

    return out


@nb.jit(nb.types.Array(nb.types.float32, 1, 'C', readonly=True)(
    nb.int64,
    nb.types.Array(nb.types.float32, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True)
), nopython=True, nogil=True, cache=USE_CACHE)
def get_weights(node: int, weights: np.ndarray, indptr: np.ndarray):
    return weights[indptr[node]:indptr[node + 1]]


@nb.jit(nb.types.Tuple((
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[:, ::1],
        nb.float32[:, ::1],
        nb.float32[:, ::1],
        nb.float32[:, ::1]
))(
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.float32, 2, 'C', readonly=True),
), nopython=True, nogil=True, parallel=True, cache=USE_CACHE)
def _count_number_triangles_directed_weighted(
        out_indices: np.ndarray, out_indptr: np.ndarray,
        in_indices: np.ndarray, in_indptr: np.ndarray,
        out_edge_index: np.ndarray, in_edge_index: np.ndarray,
        weights: np.ndarray
):
    """ Count the number of out, in, cycle and middleman triangles and weighted versions based on the definitions in
    https://arxiv.org/pdf/physics/0612169.pdf.
    """

    assert len(out_indptr) == len(in_indptr)
    assert len(out_edge_index) == len(out_indices)
    assert len(in_edge_index) == len(in_indices)
    assert np.max(out_edge_index) < weights.shape[0]
    assert np.max(in_edge_index) < weights.shape[0]
    num_nodes = len(out_indptr) - 1
    num_weights = weights.shape[1]

    num_out_triangles = np.zeros((num_nodes,), dtype=np.float32)
    num_in_triangles = np.zeros((num_nodes,), dtype=np.float32)
    num_cycle_triangles = np.zeros((num_nodes,), dtype=np.float32)
    num_middle_triangles = np.zeros((num_nodes,), dtype=np.float32)
    weight_out_triangles = np.zeros((num_nodes, num_weights), dtype=np.float32)
    weight_in_triangles = np.zeros((num_nodes, num_weights), dtype=np.float32)
    weight_cycle_triangles = np.zeros((num_nodes, num_weights), dtype=np.float32)
    weight_middle_triangles = np.zeros((num_nodes, num_weights), dtype=np.float32)
    for v in nb.prange(num_nodes):
        out_egonet_w = {out_neig: weights[e_ij] for out_neig, e_ij in zip(get_neigh(v, out_indices, out_indptr),
                                                                          get_data(v, out_edge_index, out_indptr))}
        in_egonet_w = {in_neig: weights[e_ji] for in_neig, e_ji in zip(get_neigh(v, in_indices, in_indptr),
                                                                       get_data(v, in_edge_index, in_indptr))}
        for out_neigh, w_ij in out_egonet_w.items():
            for out_out_neigh, e_jk in zip(get_neigh(out_neigh, out_indices, out_indptr),
                                           get_data(out_neigh, out_edge_index, out_indptr)):
                w_jk = weights[e_jk]
                if out_out_neigh == v:
                    continue
                if out_out_neigh in out_egonet_w:
                    num_out_triangles[v] += 1.
                    w_ik = out_egonet_w[out_out_neigh]
                    dw = np.power(w_ij * w_jk * w_ik, 1. / 3.)
                    weight_out_triangles[v] += dw
                if out_out_neigh in in_egonet_w:
                    num_cycle_triangles[v] += 1.
                    w_ki = in_egonet_w[out_out_neigh]
                    dw = np.power(w_ij * w_jk * w_ki, 1. / 3.)
                    weight_cycle_triangles[v] += dw
        for in_neigh, w_ji in in_egonet_w.items():
            for out_in_neigh, e_jk in zip(get_neigh(in_neigh, out_indices, out_indptr),
                                          get_data(in_neigh, out_edge_index, out_indptr)):
                w_jk = weights[e_jk]
                if out_in_neigh == v:
                    continue
                if out_in_neigh in out_egonet_w:
                    num_middle_triangles[v] += 1.
                    w_ik = out_egonet_w[out_in_neigh]
                    dw = np.power(w_ji * w_jk * w_ik, 1. / 3.)
                    weight_middle_triangles[v] += dw
                if out_in_neigh in in_egonet_w:
                    num_in_triangles[v] += 1
                    w_ki = in_egonet_w[out_in_neigh]
                    dw = np.power(w_ji * w_jk * w_ki, 1. / 3.)
                    weight_in_triangles[v] += dw
    return (
        num_out_triangles,
        num_in_triangles,
        num_cycle_triangles,
        num_middle_triangles,
        weight_out_triangles,
        weight_in_triangles,
        weight_cycle_triangles,
        weight_middle_triangles
    )


@nb.jit(nb.float32[:, :](
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.float32, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.float32, 1, 'C', readonly=True)
), nopython=True, nogil=True, parallel=True, cache=USE_CACHE)
def _extract_egonet_features(
        out_indices: np.ndarray, out_indptr: np.ndarray, out_weights: np.ndarray,
        in_indices: np.ndarray, in_indptr: np.ndarray, in_weights: np.ndarray,
):
    num_nodes = len(out_indptr) - 1

    features = np.zeros((num_nodes, 6), dtype=np.float32)

    for v in nb.prange(num_nodes):
        egonet = set()
        egonet.add(v)
        egonet.update(get_neigh(v, out_indices, out_indptr))
        egonet.update(get_neigh(v, in_indices, in_indptr))
        num_internal_edges = 0
        num_out_edges = 0
        num_in_edges = 0

        internal_weight_sum = 0
        out_weight_sum = 0
        in_weight_sum = 0
        for neigh in egonet:
            for neigh_neigh, weight in zip(get_neigh(neigh, out_indices, out_indptr),
                                           get_weights(neigh, out_weights, out_indptr)):
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                    internal_weight_sum += weight
                else:
                    num_out_edges += 1
                    out_weight_sum += weight

            for neigh_neigh, weight in zip(get_neigh(neigh, in_indices, in_indptr),
                                           get_weights(neigh, in_weights, in_indptr)):
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                    internal_weight_sum += weight
                else:
                    num_in_edges += 1
                    in_weight_sum += weight

        features[v] = np.asarray([
            num_internal_edges / 2.,
            num_out_edges,
            num_in_edges,
            internal_weight_sum / 2.,
            out_weight_sum,
            in_weight_sum
        ], dtype=np.float32)
    return features


@nb.jit(nb.float32[:, :](
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
), nopython=True, nogil=True, parallel=True, cache=USE_CACHE)
def _extract_egonet_features_no_weights(
        out_indices: np.ndarray, out_indptr: np.ndarray, in_indices: np.ndarray, in_indptr: np.ndarray
):
    num_nodes = len(out_indptr) - 1

    features = np.zeros((num_nodes, 3), dtype=np.float32)

    for v in nb.prange(num_nodes):
        egonet = set()
        egonet.add(v)
        egonet.update(get_neigh(v, out_indices, out_indptr))
        egonet.update(get_neigh(v, in_indices, in_indptr))
        num_internal_edges = 0
        num_out_edges = 0
        num_in_edges = 0

        for neigh in egonet:
            for neigh_neigh in get_neigh(neigh, out_indices, out_indptr):
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                else:
                    num_out_edges += 1

            for neigh_neigh in get_neigh(neigh, in_indices, in_indptr):
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                else:
                    num_in_edges += 1

        features[v] = np.asarray([
            num_internal_edges / 2.,
            num_out_edges,
            num_in_edges
        ], dtype=np.float32)
    return features
