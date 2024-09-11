# Aggregate, compress, and concatenate: Fast node embeddings for directed graphs via message-passing

This is code for the ACC node embedding model, described in the paper "Full-Rank Unsupervised Node Embeddings for Directed
Graphs via Message Aggregation" (under review for TMLR 2024).

## Installation

Clone the repo and then install ot into your environment using pip:
```bash
pip install -e ./acc-mp
```

### Requirements
```commandline
numpy
scipy
pandas
sklearn
numba
torch
torch_geometric
torch_sparse
```

#### Optional requirements
To run PCAPass on GPU with rSVD, please install `torch_sprsvd` (Currently under development).
There optional packages can be installed with conda to speed up Numba.
```
conda install icc_rt -c numba
conda install tbb -c conda-forge
```

To run tests, install `pytest` and `networkx`.

## Usage example 
Here we show how to call ACC for compute embeddings for graph alignment on the Arenas graph.

```python
import numpy as np
import torch
from sklearn.decomposition import PCA
from tests.alignment import load_alignment_problem, eval_topk_sim
import accmp.transforms
import accmp.preprocessing as preproc
import accmp.acc as acc

merged_graph, alignment_obj = load_alignment_problem("arenas", noise_p=0.15, seed=1235233413)
params = acc.ACCParams(
    max_steps=8,
    max_dim=64,
    initial_feature_standardization=accmp.transforms.FeatureNormalization(mode='std', subtract_mean=True),
    mp_feature_normalization=accmp.transforms.FeatureNormalization(mode=None, subtract_mean=False),
    init_params=preproc.InitFeaturesWeightsParams(use_weights=False, use_node_attributes=False,
        as_undirected=True, use_degree=True, use_lcc=True, use_log1p_degree=False, dtype=np.float32),
)

embeddings = acc.agg_compress_cat_embeddings(
    edge_index=merged_graph.edges.T,  # Required shape is [2, num_edges]
    num_nodes=merged_graph.num_nodes,
    directed_conv=False,
    params=params,
    device=torch.device('cpu'),
    weights=None,
    node_attributes=None
)
embeddings = PCA(whiten=True).fit_transform(embeddings)

res = eval_topk_sim(embeddings, alignment_obj)
```
