# Aggregate, compress, and concatenate: Fast node embeddings for directed graphs via message-passing

Clone the repo and then install into your environment using

```bash
pip install ./acc-mp
```

### These packages are required
```commandline
numpy
scipy
pandas
sklearn
numba
torch
torch-geometric
torch-sparse
```

### Optional requirements
To run PCAPass on GPU with rSVD, please install `torch_sprsvd`.
There optional packages can be installed with conda to speed up Numba.
```
icc_rt -c numba
tbb -c conda-forge
```

To run tests, install `pytest` and `networkx`.

Usage example

```python
import accmp
```
