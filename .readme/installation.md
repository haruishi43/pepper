# Installation

## Compile with cython to accelerate evaluation

```Bash
cd pepper/core/evaluation/rank_cylib; make all
```

For now, numpy will raise warnings about not using deprecated API, but it will be resolved in `cython>=0.30`.


## Install faiss

It is recommended to install the GPU accelerated version:

```Bash
pip install faiss-gpu
```
