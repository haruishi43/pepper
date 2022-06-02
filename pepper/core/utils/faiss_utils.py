#!/usr/bin/env python3

# copy from: https://github.com/open-mmlab/OpenUnReID/blob/66bb2ae0b00575b80fbe8915f4d4f4739cc21206/openunreid/core/utils/faiss_utils.py
# updated in: https://github.com/facebookresearch/faiss/blob/main/contrib/torch_utils.py

import numpy as np

import torch

import faiss
from faiss.contrib.torch_utils import (
    swig_ptr_from_FloatTensor,
    swig_ptr_from_HalfTensor,
    swig_ptr_from_IndicesTensor,
    swig_ptr_from_IntTensor,
    using_stream,
)


def search_index_pytorch(
    index,
    x,
    k,
    D=None,
    I=None,
):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    # Iptr = swig_ptr_from_LongTensor(I)
    Iptr = swig_ptr_from_IndicesTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr, k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I


def search_raw_array_pytorch(
    res,
    xq,
    xb,
    k,
    D=None,
    I=None,
    metric=faiss.METRIC_L2,
):
    """knn gpu

    Replace this method with the maintained version:
    https://github.com/facebookresearch/faiss/blob/main/contrib/torch_utils.py
    """

    # check if the arrays are torch
    if type(xb) == np.ndarray:
        # forward parameters to numpy method
        return faiss.knn_gpu(res, xq, xb, k, D, I, metric)

    nb, d = xb.size()
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError("matrix xb should be row or column-major")

    if xb.dtype == torch.float32:
        xb_type = faiss.DistanceDataType_F32
        xb_ptr = swig_ptr_from_FloatTensor(xb)
    elif xb.dtype == torch.float16:
        xb_type = faiss.DistanceDataType_F16
        xb_ptr = swig_ptr_from_HalfTensor(xb)
    else:
        raise TypeError("xb must be f32 or f16")

    nq, d2 = xq.size()
    assert d2 == d
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()
        xq_row_major = False
    else:
        raise TypeError("matrix xq should be row or column-major")

    if xq.dtype == torch.float32:
        xq_type = faiss.DistanceDataType_F32
        xq_ptr = swig_ptr_from_FloatTensor(xq)
    elif xq.dtype == torch.float16:
        xq_type = faiss.DistanceDataType_F16
        xq_ptr = swig_ptr_from_HalfTensor(xq)
    else:
        raise TypeError("xq must be f32 or f16")

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        # interface takes void*, we need to check this
        assert D.dtype == torch.float32

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)

    if I.dtype == torch.int64:
        I_type = faiss.IndicesDataType_I64
        I_ptr = swig_ptr_from_IndicesTensor(I)
    elif I.dtype == torch.int32:
        I_type = faiss.IndicesDataType_I32
        I_ptr = swig_ptr_from_IntTensor(I)
    else:
        raise TypeError("I must be i64 or i32")

    D_ptr = swig_ptr_from_FloatTensor(D)

    args = faiss.GpuDistanceParams()
    args.metric = metric
    args.k = k
    args.dims = d
    args.vectors = xb_ptr
    args.vectorsRowMajor = xb_row_major
    args.vectorType = xb_type
    args.numVectors = nb
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.queryType = xq_type
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr
    args.outIndicesType = I_type

    with using_stream(res):
        faiss.bfKnn(res, args)

    return D, I


def index_init_gpu(ngpus, feat_dim):
    flat_config = []
    for i in range(ngpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    res = [faiss.StandardGpuResources() for i in range(ngpus)]
    indexes = [
        faiss.GpuIndexFlatL2(res[i], feat_dim, flat_config[i])
        for i in range(ngpus)
    ]
    index = faiss.IndexShards(feat_dim)
    for sub_index in indexes:
        index.add_shard(sub_index)
    index.reset()
    return index


def index_init_cpu(feat_dim):
    return faiss.IndexFlatL2(feat_dim)
