#!/usr/bin/env python3

"""
The main evaluation function.
"""

from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from pepper.core.utils.distance import build_dist

from .rank import evaluate_rank


def evaluate(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    metric: str,  # euclidean, cosine, jaccard
    ranks: List[int] = [1, 5, 10],
    use_aqe: bool = False,
    qe_times: int = 1,
    qe_k: int = 5,
    alpha: float = 3.0,
    rerank: bool = False,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
    use_roc: bool = False,
):

    results = OrderedDict()

    if use_aqe:
        from .query_expansion import aqe

        q_feat, g_feat = aqe(
            query_feat=q_feat,
            gallery_feat=g_feat,
            qe_times=qe_times,
            qe_k=qe_k,
            alpha=alpha,
        )

    dist = build_dist(
        q_feat,
        g_feat,
        metric=metric,
    )

    if rerank:
        if metric == "cosine":
            q_feat = F.normalize(q_feat, dim=1)
            g_feat = F.normalize(g_feat, dim=1)
        rerank_dist = build_dist(
            q_feat,
            g_feat,
            metric="jaccard",
            k1=k1,
            k2=k2,
        )
        dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

    cmc, all_AP, all_INP = evaluate_rank(
        dist,
        q_pids=q_pids,
        g_pids=g_pids,
        q_camids=q_camids,
        g_camids=g_camids,
        use_metric_cuhk03=False,
        use_cython=True,
    )

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    for r in ranks:
        results["Rank-{}".format(r)] = cmc[r - 1]  # * 100

    results["CMC"] = cmc
    results["mAP"] = mAP  # * 100
    results["mINP"] = mINP  # * 100
    results["metric"] = (mAP + cmc[0]) / 2  # * 100

    if use_roc:
        from .roc import evaluate_roc

        scores, labels = evaluate_roc(
            dist,
            q_pids=q_pids,
            g_pids=g_pids,
            q_camids=q_camids,
            g_camids=g_camids,
            use_cython=True,
        )
        fprs, tprs, thres = metrics.roc_curve(labels, scores)

        for fpr in [1e-4, 1e-3, 1e-2]:
            ind = np.argmin(np.abs(fprs - fpr))
            results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

    return results
