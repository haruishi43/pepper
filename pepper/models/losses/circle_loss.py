#!/usr/bin/env python3

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def pairwise_circleloss(
    embedding: torch.Tensor,
    targets: torch.Tensor,
    margin: float,
    gamma: float,
    loss_weight: float = 1.0,
) -> torch.Tensor:
    embedding = F.normalize(embedding, dim=1)

    dist_mat = torch.matmul(embedding, embedding.t())

    N = dist_mat.size(0)

    is_pos = (
        targets.view(N, 1)
        .expand(N, N)
        .eq(targets.view(N, 1).expand(N, N).t())
        .float()
    )
    is_neg = (
        targets.view(N, 1)
        .expand(N, N)
        .ne(targets.view(N, 1).expand(N, N).t())
        .float()
    )

    # Mask scores related to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.0)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.0)
    delta_p = 1 - margin
    delta_n = margin

    logit_p = -gamma * alpha_p * (s_p - delta_p) + (-99999999.0) * (1 - is_pos)
    logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.0) * (1 - is_neg)

    loss = F.softplus(
        torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)
    ).mean()

    return loss_weight * loss


@LOSSES.register_module()
class CircleLoss(nn.Module):

    loss_name = "circle_loss"

    def __init__(
        self,
        margin=0.25,
        gamma=128,
        loss_weight=1.0,
    ):
        super(CircleLoss, self).__init__()
        self.circle_loss = partial(
            pairwise_circleloss,
            margin=margin,
            gamma=gamma,
            loss_weight=loss_weight,
        )

    def forward(
        self,
        inputs,
        targets,
        **kwargs,
    ):
        return self.circle_loss(inputs, targets)
