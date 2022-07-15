#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import cosine_dist, euclidean_dist


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = (
        torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6
    )  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


@LOSSES.register_module()
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
        Hermans et al. In Defense of the Triplet Loss for
            Person Re-Identification. arXiv:1703.07737.
    Imported from `<https://github.com/KaiyangZhou/deep-person-reid/blob/
        master/torchreid/losses/hard_mine_triplet_loss.py>`_.
    Args:
        margin (float, optional): Margin for triplet loss. Default to 0.3.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
    """

    def __init__(
        self,
        margin=0.3,
        loss_weight=1.0,
        norm_feat=False,
        hard_mining=True,
    ):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.norm_feat = norm_feat
        self.loss_weight = loss_weight
        self.hard_mining = hard_mining

    def forward(self, inputs, targets, **kwargs):
        """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
        Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
        Loss for Person Re-Identification'."""

        if self.norm_feat:
            dist_mat = cosine_dist(inputs, inputs)
        else:
            dist_mat = euclidean_dist(inputs, inputs)

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

        if self.hard_mining:
            dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
        else:
            dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin > 0:
            loss = F.margin_ranking_loss(
                dist_an, dist_ap, y, margin=self.margin
            )
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            if loss == float("Inf"):
                loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)

        return self.loss_weight * loss
