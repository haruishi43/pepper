#!/usr/bin/env python3

import torch
from torch import nn

from pepper.models.builder import LOSSES


@LOSSES.register_module()
class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    loss_name = "center_loss"

    def __init__(
        self,
        num_classes=751,
        feat_dim=2048,
        loss_weight=1.0,
    ):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.feat_dim)
        )

    def forward(
        self,
        inputs,
        targets,
        **kwargs,
    ):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim).
            targets: ground truth labels with shape (num_classes).
        """

        center = self.centers[targets]
        dist = (inputs - center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e12).mean(dim=-1)

        # non-optimized
        # assert inputs.size(0) == targets.size(
        #     0
        # ), "features.size(0) is not equal to labels.size(0)"

        # batch_size = inputs.size(0)
        # distmat = (
        #     torch.pow(inputs, 2)
        #     .sum(dim=1, keepdim=True)
        #     .expand(batch_size, self.num_classes)
        #     + torch.pow(self.centers, 2)
        #     .sum(dim=1, keepdim=True)
        #     .expand(self.num_classes, batch_size)
        #     .t()
        # )
        # distmat.addmm_(1, -2, inputs, self.centers.t())

        # classes = torch.arange(self.num_classes).long()
        # if self.use_gpu:
        #     classes = classes.cuda()
        # targets = targets.unsqueeze(1).expand(batch_size, self.num_classes)
        # mask = targets.eq(classes.expand(batch_size, self.num_classes))

        # dist = distmat * mask.float()
        # loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size

        # notused!
        # dist = []
        # for i in range(batch_size):
        #    value = distmat[i][mask[i]]
        #    value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #    dist.append(value)
        # dist = torch.cat(dist)
        # loss = dist.mean()

        return self.loss_weight * loss


if __name__ == "__main__":
    use_gpu = False
    center_loss = CenterLoss(use_gpu=use_gpu)
    features = torch.rand(16, 2048)
    targets = torch.Tensor(
        [0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]
    ).long()
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor(
            [0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]
        ).cuda()

    loss = center_loss(features, targets)
    print(loss)
