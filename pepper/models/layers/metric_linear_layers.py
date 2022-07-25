#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import METRIC_LINEAR_LAYERS

__all__ = [
    "Linear",
    "SphereSoftmax",
    "ArcSoftmax",
    "CosSoftmax",
    "CircleSoftmax",
]


@METRIC_LINEAR_LAYERS.register_module()
class Linear(nn.Module):
    def __init__(self, in_channels, num_classes, scale=1.0, margin=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.s = scale
        self.m = margin

        # could replace it with a Linear layer
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_channels))

    def forward(self, logits, targets):
        logits = F.linear(logits, self.weight)
        return logits.mul_(self.s)

    def extra_repr(self):
        return (
            f"num_classes={self.num_classes}, scale={self.s}, margin={self.m}"
        )


@METRIC_LINEAR_LAYERS.register_module()
class SphereSoftmax(Linear):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0

        # duplication formula
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x,
            lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x,
        ]

    def foward(self, logits, targets):

        if targets is None:
            return logits

        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(
            self.LambdaMin,
            self.base * (1 + self.gamma * self.iter) ** (-1 * self.power),
        )

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(logits), F.normalize(self.weight))

        # if targets is None:
        #     # NOTE: we don't have scale
        #     return cos_theta

        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, targets.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (
            one_hot * (phi_theta - cos_theta) / (1 + self.lamb)
        ) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output


@METRIC_LINEAR_LAYERS.register_module()
class CosSoftmax(Linear):
    """Implement of large margin cosine distance:"""

    def forward(self, logits, targets):
        logits = F.linear(
            F.normalize(logits), F.normalize(self.weight), bias=None
        )

        if targets is None:
            return logits.mul_(self.s)

        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(
            index.size()[0],
            logits.size()[1],
            device=logits.device,
            dtype=logits.dtype,
        )
        m_hot.scatter_(1, targets[index, None], self.m)
        logits[index] -= m_hot
        logits.mul_(self.s)
        return logits


@METRIC_LINEAR_LAYERS.register_module()
class ArcSoftmax(Linear):
    def forward(self, logits, targets):
        logits = F.linear(
            F.normalize(logits), F.normalize(self.weight), bias=None
        )

        if targets is None:
            return logits * self.s

        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(
            index.size()[0],
            logits.size()[1],
            device=logits.device,
            dtype=logits.dtype,
        )
        m_hot.scatter_(1, targets[index, None], self.m)
        logits.acos_()
        logits[index] += m_hot
        logits.cos_().mul_(self.s)
        return logits


@METRIC_LINEAR_LAYERS.register_module()
class CircleSoftmax(Linear):
    def forward(self, logits, targets):
        logits = F.linear(
            F.normalize(logits), F.normalize(self.weight), bias=None
        )

        if targets is None:
            return logits.mul_(self.s)

        alpha_p = torch.clamp_min(-logits.detach() + 1 + self.m, min=0.0)
        alpha_n = torch.clamp_min(logits.detach() + self.m, min=0.0)
        delta_p = 1 - self.m
        delta_n = self.m

        # When use model parallel, there are some targets not in class centers of local rank
        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(
            index.size()[0],
            logits.size()[1],
            device=logits.device,
            dtype=logits.dtype,
        )
        m_hot.scatter_(1, targets[index, None], 1)

        logits_p = alpha_p * (logits - delta_p)
        logits_n = alpha_n * (logits - delta_n)

        logits[index] = logits_p[index] * m_hot + logits_n[index] * (1 - m_hot)

        neg_index = torch.where(targets == -1)[0]
        logits[neg_index] = logits_n[neg_index]

        logits.mul_(self.s)

        return logits
