#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16, force_fp32

from .basic_head import BasicHead
from ..builder import HEADS
from ..utils.mgn_utils import Pruning, Classifier, PartClassifier

EPSILON = 1e-12


class AttentionAwareModule(nn.Module):
    """Attention-Aware Module with Bilinear Attention Pooling"""

    def __init__(
        self,
        in_channels,
        out_channels,
        att_channels=32,
        pool="GAP",
        norm_cfg=dict(type="BN1d", requires_grad=True),
    ):
        super().__init__()

        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.att = nn.Conv2d(out_channels, att_channels, kernel_size=1)

        assert pool in ["GAP", "GMP"]
        if pool == "GAP":
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

        self.norm = build_norm_layer(norm_cfg, out_channels * att_channels)[1]

    def init_weights(self):
        # conv
        nn.init.kaiming_normal_(self.reduce.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.att.weight, mode="fan_in")

        # bn
        nn.init.normal_(self.norm.weight, mean=1.0, std=0.02)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x):
        x = self.reduce(x)
        attentions = self.att(x)
        B, C, H, W = x.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature = (
                torch.einsum("imjk,injk->imn", (attentions, x)) / float(H * W)
            ).view(B, -1)
        else:
            feature = []
            for i in range(M):
                AiF = self.pool(x * attentions[:, i : i + 1, ...]).view(B, -1)
                feature.append(AiF)
            feature = torch.cat(feature, dim=1)

        # sign-sqrt
        output = torch.sign(feature) * torch.sqrt(torch.abs(feature) + EPSILON)

        # l2 normalization along dimension M and C
        # feature = F.normalize(feature, dim=-1)

        # normalize output
        output = self.norm(output)
        return output


@HEADS.register_module()
class AMGNHead(BasicHead):
    def __init__(
        self,
        out_channels=256,
        att_channels=8,
        **kwargs,
    ):
        self.out_channels = out_channels
        self.att_channels = att_channels
        super().__init__(**kwargs)

    def _init_layers(self):
        convs = []
        for _ in range(3):
            convs.append(
                Pruning(
                    self.in_channels,
                    self.out_channels,
                    norm_cfg=self.norm_cfg,
                )
            )
        self.convs = nn.ModuleList(convs)

        global_classifiers = []
        for _ in range(3):
            global_classifiers.append(
                Classifier(
                    self.in_channels, self.num_classes, act_cfg=self.act_cfg
                )
            )
        self.global_classifiers = nn.ModuleList(global_classifiers)

        part_convs = []
        for _ in range(2):
            part_convs.append(
                Pruning(
                    self.in_channels,
                    self.out_channels,
                    norm_cfg=self.norm_cfg,
                )
            )
        self.part_convs = nn.ModuleList(part_convs)

        # attention-aware module
        self.atta = AttentionAwareModule(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            att_channels=self.att_channels,
            pool="GAP",
            norm_cfg=dict(type="BN1d", requires_grad=True),
        )

        # part2
        self.part2_classifier = PartClassifier(
            in_channels=self.out_channels,
            num_classes=self.num_classes,
            num_parts=2,
            act_cfg=self.act_cfg,
        )

        # part3
        self.part3_classifier = PartClassifier(
            in_channels=self.out_channels,
            num_classes=self.num_classes,
            num_parts=3,
            act_cfg=self.act_cfg,
        )

        self.atta_classifier = Classifier(
            self.out_channels * self.att_channels,
            self.num_classes,
            act_cfg=self.act_cfg,
        )

    def init_weights(self):
        for conv in self.convs:
            conv.init_weights()

        for c in self.global_classifiers:
            c.init_weights()

        for c in self.part_convs:
            c.init_weights()

        self.part2_classifier.init_weights()
        self.part3_classifier.init_weights()
        self.atta.init_weights()

    @auto_fp16()
    def pre_logits(self, x):
        assert isinstance(x, (list, tuple))
        assert len(x) == 6

        (p1_global, p2_global, p3_global, p2_parts, p3_parts, a1) = x

        b = p1_global.size(0)

        p1_feat = self.convs[0](p1_global).view(b, -1)
        p2_feat = self.convs[1](p2_global).view(b, -1)
        p3_feat = self.convs[2](p3_global).view(b, -1)

        p2_part_feats = self.part_convs[0](p2_parts)
        p3_part_feats = self.part_convs[1](p3_parts)

        p2_part_feat = []
        for i in range(2):
            p2_part_feat.append(p2_part_feats[:, :, i : (i + 1), :].view(b, -1))

        p3_part_feat = []
        for i in range(3):
            p3_part_feat.append(p3_part_feats[:, :, i : (i + 1), :].view(b, -1))

        a1_feat = self.atta(a1)

        if self.training:
            return (
                p1_global.view(b, -1),
                p2_global.view(b, -1),
                p3_global.view(b, -1),
                p1_feat,
                p2_feat,
                p3_feat,
                p2_part_feat,
                p3_part_feat,
                a1_feat,
            )
        else:
            return torch.cat(
                [
                    p1_feat,
                    p2_feat,
                    p3_feat,
                    *p2_part_feat,
                    *p3_part_feat,
                    a1_feat,
                ],
                dim=-1,
            )

    def forward_train(self, x):
        (
            p1_global,
            p2_global,
            p3_global,
            p1_feat,
            p2_feat,
            p3_feat,
            p2_part_feat,
            p3_part_feat,
            a1_feat,
        ) = x

        feats = dict(
            g1_feat=p1_feat,
            g2_feat=p2_feat,
            g3_feat=p3_feat,
            a1_feat=a1_feat,
        )

        if self.loss_cls:
            cls_scores = dict()

            names = ("global1", "global2", "global3")
            gfeats = (p1_global, p2_global, p3_global)

            for name, gf, gc in zip(names, gfeats, self.global_classifiers):
                cls_scores[name] = gc(gf)

            preds = self.part2_classifier(p2_part_feat)
            for i, pred in enumerate(preds):
                cls_scores[f"part2_{i}"] = pred

            preds = self.part3_classifier(p3_part_feat)
            for i, pred in enumerate(preds):
                cls_scores[f"part3_{i}"] = pred

            cls_scores["att"] = self.atta_classifier(a1_feat)

            return (feats, cls_scores)

        return (feats, None)

    @force_fp32(apply_to=("feats", "cls_score"))
    def loss(self, gt_label, feats, cls_score=None):
        losses = dict()

        if self.loss_cls:
            assert cls_score is not None
            assert isinstance(cls_score, dict)

            if not isinstance(self.loss_cls, nn.ModuleList):
                loss_cls = [self.loss_cls]
            else:
                loss_cls = self.loss_cls

            for name, cs in cls_score.items():
                for lc in loss_cls:
                    if lc.loss_name not in losses:
                        losses[lc.loss_name] = lc(
                            cls_score=cs,
                            label=gt_label,
                        )
                    else:
                        losses[lc.loss_name] += lc(
                            cls_score=cs,
                            label=gt_label,
                        )

                # compute accuracy
                acc = self.accuracy(cs, gt_label)
                assert len(acc) == len(self.topk)
                if "accuracy" not in losses:
                    losses["accuracy"] = {
                        f"{name}_top-{k}": a for k, a in zip(self.topk, acc)
                    }
                else:
                    accuracy = {
                        f"{name}_top-{k}": a for k, a in zip(self.topk, acc)
                    }
                    losses["accuracy"].update(accuracy)

        if self.loss_pairwise:
            assert isinstance(feats, dict)

            if not isinstance(self.loss_pairwise, nn.ModuleList):
                loss_pairwise = [self.loss_pairwise]
            else:
                loss_pairwise = self.loss_pairwise

            for _, f in feats.items():
                for lp in loss_pairwise:
                    if lp.loss_name not in losses:
                        losses[lp.loss_name] = lp(
                            inputs=f,
                            targets=gt_label,
                        )
                    else:
                        losses[lp.loss_name] += lp(
                            inputs=f,
                            targets=gt_label,
                        )

        return losses
