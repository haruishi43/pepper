#!/usr/bin/env python3

import torch
import torch.nn as nn

from .resnet import ResNet
from ..builder import BACKBONES


class RGAModule(nn.Module):
    def __init__(
        self,
        in_channel: int,
        in_spatial: int,
        use_spatial: bool = True,
        use_channel: bool = True,
        cha_ratio: int = 8,
        spa_ratio: int = 8,
        down_ratio: int = 8,
    ):
        super().__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        # Embedding functions for original features
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channel,
                    out_channels=self.inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU(inplace=True),
            )
        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_spatial,
                    out_channels=self.inter_spatial,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU(inplace=True),
            )

        # Embedding functions for relation features
        if self.use_spatial:
            self.gg_spatial = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_spatial * 2,
                    out_channels=self.inter_spatial,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU(inplace=True),
            )
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channel * 2,
                    out_channels=self.inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU(inplace=True),
            )

        # Networks for learning attention weights
        if self.use_spatial:
            num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2d(
                    in_channels=num_channel_s,
                    out_channels=num_channel_s // down_ratio,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_channel_s // down_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=num_channel_s // down_ratio,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(1),
            )
        if self.use_channel:
            num_channel_c = 1 + self.inter_channel
            self.W_channel = nn.Sequential(
                nn.Conv2d(
                    in_channels=num_channel_c,
                    out_channels=num_channel_c // down_ratio,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_channel_c // down_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=num_channel_c // down_ratio,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(1),
            )

        # Embedding functions for modeling relations
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channel,
                    out_channels=self.inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU(inplace=True),
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channel,
                    out_channels=self.inter_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU(inplace=True),
            )
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_spatial,
                    out_channels=self.inter_spatial,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU(inplace=True),
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_spatial,
                    out_channels=self.inter_spatial,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        if self.use_spatial:
            # spatial attention
            theta_xs = self.theta_spatial(x)
            phi_xs = self.phi_spatial(x)
            theta_xs = theta_xs.view(b, self.inter_channel, -1)
            theta_xs = theta_xs.permute(0, 2, 1)
            phi_xs = phi_xs.view(b, self.inter_channel, -1)
            Gs = torch.matmul(theta_xs, phi_xs)
            Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)
            Gs_out = Gs.view(b, h * w, h, w)
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)
            Gs_joint = self.gg_spatial(Gs_joint)

            g_xs = self.gx_spatial(x)
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)
            ys = torch.cat((g_xs, Gs_joint), 1)

            W_ys = self.W_spatial(ys)
            if not self.use_channel:
                return torch.sigmoid(W_ys.expand_as(x)) * x

            x = torch.sigmoid(W_ys.expand_as(x)) * x

        if self.use_channel:
            # channel attention
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
            phi_xc = self.phi_channel(xc).squeeze(-1)
            Gc = torch.matmul(theta_xc, phi_xc)
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
            Gc_out = Gc.unsqueeze(-1)
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)
            Gc_joint = self.gg_channel(Gc_joint)

            g_xc = self.gx_channel(xc)
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            yc = torch.cat((g_xc, Gc_joint), 1)

            W_yc = self.W_channel(yc).transpose(1, 2)
            x = torch.sigmoid(W_yc) * x

        return x


@BACKBONES.register_module()
class RGAResNet(ResNet):
    def __init__(
        self,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 1),
        height=256,
        width=128,
        use_spatial=True,
        use_channel=True,
        scale=8,
        d_scale=8,
        **kwargs,
    ):
        assert strides == (1, 2, 2, 1)
        super().__init__(
            base_channels=base_channels,
            num_stages=num_stages,
            strides=strides,
            **kwargs,
        )

        # some hard-coded parameters
        c_ratio = s_ratio = scale
        d_ratio = d_scale

        att_channels = [256, 512, 1024, 2048][:num_stages]
        att_spatials = [
            (height // 4) * (width // 4),
            (height // 8) * (width // 8),
            (height // 16) * (width // 16),
            (height // 16) * (width // 16),
        ][:num_stages]

        # RGA Modules
        rga_modules = []
        for i in range(num_stages):
            rga_modules.append(
                RGAModule(
                    in_channel=att_channels[i],
                    in_spatial=att_spatials[i],
                    use_spatial=use_spatial,
                    use_channel=use_channel,
                    cha_ratio=c_ratio,
                    spa_ratio=s_ratio,
                    down_ratio=d_ratio,
                )
            )
        self.rga_modules = nn.ModuleList(rga_modules)

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            # add rga module here
            x = self.rga_modules[i](x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
