#!/usr/bin/env python3

# import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseTemporalLayer
from ..builder import TEMPORAL


@TEMPORAL.register_module()
class RNN(BaseTemporalLayer):

    _rnn_type = ("rnn", "lstm", "gru")

    def __init__(
        self,
        in_channels,
        rnn_type="lstm",
        num_layers=1,
        nonlinearity="tanh",
        bidirectional=False,
        return_hidden=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.return_hidden = return_hidden

        assert rnn_type in self._rnn_type
        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=self.in_channels,
                hidden_size=self.in_channels,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.in_channels,
                hidden_size=self.in_channels,
                num_layers=num_layers,
                batch_first=True,  # input is (batch, seq, feat)
                bidirectional=bidirectional,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.in_channels,
                hidden_size=self.in_channels,
                num_layers=num_layers,
                batch_first=True,  # input is (batch, seq, feat)
                bidirectional=bidirectional,
            )
        else:
            raise ValueError()

        self.last_dim = (
            self.in_channels * 2
            if bidirectional and not return_hidden
            else self.in_channels
        )

        self._init_layers()

    def _init_layers(self):
        """Initialize fc layers."""
        # for m in self.modules():
        #     if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
        #         for name, param in m.named_parameters():
        #             if 'weight_ih' in name:
        #                 torch.nn.init.xavier_uniform_(param.data)
        #             elif 'weight_hh' in name:
        #                 torch.nn.init.orthogonal_(param.data)
        #             elif 'bias' in name:
        #                 param.data.fill_(0)
        ...
        # TODO: better initialization

    def _forward(self, x, **kwargs):
        assert len(x.shape) == 3
        b, s, f_dim = x.shape
        assert f_dim == self.in_channels

        # TODO: would be better if we had a sequential inference mode

        self.rnn.flatten_parameters()  # NOTE: UserWarning
        if self.return_hidden:
            _, h = self.rnn(x)
            if self.rnn_type == "lstm":
                h = h[0]
                # c = h[1]
            if h.shape[0] > 1:
                # has more than 1 layer
                # just get the last layer
                h = h[-1, :, :]
            h = h.squeeze(0)
            return h
        else:
            v, _ = self.rnn(x)
            v = v.permute(0, 2, 1)
            v = F.avg_pool1d(v, s)
            v = v.squeeze(-1)
            return v
