#!/usr/bin/env python3

from torch import nn
from torch.nn import functional as F


class RNN(nn.Module):

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
            self.gru = nn.GRU(
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

        # TODO:
        # - what to do for multi-layer inputs (tuple of tensors)
