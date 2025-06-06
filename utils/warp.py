# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from numpy import sin
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.conv import Conv1d

class Layer(nn.Module):
    def __init__(self, channel, final=False):
        super(Layer, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1,-1), 0, 0.01), requires_grad=True)
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1,-1), 0, 0.01), requires_grad=True)
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1,-1), 0, 0.01), requires_grad=True)
        else:
            self.a = None

    def forward(self, x, single_channel=None):
        if single_channel is not None:
            h = self.h[:,single_channel]
            b = self.b[:,single_channel]
            if not self.final:
                a = self.a[:,single_channel]
        else:
            h = self.h
            b = self.b
            if not self.final:
                a = self.a
        if self.final:
            return x * F.softplus(h) + b
        else:
            x = x * F.softplus(h) + b
            return x + torch.tanh(x) * torch.tanh(a)

class MonotonicNN(nn.Module):
    def __init__(self, channel, n_layers):
        super(MonotonicNN, self).__init__()
        self._layers = []
        for _ in range(n_layers-1):
            layer = Layer(channel)
            self._layers.append(layer)
        self._layers.append(Layer(channel, final=True))
        self._layers = nn.ModuleList(self._layers)
        
    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x