# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F

class Bitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, is_symmetric=False, is_unimodal=False, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.is_unimodal = is_unimodal
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1,-1), 0, 0.01))
        if is_symmetric:
            self.b = nn.Parameter(torch.nn.init.zeros_(torch.empty(channel).view(1,-1)), requires_grad=False)
        else:
            self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1,-1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1,-1), 0, 0.01))
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
        if self.is_unimodal:
            a = torch.abs(a)
        if self.final:
            return torch.sigmoid(x * F.softplus(h) + b)
        else:
            x = x * F.softplus(h) + b
            return x + torch.tanh(x) * torch.tanh(a)

class BitEstimator(nn.Module):
    '''
    Estimate bit
    '''
    def __init__(self, channel, is_symmetric=False, is_unimodal=False, num_layers = 4):
        super(BitEstimator, self).__init__()
        self.num_layers = num_layers
        self.num_channels = channel
        self.f1 = Bitparm(channel, is_symmetric=is_symmetric, is_unimodal=is_unimodal)
        self.f2 = Bitparm(channel, is_symmetric=is_symmetric, is_unimodal=is_unimodal)
        self.f3 = Bitparm(channel, is_symmetric=is_symmetric, is_unimodal=is_unimodal)
        self.f4 = Bitparm(channel, is_symmetric=is_symmetric, is_unimodal=is_unimodal, final=True)
        
    def forward(self, x, single_channel=None):
        if self.num_layers>1:
            x = self.f1(x, single_channel)
        if self.num_layers>2:
            x = self.f2(x, single_channel)
        if self.num_layers>3:
            x = self.f3(x, single_channel)
        return self.f4(x, single_channel)
        