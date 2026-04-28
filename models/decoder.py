# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:27:13 2026

@author: Santosh Prakash
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vssm import VisionMambaBlock

class MambaDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Reduce channels after fusion
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        # VSSM block
        self.vssm = VisionMambaBlock(out_channels)

    def forward(self, x, skip):
        # Upsample
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        # Concatenate instead of add (SAFE)
        x = torch.cat([x, skip], dim=1)

        # Channel reduction
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)

        # VSSM
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.vssm(x, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x