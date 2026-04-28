import torch
import torch.nn as nn
import torch.nn.functional as F
from vssm import VisionMambaBlock

class MambaDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)


        self.vssm = VisionMambaBlock(out_channels)

    def forward(self, x, skip):

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)


        x = torch.cat([x, skip], dim=1)


        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)


        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.vssm(x, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x
