
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
from vit_encoder import ViTEncoder
from attention import SkipFusion
from decoder import MambaDecoderBlock


class TokenToMap(nn.Module):
    def __init__(self, img_size=224, patch_size=16):
        super().__init__()
        self.H = img_size // patch_size 
        self.W = img_size // patch_size   

    def forward(self, x):

        B, N, D = x.shape

        expected_tokens = self.H * self.W
        if N != expected_tokens:
            raise ValueError(
                f"Token count mismatch: got N={N}, expected {expected_tokens} "
                f"for feature map size ({self.H}, {self.W})"
            )

        return x.transpose(1, 2).reshape(B, D, self.H, self.W)


class TSSMUNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=320, out_channels=1):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        self.encoder = ViTEncoder()
        self.token2map = TokenToMap(img_size=img_size, patch_size=patch_size)


        self.reduce1 = nn.Conv2d(embed_dim, 96, 1)
        self.reduce2 = nn.Conv2d(embed_dim, 128, 1)
        self.reduce3 = nn.Conv2d(embed_dim, 192, 1)
        self.reduce4 = nn.Conv2d(embed_dim, 320, 1)

        self.skip1 = SkipFusion(96)
        self.skip2 = SkipFusion(128)
        self.skip3 = SkipFusion(192)
        self.skip4 = SkipFusion(320)

        self.dec4 = MambaDecoderBlock(320 + 320, 320)

        self.dec3 = MambaDecoderBlock(320 + 192, 192)

        self.dec2 = MambaDecoderBlock(192 + 128, 128)

        self.dec1 = MambaDecoderBlock(128 + 96, 96)


        self.final = nn.Conv2d(96, out_channels, kernel_size=1)


        if self.final.bias is not None:
            nn.init.constant_(self.final.bias, -2.0)

    def forward(self, x):
        H, W = x.shape[2:]  


        f1, f2, f3, f4, f5 = self.encoder(x)


        f1 = self.token2map(f1)
        f2 = self.token2map(f2)
        f3 = self.token2map(f3)
        f4 = self.token2map(f4)
        f5 = self.token2map(f5)

        f1 = self.reduce1(f1)   
        f2 = self.reduce2(f2)   
        f3 = self.reduce3(f3)   
        f4 = self.reduce4(f4)  

        d4 = self.dec4(f5, f4)
        d4 = self.skip4(f4, d4)

        d3 = self.dec3(d4, f3)
        d3 = self.skip3(f3, d3)

        d2 = self.dec2(d3, f2)
        d2 = self.skip2(f2, d2)

        d1 = self.dec1(d2, f1)
        d1 = self.skip1(f1, d1)


        out = self.final(d1)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out  


if __name__ == "__main__":
    model = TSSMUNet(img_size=224, patch_size=16, embed_dim=320, out_channels=1)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
