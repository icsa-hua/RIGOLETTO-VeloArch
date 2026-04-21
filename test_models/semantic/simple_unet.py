"""
SimpleUNet — lightweight encoder-decoder for semantic segmentation.

Web app usage
-------------
  Task type   : Segmentation
  Weights file: simple_unet.pth  (state dict — this .py file IS required)
  Code file   : simple_unet.py   (this file)
  Input shape : B=1  C=3  H=256  W=256
                (H and W must each be divisible by 8)

Architecture
------------
  Encoder : (3→32) → (32→64) → (64→128)  [3× MaxPool 2×2]
  Bottleneck: 128→256
  Decoder : up+concat → (256→128) → up+concat → (128→64)
            → up+concat → (64→32) → Conv 1×1 → num_classes
  ~4.0 M parameters
"""

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 21):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = _DoubleConv(in_channels, 32)   # → H×W
        self.enc2 = _DoubleConv(32, 64)             # → H/2
        self.enc3 = _DoubleConv(64, 128)            # → H/4

        # Bottleneck
        self.bottleneck = _DoubleConv(128, 256)     # → H/8

        # Decoder
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)  # → H/4
        self.dec3 = _DoubleConv(256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)   # → H/2
        self.dec2 = _DoubleConv(128, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)    # → H
        self.dec1 = _DoubleConv(64, 32)

        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


# Required by VeloArch when this file is uploaded as "Model Class Code"
model = SimpleUNet(in_channels=3, num_classes=21)
