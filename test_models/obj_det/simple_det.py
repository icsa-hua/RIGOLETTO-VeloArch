"""
SimpleDetector — anchor-free single-shot object detector backbone + head.

Web app usage
-------------
  Task type   : Object Detection
  Weights file: simple_det.pth   (state dict — this .py file IS required)
  Code file   : simple_det.py    (this file)
  Input shape : B=1  C=3  H=320  W=320

  For YOLO models (yolov8n.pt etc.) no code file is needed — they save the
  full model and are loaded automatically.

Architecture
------------
  Backbone (stride-32):
    Conv(3→32,s2) → Conv(32→64,s2) → Conv(64→128,s2) → Conv(128→256,s2)
  Detection head (per-cell):
    cls_head: Conv(256→num_classes, 1×1)   — class logits
    reg_head: Conv(256→4, 1×1)             — bbox offsets (cx,cy,w,h)
  ~1.0 M parameters
"""

import torch.nn as nn


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleDetector(nn.Module):
    def __init__(self, num_classes: int = 20):
        super().__init__()
        self.backbone = nn.Sequential(
            _ConvBnRelu(3,   32,  stride=2),   # → 160×160
            _ConvBnRelu(32,  64,  stride=2),   # → 80×80
            _ConvBnRelu(64,  128, stride=2),   # → 40×40
            _ConvBnRelu(128, 256, stride=2),   # → 20×20
        )
        self.cls_head = nn.Conv2d(256, num_classes, 1)
        self.reg_head = nn.Conv2d(256, 4, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.cls_head(feat), self.reg_head(feat)


# Required by VeloArch when this file is uploaded as "Model Class Code"
model = SimpleDetector(num_classes=20)
