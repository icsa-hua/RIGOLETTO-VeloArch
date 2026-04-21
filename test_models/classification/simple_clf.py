"""
SimpleCLF — lightweight CNN image classifier.

Web app usage
-------------
  Task type   : Classification
  Weights file: simple_clf.pth   (state dict — this .py file is required)
  Code file   : simple_clf.py    (this file)
  Input shape : B=1  C=3  H=224  W=224

  OR upload simple_clf_full.pt (full model — no code file needed).

Architecture
------------
  Conv(3→32) → Pool → Conv(32→64) → Pool → Conv(64→128) → AdaptAvgPool(4×4)
  → Flatten → Linear(2048→256) → Linear(256→10)
  ~1.1 M parameters
"""

import torch.nn as nn


class SimpleCLF(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # → 112×112

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # → 56×56

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),                  # → 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Required by VeloArch when this file is uploaded as "Model Class Code"
model = SimpleCLF(num_classes=10)
