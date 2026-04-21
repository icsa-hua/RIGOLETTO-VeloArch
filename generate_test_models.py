"""Run once to create test_models/ weight files."""
import os, shutil
import torch
import torch.nn as nn
from torchvision.models import resnet18


def _init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class _BnRelu(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.bn  = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(x))


class SimpleCLF(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), _BnRelu(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), _BnRelu(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), _BnRelu(128),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class _DC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        self.pool       = nn.MaxPool2d(2)
        self.enc1       = _DC(in_channels, 32)
        self.enc2       = _DC(32, 64)
        self.enc3       = _DC(64, 128)
        self.bottleneck = _DC(128, 256)
        self.up3        = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3       = _DC(256, 128)
        self.up2        = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec2       = _DC(128,  64)
        self.up1        = nn.ConvTranspose2d(64,  32,  2, stride=2)
        self.dec1       = _DC(64,   32)
        self.head       = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ─────────────────────────────────────────────────────────────────────────────
# OBJECT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class _CBR(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class SimpleDetector(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = nn.Sequential(
            _CBR(3,   32,  stride=2),   # 160×160
            _CBR(32,  64,  stride=2),   #  80×80
            _CBR(64,  128, stride=2),   #  40×40
            _CBR(128, 256, stride=2),   #  20×20
        )
        self.cls_head = nn.Conv2d(256, num_classes, 1)
        self.reg_head = nn.Conv2d(256, 4, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.cls_head(feat), self.reg_head(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Generate files
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("test_models/classification", exist_ok=True)
    os.makedirs("test_models/semantic",       exist_ok=True)
    os.makedirs("test_models/obj_det",        exist_ok=True)

    # --- Classification (full model from torchvision — no code file needed at load time) ---
    r18 = resnet18(weights=None)   # random weights, no download
    r18.apply(_init); r18.eval()

    with torch.no_grad():
        out = r18(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 1000)

    torch.save(r18, "test_models/classification/resnet18_full.pt")
    sz_full = os.path.getsize("test_models/classification/resnet18_full.pt") // 1024
    print(f"[classification] resnet18_full.pt     {sz_full} KB  (full torchvision model, no code needed)")

    # --- Classification (custom model — state dict, needs simple_clf.py) ---
    clf = SimpleCLF(num_classes=10)
    clf.apply(_init); clf.eval()

    with torch.no_grad():
        out = clf(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 10)

    torch.save(clf.state_dict(), "test_models/classification/simple_clf.pth")
    sz_sd = os.path.getsize("test_models/classification/simple_clf.pth") // 1024
    print(f"[classification] simple_clf.pth       {sz_sd} KB  (state dict, needs simple_clf.py)")

    # --- Segmentation ---
    seg = SimpleUNet(in_channels=3, num_classes=21)
    seg.apply(_init); seg.eval()

    with torch.no_grad():
        out = seg(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, 21, 256, 256)

    torch.save(seg.state_dict(), "test_models/semantic/simple_unet.pth")

    sz = os.path.getsize("test_models/semantic/simple_unet.pth") // 1024
    print(f"\n[semantic]       simple_unet.pth      {sz} KB  (state dict, needs simple_unet.py)")

    # --- Object Detection — custom ---
    det = SimpleDetector(num_classes=20)
    det.apply(_init); det.eval()

    with torch.no_grad():
        cls_out, reg_out = det(torch.zeros(1, 3, 320, 320))
    assert cls_out.shape == (1, 20, 20, 20)
    assert reg_out.shape == (1,  4, 20, 20)

    torch.save(det.state_dict(), "test_models/obj_det/simple_det.pth")

    sz = os.path.getsize("test_models/obj_det/simple_det.pth") // 1024
    print(f"\n[obj_det]        simple_det.pth        {sz} KB  (state dict, needs simple_det.py)")

    # --- Object Detection — YOLO (copy, no code needed) ---
    yolo_src = "models/obj_detection/yolov8n.pt"
    yolo_dst = "test_models/obj_det/yolov8n.pt"
    shutil.copy2(yolo_src, yolo_dst)
    sz = os.path.getsize(yolo_dst) // 1024
    print(f"[obj_det]        yolov8n.pt             {sz} KB  (full YOLO model, no code needed)")

    print("\nAll test models generated.")
