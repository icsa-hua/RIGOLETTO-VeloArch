# VeloArch — Cheatsheet

AI accelerator architecture analyser. Upload a PyTorch model, pick target hardware, get latency / energy / bottleneck estimates instantly.

**EU RIGOLETTO Project · Harokopio University of Athens**

---

## Quick Start

```bash
conda activate veloarch
python app.py
# open http://localhost:5000
```

---

## How It Works

```
PyTorch model file (.pt / .pth)
        │
        ▼
  load_model_from_path()          ← auto-detects format
        │
        ▼
  ModelParser.analyze()           ← hook-based FLOPs + memory count
        │
        ▼
  Evaluator.run()                 ← roofline model per architecture
        │
        ▼
  latency · energy · throughput · bottleneck
```

### 1 · Model Loading

Four formats are supported, detected automatically:

| File | Code file needed? | How saved |
|------|:-----------------:|-----------|
| `resnet18_full.pt` | No | `torch.save(model, path)` — class from installed package |
| `simple_clf.pth` | **Yes** | `torch.save(model.state_dict(), path)` — custom class |
| `yolov8n.pt` | No | Ultralytics checkpoint (`ckpt['model']`) |
| TorchScript `.pt` | No | `torch.jit.save(scripted, path)` |

**State-dict rule:** if the class is not from an installed package (torchvision, ultralytics…), you must upload a `.py` file alongside the weights.

### 2 · Model Class Code (`.py`)

Required only for state-dict `.pth` files with custom architectures.  
The script must expose the model in one of these ways:

```python
# preferred
model = MyNet(num_classes=10)

# or
def get_model():
    return MyNet(num_classes=10)

# or — works only if __init__ takes no required args
class MyNet(nn.Module): ...
```

### 3 · FLOPs Counter

Hook-based, fires on `Conv1d`, `Conv2d`, `Conv3d`, `Linear`.  
Convention: **1 multiply-add = 2 ops** (IEEE standard).

```
ops  = 2 × B × H × W × Cin × Cout × Kh × Kw   (Conv2d)
ops  = 2 × B × in_features × out_features        (Linear)
mem  = output.numel() × 4 bytes                  (all layers, float32)
```

A single zero-input forward pass is run. Partial counts are returned if the pass fails (e.g. YOLO post-processing on a blank image).

### 4 · Roofline Model

Each workload–architecture pair is evaluated analytically:

```
arithmetic_intensity  =  ops / memory_bytes          (ops/byte)
ridge_point           =  peak_compute / bandwidth     (ops/byte)

latency   = max(ops / peak_compute,  memory_bytes / bandwidth)
energy    = ops × energy_per_op  +  memory_bytes × energy_per_byte
bottleneck = "compute"  if AI ≥ ridge_point
             "memory"   if AI <  ridge_point
```

### 5 · Preset Architectures

| Name | Parallel Units | Frequency | Bandwidth | Energy/op |
|------|:--------------:|:---------:|:---------:|:---------:|
| VPU | 256 | 1 GHz | 50 GB/s | 2 pJ |
| NPU | 1024 | 1 GHz | 100 GB/s | 1 pJ |
| VPU+DVFS | 256 | 700 MHz | 50 GB/s | 1.4 pJ |

Custom architectures can be added in the UI (any number).

---

## Input Shape

Set **B × C × H × W** to match the model's expected input:

| Task | Typical shape |
|------|--------------|
| Classification | `1 × 3 × 224 × 224` |
| Segmentation | `1 × 3 × 256 × 256` (H, W divisible by 8 for UNet-style) |
| Object Detection | `1 × 3 × 320 × 320` or `640 × 640` for YOLO |

---

## Test Models

Run once to generate sample weight files:

```bash
conda activate veloarch
python generate_test_models.py
```

| File | Code file | Input shape |
|------|-----------|-------------|
| `test_models/classification/resnet18_full.pt` | — | `1,3,224,224` |
| `test_models/classification/simple_clf.pth` | `simple_clf.py` | `1,3,224,224` |
| `test_models/semantic/simple_unet.pth` | `simple_unet.py` | `1,3,256,256` |
| `test_models/obj_det/yolov8n.pt` | — | `1,3,320,320` |
| `test_models/obj_det/simple_det.pth` | `simple_det.py` | `1,3,320,320` |

---

## API

`POST /api/evaluate` — multipart form

| Field | Type | Description |
|-------|------|-------------|
| `config` | JSON string | `{ workloads: [...], architectures: [...] }` |
| `model_file_0` | file | Weights for workload 0 |
| `code_file_0` | file (optional) | Class definition for workload 0 |
| `model_file_1` | file | Weights for workload 1 (if multiple) |

Returns:
```json
{
  "results": [
    {
      "workload": "ResNet18",
      "architecture": "NPU",
      "latency_ms": 5.36,
      "energy_mj": 5.52,
      "throughput_ops": 1.024e12,
      "bottleneck": "compute",
      "arithmetic_intensity": 554.7,
      "ops": 5.49e9,
      "memory_mb": 9.9
    }
  ],
  "workloads": [...],
  "architectures": [...]
}
```

---

## Project Structure

```
app.py                      Flask web app (port 5000)
src/
  engine.py                 Core engine — loader, FLOPs counter, roofline model
templates/
  index.html                Single-page UI (Bootstrap 5 + Chart.js)
test_models/
  classification/           resnet18_full.pt · simple_clf.pth + .py
  semantic/                 simple_unet.pth + .py
  obj_det/                  yolov8n.pt · simple_det.pth + .py
generate_test_models.py     Script to create all test weight files
```

---

## Requirements

```
torch · torchvision · ultralytics · flask
```

Install: `pip install torch torchvision ultralytics flask`
