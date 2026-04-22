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

| Name | Parallel Units | Frequency | Bandwidth | Energy/op | Notes |
|------|:--------------:|:---------:|:---------:|:---------:|-------|
| VPU | 256 | 1 GHz | 50 GB/s | 2 pJ | 28 nm embedded VPU, LPDDR4x |
| NPU | 1024 | 1 GHz | 100 GB/s | 1 pJ | Systolic-array NPU, LPDDR5 |
| VPU+DVFS | 256 | 700 MHz | 50 GB/s | 0.98 pJ | Same VPU at 0.7× freq, E∝f² |
| RTX 5060 | 7680 | 2.572 GHz | 448 GB/s | 7.3 pJ | Blackwell GB206, GDDR7, 145 W TDP |

`parallel_units` for RTX 5060 = 3840 CUDA cores × 2 FP32 ops/cycle (FMA), giving the published 19.75 TFLOPS.

Custom architectures can be added in the UI (any number).

---

## Architecture Inputs → Output Metrics

This section maps every accelerator input parameter to the outputs it drives.

### Input parameters

| Parameter | Unit | What it means |
|-----------|------|---------------|
| `parallel_units` | — | Processing elements active each clock cycle. For embedded VPUs/NPUs this is the MAC array width; for CUDA GPUs use `cores × 2` to account for FMA dual-issue. |
| `frequency` | Hz | Clock rate. Together with `parallel_units` this sets the ceiling on how fast arithmetic can run. |
| `energy_per_op` | J/op | Energy consumed per floating-point operation. Derived from TDP ÷ peak throughput. Reflects silicon technology node and micro-architecture efficiency. |
| `energy_per_byte` | J/byte | Energy per byte transferred from/to main memory. Reflects memory technology (LPDDR4 ≈ 5 pJ/B, LPDDR5 ≈ 3 pJ/B, GDDR7 ≈ 20 pJ/B). |
| `memory_bandwidth` | B/s | Peak sustained data rate from main memory to the accelerator. |

Two derived architecture constants are computed internally:

```
peak_compute  =  parallel_units × frequency          [ops/s]
ridge_point   =  peak_compute   / memory_bandwidth   [ops/byte]
```

### Output metrics and their formulas

**Latency**
```
latency = max( ops / peak_compute,  memory_bytes / bandwidth )   [s]
```
The roofline model takes the *maximum* of the compute time and the memory time — whichever is larger is the bottleneck. Increasing `parallel_units` or `frequency` shrinks the first term; increasing `memory_bandwidth` shrinks the second.

**Energy**
```
energy = ops × energy_per_op  +  memory_bytes × energy_per_byte   [J]
```
Two independent cost centres: arithmetic work and memory traffic. A memory-bound workload is dominated by the second term; compute-bound by the first.

**Effective Throughput**
```
eff_throughput = ops / latency   [ops/s]
```
The actual operations-per-second the workload achieves on this hardware. This is workload-dependent — it equals `peak_compute` only when the workload is perfectly compute-bound. A memory-bound workload reaches `arithmetic_intensity × bandwidth`, which is below peak.

**Bottleneck**
```
arithmetic_intensity (AI) = ops / memory_bytes   [ops/byte]

bottleneck = "compute"  if  AI ≥ ridge_point
             "memory"   if  AI <  ridge_point
```
`AI` (the x-axis of the roofline plot) measures how many operations the workload does per byte it moves. The `ridge_point` is the hardware's knee: above it the chip is compute-saturated; below it the memory bus is saturated.

**Bound Ratio**
```
bound_ratio = AI / ridge_point
```
The *distance* from the ridge point. `bound_ratio > 1` means compute-bound (the further above 1, the more compute-bound); `< 1` means memory-bound. Displayed in the UI as `compute ×N` or `memory ×N`.

**HW Utilization**
```
attainable      = min( peak_compute,  AI × bandwidth )   [ops/s]
hw_utilization  = attainable / peak_compute              [0–1]
```
The fraction of theoretical peak compute that can be achieved given the workload's arithmetic intensity. A compute-bound workload reaches 100 %; a memory-bound workload reaches `AI / ridge_point × 100 %`. This is the single number that tells you how well the hardware fits the workload.

### How to use these relationships

| Goal | Lever |
|------|-------|
| Reduce latency of a compute-bound workload | Increase `parallel_units` or `frequency` |
| Reduce latency of a memory-bound workload | Increase `memory_bandwidth` |
| Reduce energy without changing latency | Lower `energy_per_op` (better process node) or reduce `memory_bytes` (quantisation, pruning) |
| Push a workload from memory-bound to compute-bound | Increase arithmetic intensity — e.g. larger batch size, fused kernels, or tiling |
| Compare two architectures on the same workload | Look at `hw_utilization`: the architecture with higher utilisation is a better fit |

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
