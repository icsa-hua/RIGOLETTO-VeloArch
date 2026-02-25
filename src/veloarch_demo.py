import torch
import torch.nn as nn
from torchvision.models import resnet18

# -----------------------------
# Workload class
# -----------------------------
class Workload:
    def __init__(self, name, operations, memory_bytes):
        self.name = name
        self.operations = operations
        self.memory_bytes = memory_bytes

    def __repr__(self):
        return f"<Workload {self.name}: {self.operations:.2e} ops, {self.memory_bytes/1e6:.2f} MB>"

# -----------------------------
# Architecture class
# -----------------------------
class Architecture:
    def __init__(self, name, parallel_units, frequency, energy_per_op, energy_per_byte, memory_bandwidth):
        self.name = name
        self.parallel_units = parallel_units
        self.frequency = frequency
        self.energy_per_op = energy_per_op
        self.energy_per_byte = energy_per_byte
        self.memory_bandwidth = memory_bandwidth

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def __repr__(self):
        return f"<Architecture {self.name}: {self.parallel_units} units @ {self.frequency} Hz>"

# -----------------------------
# Model Parser (hook-based)
# -----------------------------
class ModelParser:
    def __init__(self, model, input_shape=(1, 3, 224, 224)):
        self.model = model
        self.input_shape = input_shape

    def analyze(self):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        x = torch.zeros(self.input_shape, device=device, dtype=dtype)

        ops_total = 0
        mem_total = 0

        hooks = []

        def conv_hook(module, input, output):
            batch, cin, h, w = input[0].shape
            cout, _, kh, kw = module.weight.shape
            ops = batch * h * w * cin * cout * kh * kw
            mem = output.numel() * 4
            nonlocal ops_total, mem_total
            ops_total += ops
            mem_total += mem

        def linear_hook(module, input, output):
            batch, in_features = input[0].shape
            out_features = module.out_features
            ops = batch * in_features * out_features
            mem = output.numel() * 4
            nonlocal ops_total, mem_total
            ops_total += ops
            mem_total += mem

        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                hooks.append(layer.register_forward_hook(conv_hook))
            elif isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(linear_hook))

        with torch.no_grad():
            try:
                self.model(x)
            except Exception as e:
                print("Warning: model forward failed, partial ops counted:", e)

        for h in hooks:
            h.remove()

        return Workload("Parsed_Model", operations=ops_total, memory_bytes=mem_total)

# -----------------------------
# Performance & Power
# -----------------------------
class PerformanceModel:
    @staticmethod
    def latency(workload, arch):
        compute_rate = arch.parallel_units * arch.frequency
        compute_time = workload.operations / compute_rate if compute_rate > 0 else float('inf')
        mem_time = workload.memory_bytes / arch.memory_bandwidth if arch.memory_bandwidth > 0 else float('inf')
        return max(compute_time, mem_time)

    @staticmethod
    def throughput(arch):
        return arch.parallel_units * arch.frequency

class PowerModel:
    @staticmethod
    def energy(workload, arch):
        compute_energy = workload.operations * arch.energy_per_op
        memory_energy = workload.memory_bytes * arch.energy_per_byte
        return compute_energy + memory_energy

    @staticmethod
    def apply_dvfs(arch, freq_scale=1.0):
        new_arch = arch.clone()
        new_arch.frequency *= freq_scale
        new_arch.energy_per_op *= freq_scale
        return new_arch

# -----------------------------
# Evaluator
# -----------------------------
class Evaluator:
    def __init__(self, workloads, architectures):
        self.workloads = workloads
        self.architectures = architectures

    def run(self):
        results = []
        for w in self.workloads:
            for a in self.architectures:
                latency = PerformanceModel.latency(w, a)
                throughput = PerformanceModel.throughput(a)
                energy = PowerModel.energy(w, a)
                results.append({
                    "workload": w.name,
                    "architecture": a.name,
                    "latency": latency,
                    "throughput": throughput,
                    "energy": energy
                })
        return results

    def print_results(self, results):
        print("\nEvaluation Results:\n")
        for r in results:
            print(f"Workload: {r['workload']} | Arch: {r['architecture']}")
            print(f"  Latency   : {r['latency']:.6f} s")
            print(f"  Throughput: {r['throughput']:.2e} ops/s")
            print(f"  Energy    : {r['energy']:.2f} J\n")

# -----------------------------
# Main Demo
# -----------------------------
def main():
    architectures = [
        Architecture("VPU", 256, 1e9, 2.0, 5e-9, 50e9),
        Architecture("NPU", 1024, 1e9, 1.0, 3e-9, 100e9),
        PowerModel.apply_dvfs(Architecture("VPU", 256, 1e9, 2.0, 5e-9, 50e9), freq_scale=0.7)
    ]
    architectures[-1].name = "VPU_DVFS"

    workloads = []

    # -------------------
    # ResNet18 classification (random weights)
    # -------------------
    cls_model = resnet18()
    for p in cls_model.parameters():
        nn.init.normal_(p)
    cls_model.eval()
    workloads.append(ModelParser(cls_model, input_shape=(1,3,224,224)).analyze())
    print("ResNet18 loaded with random weights")

    # -------------------
    # Simple semantic segmentation (UNet-like)
    # -------------------
    class SimpleSeg(nn.Module):
        def __init__(self, in_ch=3, out_ch=21):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(32, out_ch, 3, padding=1)
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            return x

    seg_model = SimpleSeg()
    for p in seg_model.parameters():
        nn.init.normal_(p)
    seg_model.eval()
    workloads.append(ModelParser(seg_model, input_shape=(1,3,224,224)).analyze())
    print("Semantic segmentation model loaded with random weights")

    # -------------------
    # Optional YOLO (if path exists)
    # -------------------
    try:
        yolo_path = "models/obj_detection/yolov8n.pt"
        ckpt = torch.load(yolo_path, map_location="cpu", weights_only=False)
        if 'ema' in ckpt and ckpt['ema'] is not None:
            yolo_model = ckpt['ema']
        elif 'model' in ckpt and ckpt['model'] is not None:
            yolo_model = ckpt['model']
        else:
            yolo_model = None

        if yolo_model:
            yolo_model.eval()
            workloads.append(ModelParser(yolo_model, input_shape=(1,3,640,640)).analyze())
            print("YOLO model loaded and parsed")
    except Exception as e:
        print("YOLO model skipped:", e)

    # -------------------
    # Run evaluation
    # -------------------
    evaluator = Evaluator(workloads, architectures)
    results = evaluator.run()
    evaluator.print_results(results)

if __name__ == "__main__":
    main()