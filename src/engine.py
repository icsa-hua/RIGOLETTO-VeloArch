"""
VeloArch analytical engine.
Estimates latency, throughput, and energy for any PyTorch model
on parameterised VPU/NPU-class architectures.

FLOPs convention: 2 ops per multiply-add (IEEE standard).

Model loading supports:
  - torch.save(model, path)            → full nn.Module
  - YOLO / ultralytics checkpoints     → checkpoint['model'] or ['ema']
  - torch.jit.save(scripted, path)     → TorchScript module
  - state-dict-only .pt/.pth           → requires code_path with class definition
  - user .py script + weights file     → code_path defines architecture, weights loaded in
"""

import copy
import importlib.util
import os
import sys
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Preset architectures
# ---------------------------------------------------------------------------

PRESET_ARCHITECTURES = {
    # All presets model 28 nm-class embedded automotive processors.
    # peak_compute = parallel_units × frequency  (MACs/s; 1 MAC = 2 FLOPs IEEE)
    # ridge_point  = peak_compute / bandwidth    (ops/byte; roofline transition)
    #
    # VPU: 256 MAC units, 1 GHz  → 256 GOPS, ridge ≈ 5.1 ops/byte
    "VPU": {
        "name": "VPU",
        "parallel_units": 256,
        "frequency": 1e9,
        "energy_per_op": 2.0e-12,    # 2 pJ/op  — 28 nm CMOS estimate
        "energy_per_byte": 5.0e-12,  # 5 pJ/B   — LPDDR4 access
        "memory_bandwidth": 50e9,    # 50 GB/s  — LPDDR4x dual-channel
    },
    # NPU: 1024 MAC units, 1 GHz → 1.024 TOPS, ridge ≈ 10.2 ops/byte
    "NPU": {
        "name": "NPU",
        "parallel_units": 1024,
        "frequency": 1e9,
        "energy_per_op": 1.0e-12,    # 1 pJ/op  — systolic array efficiency
        "energy_per_byte": 3.0e-12,  # 3 pJ/B   — wider memory bus
        "memory_bandwidth": 100e9,   # 100 GB/s — LPDDR5 dual-channel
    },
    # VPU+DVFS: same VPU core at 0.7 GHz with voltage scaled proportionally.
    # CMOS dynamic energy ∝ V² × f; with V ∝ f → E_op ∝ f².
    # At 0.7×: E_op = 2.0 × 0.7² = 0.98 pJ  (NOT 0.7× linear — that is wrong).
    "VPU_DVFS": {
        "name": "VPU+DVFS",
        "parallel_units": 256,
        "frequency": 0.7e9,          # 700 MHz
        "energy_per_op": 0.98e-12,   # 2.0 × 0.7² pJ — quadratic DVFS scaling
        "energy_per_byte": 5.0e-12,  # unchanged (memory bus voltage not scaled)
        "memory_bandwidth": 50e9,
    },
    # RTX 5060 (Blackwell GB206, desktop, released 2025).
    # Source: NVIDIA official + TechPowerUp specs.
    # CUDA cores: 3840 (30 SMs × 128). Each core executes FMA → 2 FP32 ops/cycle.
    # parallel_units = 3840 × 2 = 7680 so that peak_compute = parallel_units × freq
    # yields the correct FP32 TFLOPS figure (19.75 TFLOPS).
    # energy_per_op = TDP / peak_compute = 145 W / 19.75e12 ops/s ≈ 7.3 pJ/op.
    # energy_per_byte: GDDR7 interface, ~20 pJ/byte (literature estimate for GDDR7).
    "RTX5060": {
        "name": "RTX 5060",
        "parallel_units": 7680,       # 3840 CUDA cores × 2 FP32 ops/cycle (FMA)
        "frequency": 2.572e9,         # 2572 MHz boost clock
        "energy_per_op": 7.3e-12,     # 7.3 pJ/op  — 145 W TDP / 19.75 TOPS
        "energy_per_byte": 20.0e-12,  # 20 pJ/byte — GDDR7 interface estimate
        "memory_bandwidth": 448e9,    # 448 GB/s  — GDDR7 28 Gbps 128-bit bus
    },
}

# ---------------------------------------------------------------------------
# JSON-safe float helper — replaces inf/nan with None for browser compatibility
# ---------------------------------------------------------------------------

def _safe(x):
    """Return None for any non-finite float; pass-through everything else."""
    if isinstance(x, float) and (x != x or abs(x) == float("inf")):
        return None
    return x


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

class Workload:
    """Represents a characterised AI workload (FLOPs + memory footprint)."""

    def __init__(self, name: str, operations: float, memory_bytes: float,
                 task_type: str = "custom"):
        self.name = name
        self.operations = float(operations)
        self.memory_bytes = float(memory_bytes)
        self.task_type = task_type

    @property
    def arithmetic_intensity(self):
        """ops/byte — roofline model x-axis. None when memory is zero."""
        if self.memory_bytes == 0:
            return None
        return self.operations / self.memory_bytes

    @property
    def is_zero(self) -> bool:
        return self.operations == 0 and self.memory_bytes == 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "task_type": self.task_type,
            "operations": self.operations,
            "memory_bytes": self.memory_bytes,
            "memory_mb": self.memory_bytes / 1e6,
            "arithmetic_intensity": _safe(self.arithmetic_intensity),
            "is_zero": self.is_zero,
        }

    def __repr__(self):
        return (f"<Workload {self.name!r}: {self.operations:.2e} ops, "
                f"{self.memory_bytes/1e6:.2f} MB>")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class Architecture:
    """
    Analytical hardware model.

    Parameters
    ----------
    parallel_units : int
        Number of processing elements (MACs) in parallel.
    frequency : float
        Operating frequency in Hz.
    energy_per_op : float
        Energy per operation in Joules (e.g. 2e-12 = 2 pJ).
    energy_per_byte : float
        Energy per byte transferred in Joules.
    memory_bandwidth : float
        Peak memory bandwidth in bytes/s.
    """

    def __init__(self, name: str, parallel_units: float, frequency: float,
                 energy_per_op: float, energy_per_byte: float,
                 memory_bandwidth: float):
        self.name = name
        self.parallel_units = float(parallel_units)
        self.frequency = float(frequency)
        self.energy_per_op = float(energy_per_op)
        self.energy_per_byte = float(energy_per_byte)
        self.memory_bandwidth = float(memory_bandwidth)

    @property
    def peak_compute(self) -> float:
        return self.parallel_units * self.frequency

    @property
    def ridge_point(self):
        """Roofline ridge point in ops/byte. None when bandwidth is zero."""
        if self.memory_bandwidth == 0:
            return None
        return self.peak_compute / self.memory_bandwidth

    def clone(self):
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parallel_units": self.parallel_units,
            "frequency": self.frequency,
            "energy_per_op": self.energy_per_op,
            "energy_per_byte": self.energy_per_byte,
            "memory_bandwidth": self.memory_bandwidth,
            "peak_compute": self.peak_compute,
            "ridge_point": _safe(self.ridge_point),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Architecture":
        return cls(
            name=str(d["name"]),
            parallel_units=float(d["parallel_units"]),
            frequency=float(d["frequency"]),
            energy_per_op=float(d["energy_per_op"]),
            energy_per_byte=float(d["energy_per_byte"]),
            memory_bandwidth=float(d["memory_bandwidth"]),
        )

    @classmethod
    def from_preset(cls, preset_name: str) -> "Architecture":
        if preset_name not in PRESET_ARCHITECTURES:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(PRESET_ARCHITECTURES.keys())}"
            )
        return cls(**PRESET_ARCHITECTURES[preset_name])

    def __repr__(self):
        return (f"<Architecture {self.name!r}: {self.parallel_units:.0f} units "
                f"@ {self.frequency/1e9:.2f} GHz>")


# ---------------------------------------------------------------------------
# Model script helpers
# ---------------------------------------------------------------------------

def _exec_module_file(path: str):
    """Execute a .py file and return it as an importlib module object."""
    abs_path = os.path.abspath(path)
    script_dir = os.path.dirname(abs_path)
    spec = importlib.util.spec_from_file_location("_veloarch_user_model", abs_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, script_dir)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ValueError(f"Failed to execute model script: {exc}") from exc
    finally:
        if script_dir in sys.path:
            sys.path.remove(script_dir)
    return module


def _find_model_in_module(module) -> nn.Module:
    """
    Look for an nn.Module instance or subclass in the executed module.

    Priority order:
    1. get_model() function
    2. A variable named 'model'
    3. Any nn.Module instance at module scope
    4. An nn.Module subclass that can be instantiated with no arguments
    """
    # 1. get_model()
    fn = getattr(module, "get_model", None)
    if callable(fn):
        result = fn()
        if isinstance(result, nn.Module):
            return result

    # 2. variable named 'model'
    candidate = getattr(module, "model", None)
    if isinstance(candidate, nn.Module):
        return candidate

    # 3. any nn.Module instance in module namespace
    for name, obj in vars(module).items():
        if not name.startswith("_") and isinstance(obj, nn.Module):
            return obj

    # 4. nn.Module subclasses — try no-arg instantiation
    subclasses = [
        (name, obj) for name, obj in vars(module).items()
        if not name.startswith("_")
        and isinstance(obj, type)
        and issubclass(obj, nn.Module)
        and obj is not nn.Module
    ]
    errors = []
    for name, cls in subclasses:
        try:
            return cls()
        except TypeError as e:
            errors.append(f"  {name}: {e}")

    if subclasses:
        names = [n for n, _ in subclasses]
        hint = "\n".join(errors)
        raise ValueError(
            f"Found class(es) {names} but could not instantiate with no arguments:\n{hint}\n"
            f"Add this to your script:\n  model = {names[0]}(<your constructor args>)"
        )

    raise ValueError(
        "No nn.Module found in the script.\n"
        "Add one of:\n"
        "  model = YourModel()            # preferred\n"
        "  def get_model(): return YourModel()"
    )


def _extract_state_dict(ckpt) -> dict:
    """
    Extract a flat state dict (str → Tensor) from various checkpoint formats.
    Returns None if no state dict can be found.
    """
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    if not isinstance(ckpt, dict):
        return None

    # Named state dict inside checkpoint
    for key in ("state_dict", "model_state_dict", "model_state", "net", "net_state_dict"):
        val = ckpt.get(key)
        if isinstance(val, nn.Module):
            return val.state_dict()
        if isinstance(val, dict) and val and all(isinstance(v, torch.Tensor) for v in val.values()):
            return val

    # For YOLO checkpoints, extract from model key
    for key in ("ema", "model"):
        val = ckpt.get(key)
        if isinstance(val, nn.Module):
            return val.state_dict()

    # Pure state dict
    if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    return None


# ---------------------------------------------------------------------------
# Model loader — handles any .pt / .pth file, optional .py code file
# ---------------------------------------------------------------------------

def load_model_from_path(weights_path: str, code_path: str = None):
    """
    Load an nn.Module from a weights file, optionally paired with a class script.

    Parameters
    ----------
    weights_path : str
        Path to a .pt or .pth file.  Can be:
          • torch.save(model, path)              → full nn.Module
          • YOLO / ultralytics checkpoint         → has 'model' or 'ema' key
          • torch.jit.save(scripted, path)        → TorchScript module
          • torch.save(model.state_dict(), path)  → requires code_path
    code_path : str, optional
        Path to a .py file that defines the model class.
        Required when weights_path contains only a state dict.
        The script must expose the model in one of these ways:
          • model = YourModel(...)                ← preferred
          • def get_model(): return YourModel(...)
          • class YourModel(nn.Module): ...       (no-arg __init__)

    Returns
    -------
    (model: nn.Module, source_description: str)
    """
    if code_path is not None:
        # --- User-supplied architecture + weights ---
        module = _exec_module_file(code_path)
        model = _find_model_in_module(module)

        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(f"Could not load weights file: {exc}") from exc

        state_dict = _extract_state_dict(ckpt)
        if state_dict is None:
            raise ValueError(
                "Could not extract a state dict from the weights file. "
                "Supported formats: torch.save(model, path), torch.save(model.state_dict(), path), "
                "or YOLO-style checkpoint."
            )

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model, "user script + weights"

    # --- Auto-detect mode (no code file) ---

    # TorchScript
    try:
        model = torch.jit.load(weights_path, map_location="cpu")
        model.eval()
        return model, "TorchScript"
    except Exception:
        pass

    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    except (AttributeError, ModuleNotFoundError) as exc:
        raise ValueError(
            f"Could not load model — the class definition was not found at load time.\n"
            f"This happens when the model was saved with torch.save(model, path) and the "
            f"class is not from an installed package (e.g. torchvision, ultralytics).\n"
            f"Fix: also upload the Model Class Code (.py) file alongside the weights.\n"
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not load file: {exc}") from exc

    # Full nn.Module saved directly
    if isinstance(ckpt, nn.Module):
        ckpt.eval()
        return ckpt, "nn.Module (full model)"

    if isinstance(ckpt, dict):
        # YOLO / ultralytics — has nn.Module at 'ema' or 'model' key
        for key in ("ema", "model"):
            candidate = ckpt.get(key)
            if isinstance(candidate, nn.Module):
                candidate.eval()
                return candidate, f"checkpoint['{key}']"

        # Pure state dict — architecture unknown
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            raise ValueError(
                "The file contains only a state dict (weights without architecture).\n"
                "Please also upload a Model Class Code (.py) file that defines your model.\n"
                "The script should have one of:\n"
                "  model = YourModel()\n"
                "  def get_model(): return YourModel()"
            )

        raise ValueError(
            f"Checkpoint is a dict but no nn.Module was found. "
            f"Keys: {list(ckpt.keys())[:10]}"
        )

    raise ValueError(
        f"Unexpected checkpoint type: {type(ckpt).__name__}. "
        "Expected an nn.Module or a dict-based checkpoint."
    )


# ---------------------------------------------------------------------------
# Model parser — hook-based FLOPs counter, works on any nn.Module
# ---------------------------------------------------------------------------

class ModelParser:
    """
    Runs a single forward pass with zero-filled input and counts FLOPs + memory
    via hooks on Conv1d, Conv2d, Conv3d, and Linear layers.

    Handles:
    - Standard classification models (ResNet, EfficientNet, VGG, …)
    - Segmentation models that return dicts (DeepLab, FCN, …)
    - Detection models — partial count if post-processing fails on zero input
    - float16 / bfloat16 models (dtype auto-matched)
    - TorchScript modules (hooks may not fire; returns zero ops with warning)
    """

    def __init__(self, model: nn.Module, input_shape: tuple = (1, 3, 224, 224)):
        self.model = model
        self.input_shape = tuple(int(x) for x in input_shape)

    def analyze(self) -> Workload:
        device = torch.device("cpu")
        dtype = torch.float32

        # Match device and dtype to the model's parameters
        try:
            p = next(self.model.parameters())
            device = p.device
            dtype = p.dtype
        except StopIteration:
            pass  # model has no parameters (e.g. pure functional)

        # float16/bfloat16 forward passes can be numerically unstable on CPU;
        # cast to float32 for the zero-input probe, then restore on error
        probe_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        x = torch.zeros(self.input_shape, device=device, dtype=probe_dtype)

        # If the model parameters are fp16, temporarily cast to fp32 for analysis
        original_dtype = dtype
        cast_to_fp32 = probe_dtype != dtype
        if cast_to_fp32:
            self.model = self.model.float()

        ops_total = 0
        mem_total = 0
        hooks = []

        def _conv2d(module, inp, out):
            nonlocal ops_total, mem_total
            try:
                # Use OUTPUT spatial dims — input dims are wrong for strided/valid-padded convs
                b, cout, h_out, w_out = out.shape
                cin_per_group = module.weight.shape[1]   # in_channels / groups
                kh, kw = module.weight.shape[2], module.weight.shape[3]
                ops_total += 2 * b * h_out * w_out * cin_per_group * cout * kh * kw
                # weights + output activations (inputs counted as output of previous layer)
                mem_total += (module.weight.numel() + out.numel()) * 4
            except Exception:
                pass

        def _conv1d(module, inp, out):
            nonlocal ops_total, mem_total
            try:
                b, cout, l_out = out.shape
                cin_per_group = module.weight.shape[1]
                k = module.weight.shape[2]
                ops_total += 2 * b * l_out * cin_per_group * cout * k
                mem_total += (module.weight.numel() + out.numel()) * 4
            except Exception:
                pass

        def _conv3d(module, inp, out):
            nonlocal ops_total, mem_total
            try:
                b, cout, d_out, h_out, w_out = out.shape
                cin_per_group = module.weight.shape[1]
                kd, kh, kw = module.weight.shape[2], module.weight.shape[3], module.weight.shape[4]
                ops_total += 2 * b * d_out * h_out * w_out * cin_per_group * cout * kd * kh * kw
                mem_total += (module.weight.numel() + out.numel()) * 4
            except Exception:
                pass

        def _linear(module, inp, out):
            nonlocal ops_total, mem_total
            try:
                flat = inp[0].reshape(-1, inp[0].shape[-1])
                b = flat.shape[0]
                ops_total += 2 * b * module.in_features * module.out_features
                mem_total += (module.weight.numel() + out.numel()) * 4
            except Exception:
                pass

        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                hooks.append(layer.register_forward_hook(_conv2d))
            elif isinstance(layer, nn.Conv1d):
                hooks.append(layer.register_forward_hook(_conv1d))
            elif isinstance(layer, nn.Conv3d):
                hooks.append(layer.register_forward_hook(_conv3d))
            elif isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(_linear))

        with torch.no_grad():
            try:
                self.model(x)
            except Exception:
                pass  # partial counts are still informative

        for h in hooks:
            h.remove()

        # Restore original dtype if we temporarily cast
        if cast_to_fp32:
            try:
                self.model = self.model.to(original_dtype)
            except Exception:
                pass

        return Workload("parsed", ops_total, mem_total)


# ---------------------------------------------------------------------------
# Performance model
# ---------------------------------------------------------------------------

class PerformanceModel:
    """
    Roofline analytical model (Williams et al., 2009).

    Latency   = max(ops / peak_compute,  bytes / bandwidth)
    Attainable = min(peak_compute,  AI × bandwidth)    [ops/s]
    HW Util   = attainable / peak_compute               [0–1]
    Bound ratio = AI / ridge_point  (>1 compute-bound, <1 memory-bound)
    """

    @staticmethod
    def latency(workload: Workload, arch: Architecture) -> float:
        compute_time = (workload.operations / arch.peak_compute
                        if arch.peak_compute > 0 else float("inf"))
        mem_time = (workload.memory_bytes / arch.memory_bandwidth
                    if arch.memory_bandwidth > 0 else float("inf"))
        return max(compute_time, mem_time)

    @staticmethod
    def attainable(workload: Workload, arch: Architecture):
        """Roofline attainable performance in ops/s. None if AI unavailable."""
        ai = workload.arithmetic_intensity
        if ai is None:
            return None
        return min(arch.peak_compute, ai * arch.memory_bandwidth)

    @staticmethod
    def hw_utilization(workload: Workload, arch: Architecture):
        """Fraction of peak compute achievable under the roofline (0–1)."""
        att = PerformanceModel.attainable(workload, arch)
        if att is None or arch.peak_compute == 0:
            return None
        return att / arch.peak_compute

    @staticmethod
    def bound_ratio(workload: Workload, arch: Architecture):
        """
        AI / ridge_point.
        > 1 → compute-bound (higher = deeper into compute roof).
        < 1 → memory-bound  (lower  = deeper into memory roof).
        """
        rp = arch.ridge_point
        ai = workload.arithmetic_intensity
        if rp is None or ai is None or rp == 0:
            return None
        return ai / rp

    @staticmethod
    def bottleneck(workload: Workload, arch: Architecture) -> str:
        rp = arch.ridge_point
        ai = workload.arithmetic_intensity
        if rp is None or ai is None:
            return "unknown"
        return "compute" if ai >= rp else "memory"


# ---------------------------------------------------------------------------
# Power model
# ---------------------------------------------------------------------------

class PowerModel:

    @staticmethod
    def energy(workload: Workload, arch: Architecture) -> float:
        return (workload.operations * arch.energy_per_op
                + workload.memory_bytes * arch.energy_per_byte)

    @staticmethod
    def apply_dvfs(arch: Architecture, freq_scale: float = 1.0) -> Architecture:
        new_arch = arch.clone()
        new_arch.frequency *= freq_scale
        # CMOS dynamic power P = αCV²f; with V ∝ f → E_op = P/f = αCV² ∝ f²
        new_arch.energy_per_op *= freq_scale ** 2
        return new_arch


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:

    def __init__(self, workloads: list, architectures: list):
        self.workloads = workloads
        self.architectures = architectures

    def run(self) -> list:
        results = []
        for w in self.workloads:
            for a in self.architectures:
                latency   = PerformanceModel.latency(w, a)
                energy    = PowerModel.energy(w, a)
                bottleneck = PerformanceModel.bottleneck(w, a)
                att       = PerformanceModel.attainable(w, a)
                hw_util   = PerformanceModel.hw_utilization(w, a)
                b_ratio   = PerformanceModel.bound_ratio(w, a)

                # Effective throughput = ops actually processed per second.
                # This is workload-dependent (unlike peak_compute which is constant).
                # For compute-bound: eff_throughput ≈ peak_compute.
                # For memory-bound:  eff_throughput = AI × bandwidth < peak_compute.
                eff_tput = w.operations / latency if latency > 0 else float("inf")

                results.append({
                    "workload":    w.name,
                    "task_type":   w.task_type,
                    "architecture": a.name,
                    # ── latency ──────────────────────────────────────
                    "latency_s":   _safe(latency),
                    "latency_ms":  _safe(latency * 1000),
                    # ── throughput ────────────────────────────────────
                    # effective = ops/latency  (varies per workload+arch)
                    # peak      = parallel_units × frequency  (arch property only)
                    "throughput_ops":      _safe(eff_tput),
                    "peak_compute_ops":    _safe(a.peak_compute),
                    # ── energy ───────────────────────────────────────
                    "energy_j":    _safe(energy),
                    "energy_mj":   _safe(energy * 1000),
                    # ── roofline metrics ──────────────────────────────
                    "bottleneck":           bottleneck,
                    "bound_ratio":          _safe(b_ratio),
                    "hw_utilization":       _safe(hw_util),
                    "attainable_ops":       _safe(att),
                    "arithmetic_intensity": _safe(w.arithmetic_intensity),
                    "ridge_point":          _safe(a.ridge_point),
                    # ── workload characteristics ──────────────────────
                    "ops":           w.operations,
                    "memory_mb":     w.memory_bytes / 1e6,
                    "peak_compute":  _safe(a.peak_compute),
                    "partial_analysis": w.is_zero,
                })
        return results
