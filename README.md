# VeloArch

## Overview
**VeloArch** is a lightweight Python framework for high-level analysis of AI accelerator architectures, focusing on **Vector Processing Units (VPUs)** and **Neural Processing Units (NPUs)**. 

It provides fast, approximate estimations of performance and energy consumption to support early-stage architectural decisions, without requiring full hardware implementation.

---

## What It Does
* **Models AI workloads:** Supports CNNs and matrix operations.
* **Estimates latency and throughput:** Provides timing predictions.
* **Evaluates energy consumption:** Calculates efficiency metrics.
* **Compares architectures:** Benchmarks VPU vs. NPU performance.
* **Analyzes trade-offs:** Visualizes performance-power balance.

---

## Use Case
Designed specifically for **automotive AI applications** where low latency and energy efficiency are critical:
* Object detection
* Sensor fusion
* ADAS and decision systems

---

## Approach
* **High-level analytical models:** Uses mathematical logic rather than hardware simulation.
* **Parameterized descriptions:** Easily adjust architecture specs.
* **Simple Python implementation:** Built for speed and flexibility.
* **Design Exploration:** Provides order-of-magnitude estimates for rapid prototyping.

---

## Project Structure
```text
src/
├── workload.py           # Workload definitions (CNNs, etc.)
├── architecture.py       # Hardware parameterization
├── performance_model.py  # Latency/throughput logic
├── power_model.py        # Energy/efficiency logic
├── analysis.py           # Comparative analysis tools
└── main.py               # Framework entry point