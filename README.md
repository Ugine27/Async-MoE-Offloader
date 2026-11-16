# Async MoE Offloader 🚀

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance PyTorch interception and dynamic offloading engine designed to run massive **Mixture of Experts (MoE)** models (e.g., `Qwen1.5-MoE-A2.7B`) on memory-constrained GPUs (such as edge cards with $\le$ 4GB VRAM).

---

## The Problem
Standard Mixture-of-Experts architectures load every single expert into GPU VRAM. Even quantized 2.7B–14B MoE models require 6–16 GB of dedicated GPU memory, causing immediate Out-of-Memory (OOM) failures on budget or edge GPUs.

## The Solution: Dynamic Asynchronous Offloading

```
+-------------------------------------------------------------+
|                          HOST RAM                           |
|  [ Expert 0 ]  [ Expert 1 ]  [ Expert 2 ] ... [ Expert N ]  |  <-- Page-Locked (Pinned)
+------------------------------+------------------------------+
                               |
                   PCIe Bus    | Asynchronous DMA Transfer
                  (Non-blocking| via CUDA Streams)
                               v
+-------------------------------------------------------------+
|                          GPU VRAM                           |
|  +-------------------------------------------------------+  |
|  | Base Model Layers + Active Router                     |  |
|  +-------------------------------------------------------+  |
|  | Pre-allocated Static VRAM Slots (LRU Cache Pool)      |  |
|  |   [ Slot 0: Active Exp A ]   [ Slot 1: Active Exp B ] |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
```

1. **Host CPU "Parking Lot"**: Stores all MoE expert weights in pinned CPU memory.
2. **GPU Static Slot Pool**: Pre-allocates a fixed number of execution slots directly in GPU VRAM to prevent allocation fragmentation and eliminate `torch.cuda.empty_cache()` latency stalls.
3. **LRU VRAM Cache Manager**: Dynamically tracks active expert recency, evicting the least-recently-used expert only when a cache miss occurs.
4. **Non-Blocking Asynchronous Streaming**: Uses dedicated PyTorch CUDA streams with event-based synchronization (`torch.cuda.Event.wait_stream`), overlapping tensor transfers over PCIe with kernel compute.
5. **Zero-Copy Pointer Swapping**: Hot-swaps underlying C++ tensor pointers into fixed slots during the forward pass instead of rewriting Python dictionary state dicts.

---

## Repository Structure

```
Async-MoE-Offloader/
├── moe_offloader/
│   ├── __init__.py
│   ├── cache_manager.py     # LRU cache manager with preallocated GPU buffer slots
│   ├── offloaded_block.py   # Drop-in replacement MoE layer with dynamic routing
│   └── patcher.py           # Model-level architecture patcher and interceptor
├── main.py                  # Benchmarking and performance profiling entrypoint
├── chat.py                  # Interactive CLI chat server with the patched model
├── safety_monitor.py        # Real-time background watchdog monitoring VRAM & RAM
├── PLAN.md                  # Comprehensive architectural blueprint & memory analysis
├── RUNNER.md                # Detailed execution manual & hardware guidelines
└── requirements.txt         # Pinned dependencies
```

---

## Performance & Optimization Highlights

- **Megatron-Style Pinned Memory**: Host tensors are locked into physical RAM (`pin_memory()`) enabling true direct memory access (DMA).
- **Zero Allocator Fragmentation**: Fixed VRAM slots prevent PyTorch caching allocator bloat during prolonged generation runs.
- **Hardware-Aware Safety Watchdog**: Multi-threaded safety monitor continuously polls hardware metrics to guarantee stability under peak load.

---

## Quickstart

### 1. Requirements
- Linux or Windows
- Python 3.9+
- NVIDIA GPU with CUDA support (tested down to 4GB VRAM)

### 2. Installation
```bash
git clone https://github.com/Ugine27/Async-MoE-Offloader.git
cd Async-MoE-Offloader

python3 -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 3. Run Benchmark
Execute the inference benchmark to verify execution within VRAM constraints:
```bash
python main.py
```

### 4. Interactive Chat
Launch interactive conversation using offloaded generation:
```bash
python chat.py
```

---

## Further Reading
- [Architecture & Mechanics Plan (PLAN.md)](PLAN.md)
- [Runner & Tuning Guide (RUNNER.md)](RUNNER.md)

---

## License
MIT License. Created for high-efficiency LLM inference on consumer hardware.
