# NPU Memory Hierarchy Design Space Explorer

![Development stage](https://img.shields.io/badge/status-educational%20prototype-blue)

A reproducible **Python analytical simulation** for exploring how on-chip memory capacity, access bandwidth, and energy assumptions impact estimated performance and energy for a tiled matrix-multiplication (GEMM) workload. The goal is to practice *architecture design-space exploration* relevant to edge-AI accelerators.

> **Scope and scientific caution:** This is a simplified educational model, **not** an RTL simulator, a cycle-accurate NPU simulator, or a validated SRAM/eDRAM model. Every memory-technology value in `configs/baseline.json` is deliberately **hypothetical**, not a measurement or manufacturer specification. In particular, the example charts **cannot establish that eDRAM outperforms SRAM** on any particular physical device.

## Research questions

1. How does increasing on-chip buffer capacity change data transferred to external DRAM for a fixed GEMM workload?
2. When is performance limited by arithmetic throughput versus DRAM bandwidth versus modeled on-chip bandwidth?
3. How do assumed SRAM/eDRAM energy, area, and refresh parameters affect *estimated* trade-offs?
4. How sensitive are the conclusions to the chosen model parameters?

## Quick start (Python 3.10+)

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
npu-explore
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1` (after creating the virtual environment).

The command writes `results/capacity_sweep.csv` and four charts into `figures/`. To use another configuration:

```bash
npu-explore --config configs/baseline.json --results results --figures figures
```

## Modeling approach (version 0.1)

We compute dense `C[M,N] = A[M,K] @ B[K,N]` with `M × N × K` multiply-accumulates (MACs). The simulator chooses the largest power-of-two **square tile** `T` that fits three `T × T` arrays in a single modeled buffer:

`buffer_bytes_required = T² × (2 × bytes_per_element + accumulator_bytes)`

The default model uses 1-byte INT8-style operands and 4-byte INT32-style accumulators. For each output tile, intermediate `C` values remain on-chip across the K reduction. The simulator counts reads of A and B for each K tile and one final DRAM write of each output element. Cross-output-tile cache reuse is **not** modeled. Boundary tiles (for non-multiple dimensions) are handled exactly.

Approximate on-chip traffic: `onchip_bytes = 2 × MACs × bytes_per_element + M × N × accumulator_bytes` (two operand reads per MAC and a final C write). This omits partial-sum buffer traffic and control/memory metadata.

The idealized roofline-style runtime is:

`latency_seconds = max(MACs / compute_MACs_per_s, DRAM_bytes / DRAM_bytes_per_s, onchip_bytes / onchip_bytes_per_s)`

**Assumption:** perfect overlap, no kernel startup overhead, and no timing/resource contention beyond these three ceilings. Energy is summed from MAC operations, external memory bytes, modeled on-chip bytes, and time-dependent static plus refresh power:

`energy_mJ = (MACs × pJ_per_MAC + DRAM_bytes × DRAM_pJ_per_byte + onchip_bytes × onchip_pJ_per_byte) / 1e9 + (static_mW + refresh_mW) × latency_seconds`

The calculated `memory_area_mm2` represents **only a hypothetical on-chip memory macro**, not total NPU silicon area. This model does not implement a real SRAM/eDRAM timing model, process-technology dependence, lifetime/reliability, device-level refresh, or an actual physical design. Neither physical accuracy nor a hardware speedup is claimed.

## Additional Experiments

The following experiments investigate memory bandwidth sensitivity
and performance bottlenecks in the analytical NPU model.

### 1. Automatic DRAM Bandwidth Exploration

Evaluate SRAM and eDRAM configurations across multiple external
DRAM bandwidths (3, 6, 12, 24, and 48 GB/s).

```bash
uv run python experiments/bandwidth_sweep.py
```

**Output:**
- `results/bandwidth_sweep.csv`
- `figures/bottleneck_heatmap_sram.png`
- `figures/bottleneck_heatmap_edram.png`

### 2. DRAM Bandwidth Comparison

Compare execution latency at 6 GB/s and 12 GB/s for different
on-chip memory capacities.

```bash
uv run python experiments/compare_bandwidth.py
```

**Output:**
- `results/bandwidth_comparison.csv`
- `figures/bandwidth_comparison_sram.png`
- `figures/bandwidth_comparison_edram.png`

**Observation:** Under the analytical model, larger on-chip
memory capacities generally reduce sensitivity to external
DRAM bandwidth.

### 3. On-Chip Memory Bandwidth Demonstration

Investigate the transition from compute-bound to on-chip
memory-bound execution using hypothetical memory configurations.

```bash
uv run npu-explore \
  --config configs/onchip_bandwidth_demo.json \
  --results results/onchip_demo \
  --figures figures/onchip_demo
```

**Observation:** With the selected assumptions, the SRAM
configuration is compute-bound while the eDRAM configuration
becomes limited by on-chip memory bandwidth.

### Model Limitations

All results are analytical estimates based on hypothetical
hardware parameters. They are not physical hardware measurements
or predictions of actual NXP processors.

## Output

| Figure | Question it helps explore |
| --- | --- |
| `latency_vs_capacity.png` | Does greater on-chip capacity reduce modeled latency? |
| `energy_vs_capacity.png` | What happens to modeled total energy? |
| `dram_traffic_vs_capacity.png` | How much data traffic does tiling require? |
| `memory_area_vs_capacity.png` | How much modeled memory macro area is used? |

All graphs are explicitly labeled as illustrative. `capacity_sweep.csv` is useful for reproducing findings in Jupyter or other tools.

## Project structure

```text
npu-memory-design-explorer/
├── configs/baseline.json           # Hypothetical hardware/workload parameters
├── src/npu_explorer/
│   ├── model.py                    # Analytical model
│   ├── cli.py                      # Experiment runner
│   └── plotting.py                 # Visualization
├── tests/test_model.py             # Unit tests and invariants
├── results/capacity_sweep.csv      # Generated example output
├── figures/*.png                   # Generated example graphs
├── pyproject.toml
└── README.md
```

## Next research steps

- Compare against a **simple reference implementation**, and check additional analytical limits.
- Implement multiple tiling/dataflow strategies and explicit buffer hierarchy levels.
- Sweep DRAM bandwidth, compute throughput, and data precision to locate bottlenecks.
- Run **sensitivity analysis** on the hypothetical SRAM/eDRAM parameters; use cited values where possible.
- Explore integration or cross-checks with Timeloop and Accelergy for better architecture mapping and energy estimation.

## Attribution

Created as an independent learning and research-oriented software project for AI accelerator architecture exploration. The parameters are not NXP proprietary information and do not represent NXP hardware. Code released under the MIT license.
