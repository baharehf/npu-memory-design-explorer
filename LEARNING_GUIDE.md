# Learning guide: understand your NPU memory explorer

This document explains the *first* experiment so you can describe it confidently in a PhD interview.

## 1. What mathematical operation is modeled?

Dense matrix multiplication (GEMM): `C[M,N] = A[M,K] × B[K,N]`.

- `M` = number of output rows.
- `N` = number of output columns.
- `K` = length of the accumulation dimension.
- MAC count = `M × N × K`. For 256 × 256 × 256, this is **16,777,216 MACs**.
- One MAC (multiply-accumulate) is typically counted as two arithmetic operations when calculating FLOPS/TOPS; definitions vary by hardware context.

The default example uses 8-bit input operands and 32-bit accumulation/output in the simple model. It does **not** implement actual quantized arithmetic or an AI neural network.

## 2. What is *tiling*?

Rather than keep entire arrays on chip, compute submatrices called **tiles**. To process one output tile, keep its partial sums on chip while iterating across the reduction dimension. The larger the tile, the more input data can often be reused before it is fetched again. With more on-chip capacity, larger tiles may become possible.

We require space for one A tile, one B tile, and one C accumulator tile:

`tile_memory_bytes = T² × (2 × operand_bytes + accumulator_bytes)`

This is intentionally a simplified single-level buffer model; it does not account for double-buffering, metadata, or parallel banks.

## 3. Why can more memory help, and why can gains stop?

The same matrix elements may need to be loaded repeatedly when output tiles are small. Larger tiles can reduce external data transfer. But once the idealized compute rate becomes the longest of our modeled time estimates, reducing DRAM traffic further does not improve predicted latency. The capacity thresholds look "stepped" because this first model allows only power-of-two square tiles.

## 4. The three modeled time limits

- **Compute time:** number of MACs / maximum modeled MACs per second.
- **External memory time:** bytes transferred to DRAM / modeled DRAM bandwidth.
- **On-chip memory time:** modeled operand/output bytes / modeled on-chip bandwidth.

Assuming perfect overlap, estimated runtime is the **maximum** of these. Real hardware adds startup time, memory access scheduling, bank conflicts, queue contention, control overhead, and more.

## 5. What do the SRAM and eDRAM comparisons prove?

**Nothing about actual device superiority.** The SRAM/eDRAM labels identify two *hypothetical parameter sets*. In the example configuration, eDRAM is assigned different energy, static power, refresh power, access bandwidth, and area coefficients. The plots therefore illustrate how the calculation responds to assumptions, not how these technologies perform on a silicon chip.

A more scientific next experiment would sweep one parameter at a time (e.g. eDRAM access energy from 1 to 5 pJ/byte) and report when conclusions reverse. Published empirical parameters with clear fabrication-node caveats would be necessary for defensible technology comparisons.

## 6. Your first hands-on exercise

1. Open `configs/baseline.json`.
2. Change `dram_bandwidth_gbps` from `12.0` to `6.0`.
3. Run `npu-explore` again.
4. Compare `figures/latency_vs_capacity.png` with the original graph.
5. In your own words, answer: **At what memory capacity does computation become the bottleneck, if at all?**
6. Restore the original parameter, then change `macs_per_second` from `200000000000` to `100000000000` and repeat.
7. Add your observations in the project README once you understand the results.

## 7. Good next experiments

- Add a plot of latency *versus DRAM bandwidth* at a fixed capacity.
- Implement different dataflow schedules and show the change in off-chip traffic.
- Verify predicted traffic using a Python reference tiled-matrix-multiply that counts logical loads.
- Compare results for different `M`, `N`, `K`, input precision, and accumulator widths.
- Add a parameter sensitivity study and uncertainty ranges.

## 8. Questions to practice answering

1. Why do we count multiply-accumulates instead of simply counting Python loop iterations?
2. Why does on-chip capacity constrain tile size?
3. When does reducing DRAM energy improve total energy but not runtime?
4. Why is this not an NPU hardware simulator?
5. Why is it important to document the origin and uncertainty of memory-energy assumptions?

A good research presentation explains assumptions and limitations as carefully as the results.
