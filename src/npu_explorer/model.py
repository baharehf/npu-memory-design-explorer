"""Analytical model of GEMM execution with a tiled, on-chip memory buffer.

This is an educational design-space model, NOT a cycle-accurate simulator or
physical SRAM/eDRAM device model. Its parameters are intentionally hypothetical.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    m: int
    n: int
    k: int
    bytes_per_element: int = 1
    accumulator_bytes: int = 4

    def __post_init__(self) -> None:
        if min(self.m, self.n, self.k, self.bytes_per_element, self.accumulator_bytes) <= 0:
            raise ValueError("Workload dimensions and element size must be positive")

    @property
    def macs(self) -> int:
        """MACs required for dense C[M,N] = A[M,K] @ B[K,N]."""
        return self.m * self.n * self.k


@dataclass(frozen=True)
class Hardware:
    macs_per_second: float
    dram_bandwidth_gbps: float
    dram_energy_pj_per_byte: float
    mac_energy_pj: float

    def __post_init__(self) -> None:
        if min(self.macs_per_second, self.dram_bandwidth_gbps) <= 0:
            raise ValueError("Throughput and bandwidth must be positive")
        if min(self.dram_energy_pj_per_byte, self.mac_energy_pj) < 0:
            raise ValueError("Energy parameters cannot be negative")


@dataclass(frozen=True)
class Architecture:
    name: str
    capacity_kib: int
    onchip_bandwidth_gbps: float
    onchip_energy_pj_per_byte: float
    static_power_mw_per_mib: float
    refresh_power_mw_per_mib: float
    memory_area_mm2_per_mib: float

    def __post_init__(self) -> None:
        if self.capacity_kib <= 0 or self.onchip_bandwidth_gbps <= 0:
            raise ValueError("On-chip capacity and bandwidth must be positive")
        if min(
            self.onchip_energy_pj_per_byte,
            self.static_power_mw_per_mib,
            self.refresh_power_mw_per_mib,
            self.memory_area_mm2_per_mib,
        ) < 0:
            raise ValueError("Energy, power and area parameters cannot be negative")

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_kib * 1024


def choose_tile_size(workload: Workload, capacity_bytes: int) -> int:
    """Largest power-of-two square tile fitting A, B and C concurrently.

    A/B each use operand width; C uses accumulator width (e.g. INT32).
    No double-buffering; only a deliberately simplified occupancy model.
    """
    smallest = min(workload.m, workload.n, workload.k)
    candidate = 1
    chosen = 0
    while candidate <= smallest:
        if candidate * candidate * (2 * workload.bytes_per_element + workload.accumulator_bytes) <= capacity_bytes:
            chosen = candidate
        else:
            break
        candidate *= 2
    if chosen == 0:
        raise ValueError("On-chip capacity too small even for a 1x1 GEMM tile")
    return chosen


def dram_transfer_bytes(workload: Workload, tile_size: int) -> int:
    """Count A/B fetches and one output write for tiled GEMM.

    For every output tile (i,j), hold partial C sums on-chip across the full
    reduction dimension K; reload A and B for every different output tile.
    No cross-tile cache reuse, no C input read, and no coherence/metadata cost.
    Handles dimensions that are not multiples of tile_size.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    w = workload
    elements = 0
    for i in range(0, w.m, tile_size):
        mi = min(tile_size, w.m - i)
        for j in range(0, w.n, tile_size):
            nj = min(tile_size, w.n - j)
            for k in range(0, w.k, tile_size):
                kk = min(tile_size, w.k - k)
                elements += mi * kk + kk * nj
    # Completed C is written once, potentially at wider accumulator precision.
    operand_bytes = elements * w.bytes_per_element
    output_bytes = w.m * w.n * w.accumulator_bytes
    return operand_bytes + output_bytes


def estimate(workload: Workload, hardware: Hardware, architecture: Architecture) -> dict:
    """Return simplified roofline latency, energy and memory-only area.

    Model: total time = max(compute time, external data time, on-chip data time).
    Perfect overlap and no startup/launch cost are assumed. On-chip traffic
    counts two operand reads per MAC plus one final accumulator write;
    this excludes C partial-sum reads/writes and many real hardware accesses.
    Energy: per-MAC + external-byte + on-chip-byte + time-proportional power.
    Values are *estimates*, and do not predict any actual NXP hardware.
    """
    t = choose_tile_size(workload, architecture.capacity_bytes)
    d_bytes = dram_transfer_bytes(workload, t)
    onchip_bytes = (2 * workload.macs * workload.bytes_per_element
                    + workload.m * workload.n * workload.accumulator_bytes)

    compute_seconds = workload.macs / hardware.macs_per_second
    dram_seconds = d_bytes / (hardware.dram_bandwidth_gbps * 1e9)
    onchip_seconds = onchip_bytes / (architecture.onchip_bandwidth_gbps * 1e9)

    # Identify the latency contribution of each system component.
    latency_components = {
        "compute": compute_seconds,
        "external_dram": dram_seconds,
        "on_chip_memory": onchip_seconds,
    }

    bottleneck = max(latency_components, key=latency_components.get)

    latency_seconds = latency_components[bottleneck]

    dynamic_pj = (
        workload.macs * hardware.mac_energy_pj
        + d_bytes * hardware.dram_energy_pj_per_byte
        + onchip_bytes * architecture.onchip_energy_pj_per_byte
    )
    mib = architecture.capacity_bytes / (1024 * 1024)
    background_power_mw = mib * (
        architecture.static_power_mw_per_mib + architecture.refresh_power_mw_per_mib
    )
    energy_mj = dynamic_pj / 1e9 + background_power_mw * latency_seconds

    return {
        "architecture": architecture.name,
        "capacity_kib": architecture.capacity_kib,
        "tile_size": t,
        "macs": workload.macs,
        "dram_bytes": d_bytes,
        "onchip_bytes": onchip_bytes,
        "compute_latency_ms": compute_seconds * 1e3,
        "dram_latency_ms": dram_seconds * 1e3,
        "onchip_latency_ms": onchip_seconds * 1e3,
        "estimated_latency_ms": latency_seconds * 1e3,
        "bottleneck": bottleneck,
        "estimated_energy_mj": energy_mj,
        "memory_area_mm2": mib * architecture.memory_area_mm2_per_mib,
        "estimated_tops": (2 * workload.macs / latency_seconds) / 1e12,
    }
