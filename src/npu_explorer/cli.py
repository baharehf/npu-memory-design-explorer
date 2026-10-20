"""Run capacity sweeps and produce a CSV plus reproducible figures."""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from .model import Architecture, Hardware, Workload, estimate
from .plotting import plot_capacity_sweep


def run_experiment(config_path: Path, output_dir: Path, figure_dir: Path) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workload = Workload(**config["workload"])
    hardware = Hardware(**config["hardware"])
    capacities = config["capacity_sweep_kib"]
    if not capacities or any(int(size) != size or size <= 0 for size in capacities):
        raise ValueError("capacity_sweep_kib must contain positive integer sizes")
    results = []
    for arch_info in config["architectures"]:
        arch = Architecture(**arch_info)
        for size in capacities:
            results.append(estimate(workload, hardware, replace(arch, capacity_kib=int(size))))

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "capacity_sweep.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plot_capacity_sweep(results, figure_dir)
    print(f"Results: {csv_file}")
    print(f"Figures: {figure_dir.resolve()}")
    print(f"Workload: {workload.m}×{workload.k} @ {workload.k}×{workload.n}; "
          f"{workload.macs:,} MACs")
    print("IMPORTANT: These are analytical estimates using hypothetical parameters.")
    for row in results:
        print(f"  {row['architecture']:<6} {row['capacity_kib']:>4} KiB | "
              f"tile={row['tile_size']:>3} | "
              f"latency={row['estimated_latency_ms']:.4f} ms | "
              f"energy={row['estimated_energy_mj']:.5f} mJ")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Educational memory-centric NPU GEMM exploration (illustrative parameters)"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    run_experiment(args.config, args.results, args.figures)


if __name__ == "__main__":
    main()
