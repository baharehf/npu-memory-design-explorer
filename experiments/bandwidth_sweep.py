
"""Explore DRAM bandwidth sensitivity in a simplified NPU memory model.

This experiment evaluates SRAM and eDRAM configurations across different
on-chip memory capacities and external DRAM bandwidths.

All results are analytical estimates using hypothetical hardware parameters.
They are not measurements or predictions of actual semiconductor devices.
"""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from npu_explorer.model import Architecture, Hardware, Workload, estimate


# Locate the project root independently of the current working directory.
ROOT = Path(__file__).resolve().parents[1]

# Default DRAM bandwidths used in the design space exploration.
DEFAULT_BANDWIDTHS = [3, 6, 12, 24, 48]

# Possible limiting components of the analytical latency model.
BOTTLENECKS = ["external_dram", "compute", "on_chip_memory"]

LABELS = {
    "external_dram": "DRAM",
    "compute": "Compute",
    "on_chip_memory": "On-chip",
}

COLORS = ["#F4CDAD", "#BCE4CE", "#C9DBF6"]


def run_bandwidth_sweep(config, bandwidths):
    """Evaluate all combinations of architecture, capacity, and bandwidth."""

    workload = Workload(**config["workload"])
    base_hardware = Hardware(**config["hardware"])

    results = []

    for architecture_config in config["architectures"]:
        base_architecture = Architecture(**architecture_config)

        for bandwidth in bandwidths:
            hardware = replace(
                base_hardware,
                dram_bandwidth_gbps=bandwidth,
            )

            for capacity in config["capacity_sweep_kib"]:
                architecture = replace(
                    base_architecture,
                    capacity_kib=int(capacity),
                )

                # Reuse the existing analytical performance model.
                metrics = estimate(workload, hardware, architecture)

                results.append({
                    "dram_bandwidth_gbps": bandwidth,
                    **metrics,
                })

    return results


def save_results(results):
    """Save the complete experiment results to a reproducible CSV file."""

    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "bandwidth_sweep.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    return output_path


def plot_bottleneck_heatmaps(results, bandwidths, capacities):
    """Visualize the dominant performance bottleneck for each configuration."""

    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    architectures = sorted(set(row["architecture"] for row in results))
    cmap = ListedColormap(COLORS)

    for architecture in architectures:

        # Index results by bandwidth and memory capacity.
        lookup = {
            (row["dram_bandwidth_gbps"], row["capacity_kib"]): row
            for row in results
            if row["architecture"] == architecture
        }

        # Convert bottleneck names into integer categories for plotting.
        grid = [
            [
                BOTTLENECKS.index(
                    lookup[(bandwidth, capacity)]["bottleneck"]
                )
                for capacity in capacities
            ]
            for bandwidth in bandwidths
        ]

        fig, ax = plt.subplots(figsize=(10, 5.5))

        ax.imshow(
            grid,
            cmap=cmap,
            vmin=-0.5,
            vmax=2.5,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_xticks(range(len(capacities)))
        ax.set_xticklabels([str(c) for c in capacities])

        ax.set_yticks(range(len(bandwidths)))
        ax.set_yticklabels([f"{b:g}" for b in bandwidths])

        ax.set_xlabel("On-chip memory capacity (KiB)")
        ax.set_ylabel("External DRAM bandwidth (GB/s)")
        ax.set_title(f"Performance Bottleneck Map — {architecture}")

        # Label each cell so results do not depend on color alone.
        for row_index, row in enumerate(grid):
            for col_index, category in enumerate(row):
                ax.text(
                    col_index,
                    row_index,
                    LABELS[BOTTLENECKS[category]],
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        legend_items = [
            Patch(facecolor=color, label=LABELS[name])
            for name, color in zip(BOTTLENECKS, COLORS)
        ]

        fig.legend(
            handles=legend_items,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.025),
        )

        fig.text(
            0.5,
            0.015,
            "Hypothetical analytical model — not hardware measurements",
            ha="center",
            fontsize=8,
        )

        fig.tight_layout(rect=(0, 0.13, 1, 1))

        output_path = (
            figure_dir / f"bottleneck_heatmap_{architecture.lower()}.png"
        )

        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore NPU performance across DRAM bandwidths."
    )

    parser.add_argument(
        "--bandwidths",
        type=float,
        nargs="+",
        default=DEFAULT_BANDWIDTHS,
        help="DRAM bandwidth values in GB/s",
    )

    args = parser.parse_args()

    if not args.bandwidths or any(b <= 0 for b in args.bandwidths):
        parser.error("All bandwidth values must be positive.")

    config_path = ROOT / "configs" / "baseline.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    capacities = [int(c) for c in config["capacity_sweep_kib"]]
    bandwidths = sorted(set(args.bandwidths))

    results = run_bandwidth_sweep(config, bandwidths)

    csv_path = save_results(results)

    plot_bottleneck_heatmaps(results, bandwidths, capacities)

    print(f"\nEvaluated {len(results)} configurations.")
    print(f"Results saved to: {csv_path}")

    print("\nBottleneck summary:")
    for architecture in sorted(set(r["architecture"] for r in results)):
        for bandwidth in bandwidths:
            subset = [
                row for row in results
                if row["architecture"] == architecture
                and row["dram_bandwidth_gbps"] == bandwidth
            ]

            counts = {
                name: sum(row["bottleneck"] == name for row in subset)
                for name in BOTTLENECKS
            }

            print(f"{architecture:6s} | {bandwidth:5g} GB/s | {counts}")


if __name__ == "__main__":
    main()
