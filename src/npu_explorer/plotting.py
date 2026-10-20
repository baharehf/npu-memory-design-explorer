"""Generate clean labeled plots with an explicit assumptions disclaimer."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_capacity_sweep(results: list[dict], figure_dir: Path) -> None:
    grouped: dict[str, list[dict]] = {}
    for item in results:
        grouped.setdefault(item["architecture"], []).append(item)

    charts = [
        ("estimated_latency_ms", "Estimated latency (ms)", "latency_vs_capacity.png"),
        ("estimated_energy_mj", "Estimated energy (mJ)", "energy_vs_capacity.png"),
        ("dram_bytes", "External memory traffic (KB)", "dram_traffic_vs_capacity.png"),
        ("memory_area_mm2", "Memory macro area (mm²)", "memory_area_vs_capacity.png"),
    ]
    for key, ylabel, filename in charts:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.2, top=0.87)
        for name, rows in grouped.items():
            ordered = sorted(rows, key=lambda row: row["capacity_kib"])
            ys = [row[key] / 1000 if key == "dram_bytes" else row[key] for row in ordered]
            ax.plot([row["capacity_kib"] for row in ordered], ys, marker="o", linewidth=2, label=name)
        ax.set_title(ylabel + " across on-chip memory capacities", pad=14)
        ax.set_xlabel("On-chip memory capacity (KiB)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(title="Hypothetical memory configuration")
        fig.text(0.5, 0.035, "Illustrative analytical model; not measured device characteristics", ha="center", fontsize=9)
        fig.savefig(figure_dir / filename, dpi=170, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
