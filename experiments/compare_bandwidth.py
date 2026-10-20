
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Define the project directory and output locations.
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FIGURES.mkdir(exist_ok=True)


def find_column(df, candidates):
    """Find a column using common alternative names."""
    columns = {name.strip().lower(): name for name in df.columns}

    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]

    raise ValueError(
        f"Could not identify a required column.\n"
        f"Available columns: {list(df.columns)}"
    )


def load_experiment(bandwidth):
    """Load and standardize the results of one experiment."""
    path = RESULTS / f"bandwidth_{bandwidth}" / "capacity_sweep.csv"

    df = pd.read_csv(path)

    memory = find_column(
    df, ["architecture", "memory_type", "memory",
         "technology", "memory_technology", "mem_type"]
    )
    capacity = find_column(
        df, ["capacity_kib", "memory_capacity_kib",
             "onchip_capacity_kib", "capacity"]
    )
    latency = find_column(
        df, ["latency_ms", "estimated_latency_ms",
             "total_latency_ms"]
    )

    return df[[memory, capacity, latency]].rename(
        columns={
            memory: "memory",
            capacity: "capacity_kib",
            latency: f"latency_{bandwidth}_ms",
        }
    )


# Load the two experimental datasets.
data_12 = load_experiment(12)
data_6 = load_experiment(6)

# Match results with the same memory technology and capacity.
comparison = pd.merge(
    data_12,
    data_6,
    on=["memory", "capacity_kib"],
    validate="one_to_one",
)

# Calculate the relative increase in execution latency.
comparison["latency_increase_pct"] = (
    (comparison["latency_6_ms"] / comparison["latency_12_ms"]) - 1
) * 100

comparison = comparison.sort_values(
    ["memory", "capacity_kib"]
)

# Save the numerical comparison for reproducibility.
comparison.to_csv(
    RESULTS / "bandwidth_comparison.csv",
    index=False,
)

print("\nBandwidth Comparison Results:\n")
print(comparison.to_string(index=False, float_format="%.3f"))


# Produce one graph per memory technology.
for memory, group in comparison.groupby("memory"):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        group["capacity_kib"],
        group["latency_12_ms"],
        marker="o",
        linewidth=2,
        label="12 GB/s",
    )

    ax.plot(
        group["capacity_kib"],
        group["latency_6_ms"],
        marker="s",
        linewidth=2,
        label="6 GB/s",
    )

    # Logarithmic scaling spaces capacities evenly.
    ax.set_xscale("log", base=2)
    capacities = group["capacity_kib"].tolist()
    ax.set_xticks(capacities)
    ax.set_xticklabels([str(value) for value in capacities])

    ax.set_ylim(bottom=0)
    ax.set_xlabel("On-chip memory capacity (KiB)")
    ax.set_ylabel("Estimated execution latency (ms)")
    ax.set_title(f"DRAM Bandwidth Sensitivity — {memory}")

    ax.grid(True, alpha=0.3)
    ax.legend(title="External DRAM bandwidth")

    fig.text(
        0.5, 0.01,
        "Illustrative analytical estimates — not hardware measurements",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.045, 1, 1))

    filename = f"bandwidth_comparison_{str(memory).lower()}.png"
    fig.savefig(FIGURES / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

print("\nComparison CSV and figures generated successfully.")
