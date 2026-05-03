"""Reproduce paper Fig. 5: expert activations over time on mixed terrain.

Usage:
    python plot_fig5_experts.py /path/to/gate_weights.csv \
        --output fig5_cmoe.png

Input CSV columns:
    t_seconds, x, y, z, terrain_type, terrain_level, expert_1, ..., expert_N

The script auto-detects terrain transitions (changes in terrain_type)
and draws vertical dashed lines + labels, mimicking paper Fig. 5.

Requirements: pip install matplotlib pandas numpy
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TERRAIN_NAMES = {
    0: "slope_up", 1: "slope_down", 2: "stair_up", 3: "stair_down",
    4: "gap", 5: "hurdle", 6: "discrete", 7: "mix1", 8: "mix2",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("--output", type=str, default="fig5_cmoe.png")
    p.add_argument("--smooth", type=int, default=5,
                   help="moving average window (steps); 1=no smoothing")
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    t = df["t_seconds"].values
    expert_cols = [c for c in df.columns if c.startswith("expert_")]
    gates = df[expert_cols].values  # [T, N]

    # smooth
    if args.smooth > 1:
        from scipy.ndimage import uniform_filter1d
        gates = uniform_filter1d(gates, size=args.smooth, axis=0, mode="nearest")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    cmap = plt.get_cmap("tab10")
    for i in range(gates.shape[1]):
        ax.plot(t, gates[:, i], label=f"Expert {i+1}",
                linewidth=1.8, color=cmap(i % 10), alpha=0.9)

    # mark terrain transitions
    if "terrain_type" in df.columns:
        ttypes = df["terrain_type"].values
        # find points where terrain changes
        transitions = np.where(np.diff(ttypes) != 0)[0]
        for tr in transitions:
            ax.axvline(t[tr+1], color="grey", linestyle="--", alpha=0.5, linewidth=1)
        # annotate each segment
        seg_start = 0
        for tr in list(transitions) + [len(t) - 1]:
            mid = (t[seg_start] + t[tr]) / 2
            tt = int(ttypes[seg_start])
            name = TERRAIN_NAMES.get(tt, f"t{tt}")
            ax.text(mid, 0.95, name, ha="center", fontsize=9,
                    color="dimgrey", style="italic",
                    transform=ax.get_xaxis_transform())
            seg_start = tr + 1

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Expert activation")
    ax.set_title("CMoE Expert Activations Across Terrain Transitions (paper Fig. 5)")
    ax.set_ylim(0, 1)
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
