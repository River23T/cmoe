"""Plot t-SNE of MoE gate activations across terrains — paper Fig. 2.

Usage:
    python plot_tsne_fig2.py /path/to/tsne_data.npz \
        --output fig2_cmoe.png \
        --perplexity 30

Recreates the right panel of paper Fig. 2 (CMoE Ours).

The .npz file is produced by evaluate_table3.py --export_tsne ...
It contains:
  - gates: [N, num_experts] gate activation weights
  - labels: [N] integer label = terrain_type * 100 + level
  - terrain_names: [num_terrains] array of column names

This script does NOT require Isaac Lab. Run it on any machine with:
    pip install scikit-learn matplotlib numpy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz_path", type=str)
    p.add_argument("--output", type=str, default="fig2_cmoe.png")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--max_points", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    gates = data["gates"]
    labels = data["labels"]
    terrain_names = list(data["terrain_names"])

    print(f"Loaded {len(gates)} samples, {gates.shape[1]} experts, "
          f"{len(terrain_names)} terrain types.")

    # subsample
    if len(gates) > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(gates), args.max_points, replace=False)
        gates = gates[idx]
        labels = labels[idx]
        print(f"Subsampled to {len(gates)} points.")

    # decode labels: terrain_type, level
    terrain_id = labels // 100
    level_id = labels % 100

    print("Running t-SNE...")
    embed = TSNE(n_components=2,
                 perplexity=args.perplexity,
                 random_state=args.seed,
                 init="pca",
                 learning_rate="auto").fit_transform(gates)
    print("Done.")

    # Color per terrain type, marker per difficulty bucket
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 8))

    n_terrains = len(terrain_names)
    legend_handles = []

    # Group level into easy / medium / hard markers
    level_max = level_id.max() if len(level_id) > 0 else 1
    def level_marker(lvl):
        frac = lvl / max(level_max, 1)
        if frac < 0.33: return "o"   # easy = circle
        elif frac < 0.66: return "s" # med = square
        else: return "^"             # hard = triangle

    for t in range(n_terrains):
        mask = terrain_id == t
        if mask.sum() == 0:
            continue
        # plot easy/med/hard with different markers but same color
        for marker, lo, hi in [("o", 0.0, 0.33),
                                ("s", 0.33, 0.66),
                                ("^", 0.66, 1.01)]:
            sub = mask & (level_id / max(level_max, 1) >= lo) & \
                          (level_id / max(level_max, 1) < hi)
            if sub.sum() == 0:
                continue
            ax.scatter(embed[sub, 0], embed[sub, 1],
                       c=[cmap(t % 10)], s=8, alpha=0.55,
                       marker=marker, edgecolors="none")

        legend_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(t % 10), markersize=8,
                   label=terrain_names[t])
        )

    # add difficulty legend
    legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                 markerfacecolor="grey", markersize=8,
                                 label="easy"))
    legend_handles.append(Line2D([0], [0], marker="s", color="w",
                                 markerfacecolor="grey", markersize=8,
                                 label="medium"))
    legend_handles.append(Line2D([0], [0], marker="^", color="w",
                                 markerfacecolor="grey", markersize=8,
                                 label="hard"))

    ax.legend(handles=legend_handles, loc="best", fontsize=9, ncol=2)
    ax.set_title("CMoE Gate Activations — t-SNE (paper Fig. 2)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
