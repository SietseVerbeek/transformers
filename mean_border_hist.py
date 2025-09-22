import matplotlib.pyplot as plt
import numpy as np

from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    tgt_output_from_dict,
)


if __name__ == "__main__":
    device = "cuda"
    model, c, logs = pma_from_config(device)

    N = c.N_sites

    beta_count = 5
    samples = 200
    set_size = 25

    file = np.load(
        f"results/pma/predictions/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}__{c.model_id}__{c.train_id}.npz"
    )

    borders, betas = get_borders_betas(file)

    all_border_dists = []

    for beta in betas:
        border_dists = np.empty(0, dtype=np.float64)
        for border in borders:
            tgt, out = tgt_output_from_dict(file, border, beta)
            out_borders = np.argmax(out, axis=-1)
            tmp_border_dists = out_borders - (border % N)
            border_dists = np.concatenate([border_dists, tmp_border_dists])
        all_border_dists.append(border_dists)

    # ---- PLOT ----
    fig, axes = plt.subplots(1, len(betas), figsize=(4 * len(betas), 4), sharey=True)

    bins = np.arange(min(map(np.min, all_border_dists)),
                     max(map(np.max, all_border_dists)) + 1)

    if len(betas) == 1:
        axes = [axes]  # handle single β case

    for ax, beta, dists in zip(axes, betas, all_border_dists):
        ax.hist(dists, bins=bins, density=True, alpha=0.7, color="C0")
        ax.set_title(f"$\\beta$={beta}")
        ax.set_xlabel("Border distance")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Density")

    fig.suptitle(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs['samples_shown']} $\\beta$ {c.train_beta_temps}",
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"results/pma/hist_subplots_border_beta__{c.model_id}__{c.train_id}")
    plt.close()
