import matplotlib.pyplot as plt
import numpy as np

from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    tgt_output_from_dict,
)


if __name__ == "__main__":
    device = "cuda"

    N = 10
    beta_count = 5
    samples = 200
    set_size = 25

    file = np.load(
        f"results/mcmpy/closest_border/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}.npz"
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
    n_betas = len(betas)
    ncols = min(3, n_betas)
    nrows = int(np.ceil(n_betas / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)  # flatten in case of 2D array

    bins = np.arange(min(map(np.min, all_border_dists)),
                     max(map(np.max, all_border_dists)) + 1)

    for ax, beta, dists in zip(axes, betas, all_border_dists):
        ax.hist(dists, bins=bins, density=True, alpha=0.7, color="C0")
        ax.set_title(f"$\\beta$={beta}")
        ax.set_xlabel("Border distance")
        ax.grid(True, linestyle="--", alpha=0.5)

    # Remove unused subplots if betas don't fill the grid
    for ax in axes[len(betas):]:
        ax.remove()

    axes[0].set_ylabel("Density")

    fig.suptitle(
        f"ds:{N, beta_count, samples, set_size}",
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("results/pma/hist_grid_border_sub_mcm")
    plt.close()
