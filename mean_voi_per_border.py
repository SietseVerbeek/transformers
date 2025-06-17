import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    get_dataset,
    get_from_dict,
    reduce_batch_size,
)
from utils.tests import batch_normalized_vi
from utils.training import generate_predictions


if __name__ == "__main__":
    device = "cuda"
    model, c, logs = pma_from_config(device)

    N = c.N_sites

    beta_count = 20
    samples = 1000
    set_size = 50

    file = get_dataset(N, beta_count, samples, set_size)

    borders, betas = get_borders_betas(file)

    n = len(borders)
    cols = 3  # Choose the number of columns (you can tweak this)
    rows = math.ceil(n / cols)

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()  # Flatten in case of 2D array of axes

    for i, border in enumerate(borders):
        means = np.empty_like(betas)
        stds = np.empty_like(betas)
        ax = axes[i]

        for j, beta in enumerate(betas):
            src, tgt = get_from_dict(file, border, beta)
            batches = reduce_batch_size(src, tgt, 10)
            var_of_info = np.empty(0, dtype=np.float64)

            for src, tgt in batches:
                src = torch.from_numpy(src).to(device)
                tgt = torch.from_numpy(tgt).to(device)

                output = generate_predictions(model, src, device)

                fraction = (output == tgt).count_nonzero() / output.nelement()
                var_of_info = np.concat([var_of_info, batch_normalized_vi(output, tgt)])

            means[j] = var_of_info.mean()
            stds[j] = var_of_info.std()
        ax.errorbar(betas, means, yerr=stds)
        ax.set_title(f"border pos {border}")
        ax.set_xlabel("$\\beta$")
        ax.set_ylabel("VOI")

        # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # Or: axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"results/pma/mean_voi_beta_border_pos__{c.model_id}__{c.train_id}")
