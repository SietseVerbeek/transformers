import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    get_dataset,
    src_tgt_from_dict,
    reduce_batch_size,
)
from utils.tests import batch_normalized_mi, batch_normalized_vi
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
    fig_vi, axes_vi = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes_vi = axes_vi.flatten()  # Flatten in case of 2D array of axes

    fig_nmi, axes_nmi = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes_nmi = axes_nmi.flatten()  # Flatten in case of 2D array of axes

    for i, border in enumerate(borders):
        mean_vi = np.empty_like(betas)
        stds_vi = np.empty_like(betas)

        mean_nmi = np.empty_like(betas)
        stds_nmi = np.empty_like(betas)

        ax_vi = axes_vi[i]
        ax_nmi = axes_vi[i]

        for j, beta in enumerate(betas):
            src, tgt = src_tgt_from_dict(file, border, beta)
            batches = reduce_batch_size(src, tgt, 10)
            var_of_info = np.empty(0, dtype=np.float64)
            nmi = np.empty(0, dtype=np.float64)

            for src, tgt in batches:
                src = torch.from_numpy(src).to(device)
                tgt = torch.from_numpy(tgt).to(device)

                output = generate_predictions(model, src, device)

                fraction = (output == tgt).count_nonzero() / output.nelement()
                var_of_info = np.concat([var_of_info, batch_normalized_vi(output, tgt)])
                nmi = np.concat([nmi, batch_normalized_mi(output, tgt)])

            mean_vi[j] = var_of_info.mean()
            stds_vi[j] = var_of_info.std()

            mean_nmi[j] = var_of_info.mean()
            stds_nmi[j] = var_of_info.std()

        ax_vi.errorbar(betas, mean_vi, yerr=stds_vi)
        ax_vi.set_title(f"border pos {border}")
        ax_vi.set_xlabel("$\\beta$")
        ax_vi.set_ylabel("VOI")

        ax_nmi.errorbar(betas, mean_nmi, yerr=stds_nmi)
        ax_nmi.set_title(f"border pos {border}")
        ax_nmi.set_xlabel("$\\beta$")
        ax_nmi.set_ylabel("NMI")

        # Hide any unused subplots
    for j in range(i + 1, len(axes_vi)):
        fig_vi.delaxes(axes_vi[j])  # Or: axes[j].axis('off')
        fig_nmi.delaxes(axes_nmi[j])  # Or: axes[j].axis('off')

    fig_vi.suptitle(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )
    fig_vi.tight_layout()
    fig_vi.savefig(f"results/pma/mean_voi_beta_border_pos__{c.model_id}__{c.train_id}")

    fig_nmi.suptitle(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )
    fig_nmi.tight_layout()
    fig_nmi.savefig(f"results/pma/mean_nmi_beta_border_pos__{c.model_id}__{c.train_id}")
