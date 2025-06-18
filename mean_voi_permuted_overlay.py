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

    fig_vi, axes_vi = plt.subplots()
    fig_nmi, axes_nmi = plt.subplots()

    for i, border in enumerate(borders):
        mean_vi = np.empty_like(betas)
        stds_vi = np.empty_like(betas)

        mean_nmi = np.empty_like(betas)
        stds_nmi = np.empty_like(betas)

        for j, beta in enumerate(betas):
            var_of_info = np.empty(0, dtype=np.float64)
            nmi = np.empty(0, dtype=np.float64)
            src, tgt = get_from_dict(file, border, beta)
            batches = reduce_batch_size(src, tgt, 10)
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

        axes_vi.plot(betas, mean_vi, label=f"pos {border}")
        axes_nmi.plot(betas, mean_nmi, label=f"pos {border}")

    axes_vi.set_xlabel("$\\beta$")
    axes_vi.set_ylabel("VOI")
    axes_vi.legend()
    axes_vi.set_title(
        f"ds:{N, beta_count, samples, set_size} permuted, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )
    fig_vi.savefig(
        f"results/pma/mean_voi_beta_permuted_overlay__{c.model_id}__{c.train_id}"
    )

    axes_nmi.set_xlabel("$\\beta$")
    axes_nmi.set_ylabel("VOI")
    axes_nmi.legend()
    axes_nmi.set_title(
        f"ds:{N, beta_count, samples, set_size} permuted, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )
    fig_nmi.savefig(
        f"results/pma/mean_nmi_beta_permuted_overlay__{c.model_id}__{c.train_id}"
    )
