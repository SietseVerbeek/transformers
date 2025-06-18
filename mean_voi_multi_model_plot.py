import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.pma import many_pma_from_config, pma_from_config
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
    models, c_list, logs_list = many_pma_from_config(device)

    N = c_list[0].N_sites

    beta_count = 20
    samples = 1000
    set_size = 50

    file = get_dataset(N, beta_count, samples, set_size)

    borders, betas = get_borders_betas(file)

    fig_vi, ax_vi = plt.subplots()
    fig_nmi, ax_nmi = plt.subplots()

    for model, c, logs in zip(models, c_list, logs_list):
        mean_vi = np.empty_like(betas)
        stds_vi = np.empty_like(betas)

        mean_nmi = np.empty_like(betas)
        stds_nmi = np.empty_like(betas)

        for i, beta in enumerate(betas):
            var_of_info = np.empty(0, dtype=np.float64)
            nmi = np.empty(0, dtype=np.float64)

            for border in borders:
                src, tgt = get_from_dict(file, border, beta)
                batches = reduce_batch_size(src, tgt, 10)

                for src, tgt in batches:
                    src = torch.from_numpy(src).to(device)
                    tgt = torch.from_numpy(tgt).to(device)

                    output = generate_predictions(model, src, device)

                    fraction = (output == tgt).count_nonzero() / output.nelement()
                    var_of_info = np.concat([var_of_info, batch_normalized_vi(output, tgt)])
                    nmi = np.concat([nmi, batch_normalized_mi(output, tgt)])

            mean_vi[i] = var_of_info.mean()
            stds_vi[i] = var_of_info.std()

            mean_nmi[i] = nmi.mean()
            stds_nmi[i] = nmi.std()


        ax_vi.plot(betas, mean_vi)
        ax_vi.set_xlabel("$\\beta$")
        ax_vi.set_ylabel("VOI")

        ax_nmi.plot(betas, mean_nmi)
        ax_nmi.set_xlabel("$\\beta$")
        ax_nmi.set_ylabel("NMI")

    fig_vi.suptitle(
        f"ds:{N, beta_count, samples, set_size} $\\beta$ {c_list[0].train_beta_temps}"
    )
    fig_vi.savefig("results/pma/mean_voi_beta_multi_model")

    fig_nmi.suptitle(
        f"ds:{N, beta_count, samples, set_size} $\\beta$ {c_list[0].train_beta_temps}"
    )
    fig_nmi.savefig("results/pma/mean_nmi_beta_multi_model")
