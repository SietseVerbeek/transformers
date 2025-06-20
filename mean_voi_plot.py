import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    get_dataset,
    src_tgt_from_dict,
    reduce_batch_size,
    voi_nmi_from_dict,
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

    file = np.load(
        f"results/pma/predictions/dataset__N_{N}__beta_{beta_count}__set_50__samples_{samples}__{c.model_id}__{c.train_id}.npz"
    )

    borders, betas = get_borders_betas(file)

    mean_vi = np.empty_like(betas)
    stds_vi = np.empty_like(betas)

    mean_nmi = np.empty_like(betas)
    stds_nmi = np.empty_like(betas)

    for i, beta in enumerate(betas):
        var_of_info = np.empty(0, dtype=np.float64)
        nmi = np.empty(0, dtype=np.float64)

        for border in borders:
            voi_temp, nmi_temp = voi_nmi_from_dict(file, border, beta)

            var_of_info = np.concat([var_of_info, voi_temp])
            nmi = np.concat([nmi, nmi_temp])

        mean_vi[i] = var_of_info.mean()
        stds_vi[i] = var_of_info.std()

        mean_nmi[i] = nmi.mean()
        stds_nmi[i] = nmi.std()


    plt.errorbar(betas, mean_vi, yerr=stds_vi)
    plt.xlabel("$\\beta$")
    plt.ylabel("VOI")
    plt.title(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )
    plt.savefig(f"results/pma/mean_voi_beta__{c.model_id}__{c.train_id}")
    plt.clf()

    plt.errorbar(betas, mean_nmi, yerr=stds_nmi)
    plt.xlabel("$\\beta$")
    plt.ylabel("NMI")
    plt.title(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )
    plt.savefig(f"results/pma/mean_nmi_beta__{c.model_id}__{c.train_id}")
