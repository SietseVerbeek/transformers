import matplotlib.pyplot as plt
import numpy as np

from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    voi_nmi_from_dict,
)


if __name__ == "__main__":
    device = "cuda"
    model, c, logs = pma_from_config(device)

    N = c.N_sites

    beta_count = 20
    samples = 1000
    set_size = 50

    file = np.load(
        f"results/pma/predictions/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}__{c.model_id}__{c.train_id}.npz"
    )

    borders, betas = get_borders_betas(file)

    fig_vi, axes_vi = plt.subplots()
    fig_nmi, axes_nmi = plt.subplots()

    for i, border in enumerate(borders):
        mean_vi = np.empty_like(betas)
        stds_vi = np.empty_like(betas)

        mean_nmi = np.empty_like(betas)
        stds_nmi = np.empty_like(betas)

        for j, beta in enumerate(betas):
            voi, nmi = voi_nmi_from_dict(file, border, beta)

            mean_vi[j] = voi.mean()
            stds_vi[j] = voi.std()

            mean_nmi[j] = nmi.mean()
            stds_nmi[j] = nmi.std()

        axes_vi.plot(betas, mean_vi, label=f"pos {border}")
        axes_nmi.plot(betas, mean_nmi, label=f"pos {border}")

    axes_vi.set_xlabel("$\\beta$")
    axes_vi.set_ylabel("VOI")
    axes_vi.legend()
    axes_vi.set_title(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )
    fig_vi.savefig(
        f"results/pma/mean_voi_beta_border_overlay__{c.model_id}__{c.train_id}"
    )

    axes_nmi.set_xlabel("$\\beta$")
    axes_nmi.set_ylabel("NMI")
    axes_nmi.legend()
    axes_nmi.set_title(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )
    fig_nmi.savefig(
        f"results/pma/mean_nmi_beta_border_overlay__{c.model_id}__{c.train_id}"
    )
