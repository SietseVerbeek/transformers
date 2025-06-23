import matplotlib.pyplot as plt
import numpy as np

from utils.pma import many_pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    voi_nmi_from_dict,
)

if __name__ == "__main__":
    device = "cuda"
    models, c_list, logs_list = many_pma_from_config(device)

    beta_count = 20
    samples = 1000
    set_size = 50

    fig_vi, ax_vi = plt.subplots()
    fig_nmi, ax_nmi = plt.subplots()

    for model, c, logs in zip(models, c_list, logs_list):
        N = c.N_sites
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
