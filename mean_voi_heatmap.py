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

    N = 10
    beta_count = 20
    samples = 1000
    set_size = 50

    mean_voi_grid = np.zeros((len(c_list), beta_count))
    voi_std_grid = np.zeros((len(c_list), beta_count))

    mean_nmi_grid = np.zeros((len(c_list), beta_count))
    nmi_std_grid = np.zeros((len(c_list), beta_count))

    beta_train = np.zeros((len(c_list)))

    fig_vi, ax_vi = plt.subplots(1, 1)
    fig_nmi, ax_nmi = plt.subplots(1, 1)

    for z, (model, c, logs) in enumerate(zip(models, c_list, logs_list)):
        file = np.load(
            f"results/pma/predictions/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}__{c.model_id}__{c.train_id}.npz"
        )

        borders, betas = get_borders_betas(file)
        beta_train[z] = c.train_beta_temps[0]

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

            mean_voi_grid[z, i] = var_of_info.mean()
            voi_std_grid[z, i] = var_of_info.std()

            mean_nmi_grid[z, i] = nmi.mean()
            nmi_std_grid[z, i] = nmi.std()

    vi_cmesh = ax_vi.pcolormesh(beta_train, betas, mean_voi_grid.T, shading="auto")
    # vi_std_cmesh = ax_vi[1].pcolormesh(beta_train, betas, voi_std_grid.T, shading="auto")

    fig_vi.colorbar(vi_cmesh, ax=ax_vi)
    # fig_vi.colorbar(vi_std_cmesh, ax=ax_vi[1])

    ax_vi.invert_yaxis()
    # ax_vi[1].invert_yaxis()

    ax_vi.set_ylabel("$\\beta$")
    ax_vi.set_xlabel("training $\\beta$")
    ax_vi.set_title("VI")

    # ax_vi[1].set_ylabel("$\\beta$")
    # ax_vi[1].set_xlabel("training $\\beta$")
    # ax_vi[1].set_title("standard deviation")

    fig_vi.suptitle(
        f"ds:{N, beta_count, samples, set_size}"
    )
    fig_vi.savefig("results/pma/mean_voi_heatmap")

    nmi_cmesh = ax_nmi.pcolormesh(beta_train, betas, mean_nmi_grid.T, shading="auto")
    # nmi_std_cmesh = ax_nmi[1].pcolormesh(beta_train, betas, nmi_std_grid.T, shading="auto")

    ax_nmi.invert_yaxis()
    # ax_nmi[1].invert_yaxis()

    ax_nmi.set_ylabel("$\\beta$")
    ax_nmi.set_xlabel("training $\\beta$")

    # ax_nmi[1].set_ylabel("$\\beta$")
    # ax_nmi[1].set_xlabel("training $\\beta$")

    fig_nmi.colorbar(nmi_cmesh, ax=ax_nmi)
    # fig_nmi.colorbar(nmi_std_cmesh, ax=ax_nmi[1])

    fig_nmi.suptitle(
        f"ds:{N, beta_count, samples, set_size}"
    )
    fig_nmi.savefig(f"results/pma/mean_nmi_heatmap__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}")
