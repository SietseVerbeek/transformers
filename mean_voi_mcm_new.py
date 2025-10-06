import matplotlib.pyplot as plt
import numpy as np

from utils.test_datasets import (
    get_borders_betas,
    voi_nmi_from_dict,
)


if __name__ == "__main__":
    N = 10

    beta_count = 5
    samples = 200
    set_size = 25

    file = np.load(
        f"results/mcmpy/best_bipart/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}.npz"
    )

    borders, betas = get_borders_betas(file)

    correct_voi = np.empty_like(betas)
    mean_vi = np.empty_like(betas)
    stds_vi = np.empty_like(betas)
    con_vi = np.empty((len(betas), 2))

    correct_nmi = np.empty_like(betas)
    mean_nmi = np.empty_like(betas)
    stds_nmi = np.empty_like(betas)
    con_nmi = np.empty((len(betas), 2))

    for i, beta in enumerate(betas):
        var_of_info = np.empty(0, dtype=np.float64)
        nmi = np.empty(0, dtype=np.float64)

        for border in borders:
            voi_temp, nmi_temp = voi_nmi_from_dict(file, border, beta)

            var_of_info = np.concat([var_of_info, voi_temp])
            nmi = np.concat([nmi, nmi_temp])

        correct_voi[i] = np.mean(var_of_info == 0)
        mean_vi[i] = var_of_info.mean()
        stds_vi[i] = var_of_info.std()
        con_vi[i] = np.percentile(var_of_info, [2.5, 97.5])

        correct_nmi[i] = np.mean(nmi == 1)
        mean_nmi[i] = nmi.mean()
        stds_nmi[i] = nmi.std()
        con_nmi[i] = np.percentile(nmi, [2.5, 97.5])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    plt.plot(betas, mean_vi)
    plt.xlabel("$\\beta$")
    # plt.ylabel("VOI")
    # plt.fill_between(betas, con_vi[:, 0], con_vi[:, 1], alpha=0.2)
    plt.title(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )

    plt.plot(betas, correct_voi, label="accuracy", color="tab:green")
    # plt.xlabel("$\\beta$")
    # plt.ylabel("Fraction Correct")
    plt.legend()

    # share axis, so flip only one
    axes[0].invert_xaxis()
    plt.tight_layout()
    plt.savefig(f"results/pma/mean_voi_beta_mcm_accuracy")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    axes[0].plot(betas, mean_nmi)
    axes[0].set_xlabel("$\\beta$")
    axes[0].set_ylabel("NMI")
    # plt.fill_between(betas, con_vi[:, 0], con_vi[:, 1], alpha=0.2)
    axes[0].set_title(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )

    axes[1].plot(betas, correct_nmi, label="Fraction Correct", color="tab:green")
    axes[1].set_xlabel("$\\beta$")
    axes[1].set_ylabel("Fraction Correct")
    axes[1].legend()

    # share axis, so flip only one
    axes[0].invert_xaxis()
    plt.tight_layout()
    plt.savefig(f"results/pma/mean_nmi_beta_mcm_accuracy")
    plt.close()
