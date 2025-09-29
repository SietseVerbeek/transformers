import matplotlib.pyplot as plt
import numpy as np

from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    tgt_output_from_dict,
    voi_nmi_from_dict,
)


if __name__ == "__main__":
    device = "cpu"
    model, c, logs = pma_from_config(device)

    N = c.N_sites

    beta_count = 20
    samples = 750
    set_size = 50

    file = np.load(
        f"results/pma/predictions/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}__{c.model_id}__{c.train_id}.npz"
    )

    borders, betas = get_borders_betas(file)

    mean_border_dist = np.empty_like(betas)
    std_border_dist = np.empty_like(betas)
    p_99 = np.empty((len(betas), 2))
    p_95 = np.empty((len(betas), 2))
    p_68 = np.empty((len(betas), 2))

    for i, beta in enumerate(betas):
        border_dists = np.empty(0, dtype=np.float64)

        for border in borders:
            tgt, out = tgt_output_from_dict(file, border, beta)
            out_borders = np.argmax(out, axis=-1)
            tmp_border_dists = out_borders - (border % N)

            border_dists = np.concat([border_dists, tmp_border_dists])

        p_99[i] = np.percentile(border_dists, [0.15, 99.85])
        p_95[i] = np.percentile(border_dists, [2.5, 97.5])
        p_68[i] = np.percentile(border_dists, [16, 84])
        mean_border_dist[i] = border_dists.mean()
        std_border_dist[i] = border_dists.std()


    plt.plot(betas, mean_border_dist)
    plt.fill_between(betas, p_99[:, 0], p_99[:, 1],
                 color="blue", alpha=0.1, label="99% band")

    plt.fill_between(betas, p_95[:, 0], p_95[:, 1],
                 color="blue", alpha=0.3, label="95% band")

    plt.fill_between(betas, p_68[:, 0], p_68[:, 1],
                 color="blue", alpha=0.5, label="68% band")
    plt.xlabel("$\\beta$")
    plt.ylabel("Border dist")
    plt.title(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )
    plt.legend()
    plt.savefig(f"results/pma/mean_border_beta__{c.model_id}__{c.train_id}")

