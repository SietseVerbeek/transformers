import matplotlib.pyplot as plt
import numpy as np

from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    tgt_output_from_dict,
    voi_nmi_from_dict,
)


if __name__ == "__main__":
    device = "cuda"
    model, c, logs = pma_from_config(device)

    N = c.N_sites

    beta_count = 5
    samples = 200
    set_size = 25

    file = np.load(
        f"results/pma/predictions/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}__{c.model_id}__{c.train_id}.npz"
    )

    borders, betas = get_borders_betas(file)

    mean_border_dist = np.empty_like(betas)
    std_border_dist = np.empty_like(betas)

    for i, beta in enumerate(betas):
        border_dists = np.empty(0, dtype=np.float64)

        for border in borders:
            tgt, out = tgt_output_from_dict(file, border, beta)
            out_borders = np.argmax(out, axis=-1)
            tmp_border_dists = out_borders - (border % N)
            print(border % N)
            print(border_dists)
            print(out[0])


            border_dists = np.concat([border_dists, tmp_border_dists])

        mean_border_dist[i] = border_dists.mean()
        std_border_dist[i] = border_dists.std()


    plt.errorbar(betas, mean_border_dist, yerr=std_border_dist)
    plt.xlabel("$\\beta$")
    plt.ylabel("Border dist")
    plt.title(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )
    plt.savefig(f"results/pma/mean_border_beta__{c.model_id}__{c.train_id}")

