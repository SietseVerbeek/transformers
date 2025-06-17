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
from utils.tests import batch_normalized_vi
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

    for i, border in enumerate(borders):
        means = np.empty_like(betas)
        stds = np.empty_like(betas)

        for j, beta in enumerate(betas):
            src, tgt = get_from_dict(file, border, beta)
            batches = reduce_batch_size(src, tgt, 10)
            for src, tgt in batches[:4]:
                src = torch.from_numpy(src).to(device)
                tgt = torch.from_numpy(tgt).to(device)

                output = generate_predictions(model, src, device)

                fraction = (output == tgt).count_nonzero() / output.nelement()
                var_of_info = batch_normalized_vi(output, tgt)

                means[j] = var_of_info.mean()
                stds[j] = var_of_info.std()
        plt.plot(betas, means, label=f"pos {border}")

    plt.xlabel("$\\beta$")
    plt.ylabel("VOI")
    plt.legend()
    plt.title(
        f"ds:{N, beta_count, samples, set_size}, shown: {logs["samples_shown"]} $\\beta$ {c.train_beta_temps}"
    )
    plt.savefig(
        f"results/pma/mean_voi_beta_permuted_overlay__{c.model_id}__{c.train_id}"
    )
