import matplotlib.pyplot as plt
import numpy as np
import torch
from numba import njit

from utils.masking import create_mask
from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    get_dataset,
    get_from_dict,
    reduce_batch_size,
)
from utils.tests import normalized_vi


@njit
def normalized_vi_loop(arr1, arr2):
    num_clusterings = arr1.shape[0]
    output = np.zeros(num_clusterings, dtype=np.float64)

    for i in range(num_clusterings):
        output[i] = normalized_vi(arr1[i], arr2[i])

    return output


def torch_check(input):
    if isinstance(input, torch.Tensor):
        if input.is_cuda:
            return input.cpu().numpy()
        return input.numpy()
    return input


def batch_normalized_vi(results, truth):
    results = torch_check(results)
    truth = torch_check(truth)

    return normalized_vi_loop(results, truth)


if __name__ == "__main__":
    N = 10
    beta_count = 20
    samples = 1000
    set_size = 50

    file = get_dataset(N, beta_count, samples, set_size)

    borders, betas = get_borders_betas(file)

    means = np.empty_like(betas)
    stds = np.empty_like(betas)

    device = "cuda"
    model, c, logs = pma_from_config(device)

    for i, beta in enumerate(betas):
        var_of_info = np.empty(0, dtype=np.float64)

        for border in borders:
            src, tgt = get_from_dict(file, border, beta)
            batches = reduce_batch_size(src, tgt, 10)

            for src, tgt in batches:
                batch_size = src.shape[0]
                src = torch.from_numpy(src).to(device)
                tgt = torch.from_numpy(tgt).to(device)

                output = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)

                for _ in range(N - 1):
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = (
                        create_mask(src, output, pad_idx=4, device=device)
                    )

                    logits = model(
                        src,
                        output,
                        tgt_mask=tgt_mask,
                    )

                    idx = logits[:, -1].argmax(dim=-1).view(-1, 1)
                    output = torch.cat((output, idx), dim=-1)

                fraction = (output == tgt).count_nonzero() / output.nelement()
                var_of_info = np.concat([var_of_info, batch_normalized_vi(output, tgt)])

            means[i] = var_of_info.mean()
            stds[i] = var_of_info.std()
            print("mean", var_of_info.mean())
            print("std", var_of_info.std())
    plt.errorbar(betas, means, yerr=stds)
    plt.xlabel("$\\beta$")
    plt.ylabel("VOI")
    plt.title(f"trained at $\\beta = {c.train_beta_temps}$, sites permuted")
    plt.savefig(f"results/pma/mean_voi_beta__{c.model_id}__{c.train_id}")
