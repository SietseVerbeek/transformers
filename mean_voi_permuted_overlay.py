import argparse
import math
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import os

import torch

from models.PMATransformer import PMA_GroupingModel
from pma import Config, _results_dir
from utils.fs import load_checkpoint
from utils.fwht import gen_spin_model_batch
from utils.masking import create_mask
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


    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    c = Config(args.config)
    N = c.N_sites
    model_file = _results_dir(f"model_{c.model_id}.params")
    checkpoint_file = _results_dir(f"model_{c.model_id}__id_{c.train_id}.pth")

    if os.path.isfile(model_file):
        model = PMA_GroupingModel.from_file(model_file).to(device)
    else:
        raise FileNotFoundError(f"model file with id {model_file} not found")

    model, _, _, logs = load_checkpoint(checkpoint_file, model, None)

    model.eval()

    def labelings_one_border(length):
        return np.array([[0] * (length - i) + [1] * i for i in range(length)])

    groupings = labelings_one_border(N)

    betas = np.linspace(.1, 1, 20, endpoint=True)

    n = len(groupings)
    cols = 3  # Choose the number of columns (you can tweak this)
    rows = math.ceil(n / cols)

    
    for i, grouping in enumerate(groupings):
        means = np.empty_like(betas)
        stds = np.empty_like(betas)

        community_size = N - i

        for j, beta in enumerate(betas): 
            src, tgt = gen_spin_model_batch(beta, 10, 50, grouping[None, ...])

            perm = torch.stack(
                [torch.randperm(src.size()[-1], device=device) for _ in range(src.size()[0])]
            )
            tgt = torch.gather(tgt, 1, perm)

            perm = perm.unsqueeze(1).expand(-1, src.size()[1], -1)
            src = torch.gather(src, 2, perm)

            batch_size = src.shape[0]
            output = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)

            for _ in range(N -1):

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src, output, pad_idx=4, device=device
                )

                logits = model(
                    src,
                    output,
                    tgt_mask=tgt_mask,
                )

                idx = logits[:, -1].argmax(dim=-1).view(-1, 1)
                output = torch.cat((output, idx), dim=-1)

            fraction = (output == tgt).count_nonzero() / output.nelement()
            var_of_info = batch_normalized_vi(output, tgt)

            means[j] = var_of_info.mean()
            stds[j] = var_of_info.std()
        plt.plot(means, label=f"pos {community_size}")

    plt.xlabel("$\\beta$")
    plt.ylabel("VOI")
    plt.legend()
    plt.savefig(f'results/pma/mean_voi_beta_permuted_overlay__{c.model_id}__{c.train_id}')

