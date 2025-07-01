from pathlib import Path
import numpy as np
import torch

from time import perf_counter

from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    get_dataset,
    src_tgt_from_dict,
    key_from_border_beta,
    reduce_batch_size,
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

    file = get_dataset(N, beta_count, samples, set_size)

    borders, betas = get_borders_betas(file)

    save_dict = {}

    for border in borders:
        for beta in betas:
            start = perf_counter()
            src, tgt_full = src_tgt_from_dict(file, border, beta)
            key_prefix = key_from_border_beta(border, beta)
            save_dict[key_prefix + "tgt"] = tgt_full

            batches = reduce_batch_size(src, tgt_full, 200)
            stack = []

            for src, tgt in batches:
                src = torch.from_numpy(src).pin_memory()
                src = src.to(device, non_blocking=True)
                tgt = torch.from_numpy(tgt).pin_memory()
                tgt = tgt.to(device, non_blocking=True)

                stack.append(generate_predictions(model, src, device).cpu().numpy())

            output = np.concatenate(stack, axis=0)
            voi = batch_normalized_vi(output, tgt_full)
            nmi = batch_normalized_mi(output, tgt_full)

            save_dict[key_prefix + "out"] = output
            save_dict[key_prefix + "voi"] = voi
            save_dict[key_prefix + "nmi"] = nmi
            print(perf_counter() - start)

    Path("results/pma/predictions/").mkdir(parents=True, exist_ok=True)
    np.savez(
        f"results/pma/predictions/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}__{c.model_id}__{c.train_id}",
        **save_dict
    )
