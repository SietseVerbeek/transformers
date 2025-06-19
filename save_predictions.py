from pathlib import Path
import numpy as np
import torch

from time import perf_counter

from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    get_dataset,
    get_from_dict,
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
            src, tgt = get_from_dict(file, border, beta)
            key_prefix = key_from_border_beta(border, beta)
            save_dict[key_prefix + "tgt"] = tgt.copy()

            batches = reduce_batch_size(src, tgt, 200)
            output = torch.empty(0, dtype=torch.int8, device=device)

            for src, tgt in batches:
                src = torch.from_numpy(src).pin_memory()
                src = src.to(device, non_blocking=True)
                tgt = torch.from_numpy(tgt).pin_memory()
                tgt = tgt.to(device, non_blocking=True)

                output = torch.cat([output, generate_predictions(model, src, device)])

            output = output.cpu().numpy()
            voi = batch_normalized_vi(output, tgt)
            nmi = batch_normalized_mi(output, tgt)

            save_dict[key_prefix + "out"] = output
            save_dict[key_prefix + "voi"] = voi
            save_dict[key_prefix + "nmi"] = nmi
            print(perf_counter() - start)

    Path("results/pma/predictions/").mkdir(parents=True, exist_ok=True)
    np.savez(
        f"results/pma/predictions/dataset__N_{N}__beta_{beta_count}__set_50__samples_{samples}__{c.model_id}__{c.train_id}"
    )
