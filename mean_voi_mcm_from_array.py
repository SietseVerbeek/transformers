from pathlib import Path
import numpy as np
from mcmpy import Data

from time import perf_counter

from utils.test_datasets import (
    get_borders_betas,
    get_dataset,
    src_tgt_from_dict,
    key_from_border_beta,
)
from utils.tests import batch_normalized_mi, batch_normalized_vi
from utils.training import generate_predictions

if __name__ == "__main__":
    N = 10
    beta_count = 5
    samples = 200
    set_size = 25

    def labelings_one_border(length):
        return np.array([[0] * (length - i) + [1] * i for i in range(length)])

    def labelings_to_partitioning(labels):
        labels = np.array(labels)
        return np.stack([1 - labels, labels], axis=-2)

    labelings = labelings_one_border(N)
    partitions = labelings_to_partitioning(labelings)

    file = get_dataset(N, beta_count, samples, set_size)

    borders, betas = get_borders_betas(file)

    save_dict = {}

    for border in borders:
        print("border", border)
        for beta in betas:
            print("beta", beta)
            start = perf_counter()
            src_full, tgt_full = src_tgt_from_dict(file, border, beta)
            key_prefix = key_from_border_beta(border, beta)
            save_dict[key_prefix + "tgt"] = tgt_full

            out = np.empty_like(tgt_full)

            for i, (src, tgt) in enumerate(zip(src_full, tgt_full)):
                data = Data(src, N, 2)
                best = float('-inf')
                best_j = None
                for j, part in enumerate(partitions):
                    evidence = data.log_evidence(part)
                    if evidence > best:
                        best = evidence
                        best_j = j
                out[i] = labelings[best_j]

            voi = batch_normalized_vi(out, tgt_full)
            nmi = batch_normalized_mi(out, tgt_full)

            save_dict[key_prefix + "out"] = out
            save_dict[key_prefix + "voi"] = voi
            save_dict[key_prefix + "nmi"] = nmi
            print(perf_counter() - start)

    Path("results/mcmpy/best_bipart/").mkdir(parents=True, exist_ok=True)
    np.savez(
        f"results/mcmpy/best_bipart/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}",
        **save_dict
    )
