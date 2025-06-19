import numpy as np


def get_borders_betas(file):
    borders = set()
    betas = set()

    for key in file.files:
        split = key.split("_")
        borders.add(int(split[0]))
        betas.add(float(split[1]))

    borders = sorted(borders)
    betas = sorted(betas)

    return borders, betas


def key_from_border_beta(border, beta):
    return f"{border}_{beta:.2f}_"

def get_from_dict(file, border, beta):
    key = key_from_border_beta(border, beta)
    tgt = file[key + "tgt"]
    src = file[key + "src"]

    return src, tgt


def reduce_batch_size(src, tgt, size):
    batches = []

    for i in range(0, len(src), size):
        src_batch = src[i : i + size]
        tgt_batch = tgt[i : i + size]
        batches.append((src_batch, tgt_batch))

    return batches


def get_dataset(N: int, beta_count: int, samples: int, set_size: int):
    file = np.load(
        f"data/pma/dataset__N_{N}__beta_{beta_count}__set_50__samples_{samples}.npz"
    )
    return file


def get_permuted_dataset(N: int, beta_count: int, samples: int, set_size: int):
    file = np.load(
        f"data/pma/dataset_permuted__N_{N}__beta_{beta_count}__set_50__samples_{samples}.npz"
    )
    return file
