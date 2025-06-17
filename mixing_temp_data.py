import argparse
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch

from models.PMATransformer import PMA_GroupingModel
from pma import Config, _results_dir
from utils.fs import load_checkpoint, load_from_txt
from utils.masking import create_mask
from utils.tests import batch_normalized_vi


def load_datasets(
    N: int,
    mu_arr: npt.NDArray[np.float64],
    beta_arr: npt.NDArray[np.float64],
    ds_per_model: int,
    samples_per_ds: int,
) -> npt.NDArray[np.int64]:
    # graph, npt.NDArray[beta, configs]
    shape = (mu_arr.shape[0], beta_arr.shape[0])
    datasets = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        datasets[idx] = []

    for i, mu in enumerate(mu_arr):
        for j, beta in enumerate(beta_arr):
            dir = Path(f"data/sbm/{N}/mu_{mu:.2f}/beta_{beta:.2f}")
            if i == j & i == 0:
                model_idxs = np.arange(sum(1 for _ in dir.iterdir()))
            for idx in model_idxs:
                samples = load_from_txt(str(dir / str(idx))).reshape(
                    ds_per_model, samples_per_ds, -1
                )
                datasets[i][j].append(samples)

    datasets = np.array(datasets.tolist(), dtype=np.int64)

    print('pre reshape', datasets.shape)
    return datasets.reshape(datasets.shape[:2] + (-1,) + datasets.shape[4:])


if __name__ == "__main__":
    q = 2
    m = 5
    N = q * m

    datasets_per_model = 10
    samples_per_dataset = 50
    n_samples = datasets_per_model * samples_per_dataset

    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    c = Config(args.config)
    model_file = _results_dir(f"model_{c.model_id}.params")
    checkpoint_file = _results_dir(f"model_{c.model_id}__id_{c.train_id}.pth")

    if os.path.isfile(model_file):
        model = PMA_GroupingModel.from_file(model_file).to(device)
    else:
        raise FileNotFoundError(f"model file with id {model_file} not found")

    model, _, _, logs = load_checkpoint(checkpoint_file, model, None)

    model.eval()

    mu_arr = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    beta_arr = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    datasets = load_datasets(
        10,
        mu_arr,
        beta_arr,
        datasets_per_model,
        samples_per_dataset,
    )
    voi_s = np.zeros(datasets.shape[:2], dtype=np.float64)
    print(datasets.shape)
    print(voi_s.shape)

    for i in range(datasets.shape[0]):
        for j in range(datasets.shape[1]):
            src = datasets[i][j]
            batch_size = src.shape[0]
            src = torch.from_numpy(src.astype("int")).to(device)

            output = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)

            for _ in range(N - 1):
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

            tgt = torch.zeros_like(output)
            tgt[..., 5:] = 1

            fraction = (output == tgt).count_nonzero() / output.nelement()
            voi_s[i, j] = batch_normalized_vi(output, tgt).mean()

    x, y = np.meshgrid(mu_arr, beta_arr)

    np.savez(f'results/pma/voi_{c.model_id}__id_{c.train_id}', mu=mu_arr, beta=beta_arr, voi=voi_s)
