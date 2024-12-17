import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import entropy

from encoder import Config, _results_dir
from models.encoder import SequenceEncoder
from utils.datasets import load_from_txt
from utils.fs import load_checkpoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(args.config)

    model_id = config.model_id
    train_id = config.train_id

    num_tokens = config.num_tokens
    d_model = config.d_model
    num_heads = config.num_heads
    num_layers = config.num_layers
    num_hidden = config.num_hidden

    N_sites = config.N_sites
    train_data_id = config.train_data_id

    learn_rate = config.learn_rate
    betas = (config.beta1, config.beta2)
    epochs = config.epochs
    batch_size = config.batch_size

    device = "cuda"

    model_file = _results_dir(f"model_{model_id}.params")
    checkpoint_file = _results_dir(f"model_{model_id}__id_{train_id}.pth")

    i = 0
    results_file = _results_dir(f"model_{model_id}__id_{train_id}__{i}.jpg")
    while os.path.isfile(results_file):
        i += 1
        results_file = _results_dir(f"model_{model_id}__id_{train_id}__{i}.jpg")

    train_datafile = f"data/N_sites_{N_sites}/sites_{N_sites}__id_{train_data_id}"

    if os.path.isfile(model_file):
        print(f"Model loaded from file {model_file}")
        model = SequenceEncoder.from_file(model_file).to(device)

    else:
        model = SequenceEncoder(
            num_tokens=num_tokens,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            num_hidden=num_hidden,
        ).to("cuda")

    model, opt, current_epoch, logs = load_checkpoint(checkpoint_file, model)

    model.eval()

    def generate(num_sequences, seq_length, model=model):
        src = torch.full((num_sequences, 1), 2, device="cuda")
        for i in range(seq_length):
            logits = model(src)
            probs = torch.softmax(logits[:, -1], -1).cumsum(-1)

            throw = torch.rand(num_sequences, 1).to("cuda")
            idx = torch.searchsorted(probs, throw)

            src = torch.cat((src, idx), dim=-1)

        return src[:, 1:]

    num_sequences = 500
    batches = 20

    total_sequences = num_sequences * batches
    out = np.ndarray((total_sequences, N_sites))

    for i in range(batches):
        print("gen", i, end="\r")
        out[i * num_sequences : (i + 1) * num_sequences] = generate(
            num_sequences, N_sites
        ).cpu()

    configs_unique, configs_counts = np.unique(
        load_from_txt(train_datafile), axis=0, return_counts=True
    )
    in_dist = configs_counts / configs_counts.sum()
    in_dist[in_dist == 0] = 10e-10

    out_mask = np.any(np.all(out[..., None, :] == configs_unique, axis=-1), axis=-1)

    out_unique, _ = np.unique(out[out_mask], axis=0, return_counts=True)
    out_counts = np.sum(np.all(out[..., None, :] == configs_unique, axis=-1), axis=0)
    out_dist = out_counts / out_counts.sum()
    out_dist[out_dist == 0] = 10e-10

    plt.plot(in_dist, label="in")
    plt.plot(out_dist, label="generated")
    plt.title(
        f"epochs: {epochs}, error frac: {np.sum(~out_mask) / total_sequences}, KL-div: {entropy(in_dist, out_dist):.6f}"
    )
    plt.legend()
    plt.savefig(results_file)
