
import argparse
import os

import torch
from torch.utils.data import DataLoader
from models.transformer import Transformer
from partition import Config, _results_dir
from utils.datasets import PartitionDataset
from utils.fs import load_checkpoint
from train_partition import _data_dir
from utils.masking import generate_square_subsequent_mask


parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)

args = parser.parse_args()

config = Config(args.config)

N_sites = config.N_sites

model_id = config.model_id
train_id = config.train_id
test_data_id = 4

batch_size = 10

device = "cuda"

model_file = _results_dir(f"model_{model_id}.params")
checkpoint_file = _results_dir(f"model_{model_id}__id_{train_id}.pth")

if os.path.isfile(model_file):
    model = Transformer.from_file(model_file).to(device)
else:
    raise FileNotFoundError(f"model file with id {model_file} not found")

model, _, _, logs = load_checkpoint(checkpoint_file, model, None)

model.eval()

test_data_file = _data_dir(f"N_{N_sites}_id_{test_data_id}.npz")
test_dataset = PartitionDataset(test_data_file)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for i, (src, tgt) in enumerate(test_dataloader):
    if i == 2:
        break
    src, tgt = src.to(device), tgt.to(device)
    out = torch.full((src.size(0), 1), 0, dtype=torch.int64, device=device)

    for _ in range(N_sites - 1):
        tgt_mask = (generate_square_subsequent_mask(out.size(1))).to(device)

        logits = model(src, out, tgt_mask=tgt_mask).detach()

        idx = logits[:, -1].argmax(dim=-1).view(-1, 1)

        """
        Sample next index from out distribution

        probabilities = torch.softmax(logits[:, -1], -1).cumsum(-1)
        throw = torch.rand(src.size(0), 1).to(device)
        idx = torch.searchsorted(probabilities, throw)
        """

        # Concatenate previous input with predicted best word
        out = torch.cat((out, idx), dim=-1)

    print("tgt")
    print(tgt)
    print("out")
    print(out)
    print((tgt == out).sum())
