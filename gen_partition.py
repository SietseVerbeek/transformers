import os

import torch
from torch.utils.data import DataLoader

from models.transformer import Transformer
from train_partition import _data_dir, _results_dir
from utils.datasets import PartitionDataset
from utils.fs import load_checkpoint
from utils.masking import generate_square_subsequent_mask

if __name__ == "__main__":
    device = "cuda"

    N_sites = 5

    model_id = 1
    train_id = 0
    test_data_id = 3

    batch_size = 300

    model_file = _results_dir(f"model_{model_id}.params")
    checkpoint_file = _results_dir(f"model_{model_id}__id_{train_id}.pth")

    if os.path.isfile(model_file):
        model = Transformer.from_file(model_file).to(device)
    else:
        raise FileNotFoundError(f"model file with id {model_file} not found")

    model, _, _, _ = load_checkpoint(model, None, checkpoint_file)

    model.eval()

    test_data_file = _data_dir(f"N_{N_sites}_id_{test_data_id}.npz")
    test_dataset = PartitionDataset(test_data_file)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    frac_correct = 0
    frac_correct_maps = 0

    for i, data in enumerate(test_dataloader):
        if i == 20:
            break

        src, tgt = data[0].to(device), data[1].to(device)
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

        correct_maps = torch.all(out == tgt, dim=-1)
        frac_correct_maps += correct_maps.sum() / correct_maps.size(0)
        correct_sites = out == tgt
        frac_correct += correct_sites.sum() / correct_sites.nelement()

    print(frac_correct / (i + 1))
    print(frac_correct_maps / (i + 1))
