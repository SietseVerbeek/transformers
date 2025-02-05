import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from models.transformer import Transformer
from partition import Config
from train_partition import _results_dir
from utils.data import get_simple_partition_batch
from utils.fs import load_checkpoint
from utils.masking import create_mask, generate_square_subsequent_mask

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(args.config)

    N_sites = config.N_sites

    model_id = config.model_id
    train_id = config.train_id
    test_data_id = config.test_data_id

    batch_size = config.batch_size

    device = "cuda"

    model_file = _results_dir(f"model_{model_id}.params")
    checkpoint_file = _results_dir(f"model_{model_id}__id_{train_id}.pth")

    if os.path.isfile(model_file):
        model = Transformer.from_file(model_file).to(device)
    else:
        raise FileNotFoundError(f"model file with id {model_file} not found")

    model, _, _, logs = load_checkpoint(checkpoint_file, model, None)

    model.eval()

    def map_one_border(N_sites, border_idx):
        map = np.zeros((2, N_sites)).astype("bool")
        map[0, :border_idx] = 1
        map[1, border_idx:] = 1
        return map

    mappings = np.array([map_one_border(N_sites, i) for i in range(0, N_sites + 1)])
    groupings = np.argmax(mappings, axis=1)

    frac_correct = 0
    frac_correct_maps = 0

    for i in range(1):

        src, tgt = get_simple_partition_batch(300, 20, groupings, device=device)
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        out = torch.full((src.size(0), 1), 0, dtype=torch.int64, device=device)
        begining = torch.full((src.size(0), 1), 0, dtype=torch.int64, device=device)

        for _ in range(N_sites - 1):
            # tgt_mask = (generate_square_subsequent_mask(out.size(1))).to(device)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, out, pad_idx=4, device=device
            )

            # logits = model(src, out, tgt_mask=tgt_mask).detach()
            print('out input', out[:3])
            logits = model(
                src,
                out,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )
            print('logits', logits[:3])
            print('tgt', tgt[:3])

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src, tgt_out[..., :3], pad_idx=4, device=device
            )
            print('part input', tgt_input[:3, ..., :3])
            part_logits = model(
                src,
                tgt_input[..., :3],
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )
            print('part_logits', part_logits[:3])

            idx = logits[:, -1].argmax(dim=-1).view(-1, 1)

            """
            Sample next index from out distribution

            probabilities = torch.softmax(logits[:, -1], -1).cumsum(-1)
            throw = torch.rand(src.size(0), 1).to(device)
            idx = torch.searchsorted(probabilities, throw)
            """

            # Concatenate previous input with predicted best word
            out = torch.cat((out, idx), dim=-1)
            out_new = logits.argmax(dim=-1)
            out = out_new
            out = torch.cat((begining, out), dim=-1)

            print('out', out_new[:3])
            # print(out[:3], out_new[:3], out[..., 1:] == out_new)
                
        print("out", out.unique(return_counts=True))
        print("tgt", tgt.unique(return_counts=True))
        correct_maps = torch.all(out == tgt, dim=-1)
        frac_correct_maps += correct_maps.sum() / correct_maps.size(0)
        correct_sites = out == tgt
        frac_correct += correct_sites.sum() / correct_sites.nelement()

    sites_correct = frac_correct / (i + 1)
    maps_correct = frac_correct_maps / (i + 1)

    print("sites correct" , sites_correct)

    plt.plot(logs["loss"], label="loss")
    plt.plot(logs["val_loss"], label="validation loss")
    plt.title(f"sites {sites_correct:.3f}, maps {maps_correct:.3f}")
    plt.legend()
    plt.savefig(_results_dir(f"model_{model_id}__id_{train_id}.jpg"))
