
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from models.PMATransformer import PMA_GroupingModel
from pma import Config, _results_dir
from utils.data import get_set_partition_batch
from utils.fs import load_checkpoint
from utils.fwht import gen_spin_model_batch
from utils.masking import create_mask
from utils.tests import batch_normalized_vi


def gen(
    model,
    gen_func,
    groupings,
    batch_size,
    set_size,
    N_sites,
    device='cuda'
):

    src, tgt = gen_func(batch_size, set_size, groupings)

    # get_set_partition does not give the exact batch_size requested,
    # it changes the batch size to give a uniform dist
    batch_size = src.shape[0]

    output = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)

    for i in range(N_sites -1):

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

    return var_of_info, fraction


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    c = Config(args.config)

    device = "cuda"

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

    groupings = labelings_one_border(c.N_sites)

    def generate_func(batch_size, set_size, groupings):
        src, tgt = gen_spin_model_batch(c.beta_temp, batch_size, set_size, groupings)
        return src, tgt

    var_of_info, sites_correct = gen(model, generate_func, groupings, c.batch_size, c.set_size, c.N_sites)

    plt.plot(logs["loss"], label="loss")
    plt.title(f"VOI {var_of_info:.4f}, sites correct {sites_correct:.4f}")
    plt.legend()
    plt.savefig(_results_dir(f"model_{c.model_id}__id_{c.train_id}.jpg"))
