
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from models.PMATransformer import PMA_GroupingModel
from pma import Config, _results_dir
from utils.data import get_set_partition_batch
from utils.fs import load_checkpoint
from utils.masking import create_mask


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

    # Flip group labels on tgt starting with 1, generation always starts at 0
    mask = tgt[..., 0].unsqueeze(1).expand(-1, tgt.size()[-1])
    tgt = tgt ^ mask

    return (output == tgt).count_nonzero() / output.nelement()


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

    def map_one_border(N_sites, border_idx):
        map = np.zeros((2, N_sites)).astype("bool")
        map[0, :border_idx] = 1
        map[1, border_idx:] = 1
        return map

    mappings = np.array([map_one_border(c.N_sites, i) for i in range(1, c.N_sites + 1)])
    groupings = np.argmax(mappings, axis=1)

    def generate_func(batch_size, set_size, groupings, device='cuda'):
        src, tgt = get_set_partition_batch(batch_size, set_size, groupings)

        perm = torch.stack(
            [torch.randperm(src.size()[-1], device=device) for _ in range(src.size()[0])]
        )
        tgt = torch.gather(tgt, 1, perm)

        perm = perm.unsqueeze(1).expand(-1, src.size()[1], -1)
        src = torch.gather(src, 2, perm)

        # flip bits based on "temperature fluctuations"
        flip = torch.rand(src.shape)
        temp_mask = flip > .95
        src[temp_mask] = src[temp_mask] ^ 1

        return src, tgt

    sites_correct = gen(model, generate_func, groupings, c.batch_size, c.set_size, c.N_sites)

    plt.plot(logs["loss"], label="loss")
    plt.title(f"sites correct {sites_correct:.4f}")
    plt.legend()
    plt.savefig(_results_dir(f"model_{c.model_id}__id_{c.train_id}.jpg"))
