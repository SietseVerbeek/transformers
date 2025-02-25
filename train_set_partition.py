import argparse
import os
from time import process_time

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from models.SetTransformer import PMA_GroupingModel
from models.transformer import Transformer
from partition import Config, _results_dir
from utils.data import get_set_partition_batch, get_simple_partition_batch
from utils.fs import load_checkpoint, save_checkpoint
from utils.masking import create_mask


def sequence_train_epoch(
    model: Transformer,
    optimizer: Optimizer,
    loss_fn: _Loss,
    batch_size: int,
    num_batches: int,
    set_size: int,
    groupings: npt.NDArray[np.int_],
    device="cuda",
):
    model.train()
    average_loss = 0

    for i in range(num_batches):
        src, tgt = get_set_partition_batch(batch_size, set_size, groupings)
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, pad_idx=4, device=device
        )

        logits = model(
            src,
            tgt_input,
            tgt_mask=tgt_mask,
        )

        optimizer.zero_grad()

        # pred shape must be (batch_size, #classes, seq length)
        logits = logits.permute(0, 2, 1)
        loss = loss_fn(logits, tgt_out)

        loss.backward()
        optimizer.step()

        average_loss += loss.item() / (batch_size)

    return average_loss


def train(
    model: Transformer,
    epochs: int,
    optimizer: Optimizer,
    loss_fn: _Loss,
    groupings: npt.NDArray[np.int_],
    num_batches: int,
    set_size: int,
    device="cuda",
):
    start_time = process_time()
    # global current_epoch, logs

    for current_epoch in range(epochs):
        epoch_start_time = process_time()

        print("=" * 30, f"starting epoch {current_epoch}", "=" * 30)
        # logs["logs"].append("=" * 30 + f"starting epoch {current_epoch}" + "=" * 30)

        train_loss = sequence_train_epoch(
            model,
            optimizer,
            loss_fn,
            batch_size,
            num_batches,
            set_size,
            groupings,
            device=device
        )
        # logs["loss"].append(train_loss)

        print("training loss: ", train_loss)
        # logs["logs"].append(f"training loss:  {train_loss}")
        print("epoch time: ", process_time() - epoch_start_time)
        # logs["logs"].append(f"epoch time: {process_time() - epoch_start_time}")

        # save_checkpoint(model, optimizer, logs, current_epoch, "crash_checkpoint.pth")

    print(f"total training time: {process_time() - start_time}s")
    # logs["logs"].append(f"total training time: {process_time() - start_time}s\n")

def gen(
    model,
    groupings,
    set_size,
    device='cuda'
):

    batch_size = 100
    src, tgt = get_set_partition_batch(batch_size, set_size, groupings)

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
        # print(output)

    print((output == tgt).count_nonzero() / output.nelement())
    # print(tgt)

if __name__ == "__main__":

    device = "cuda"

    N_sites = 5
    learn_rate = 0.001
    betas = (0.9, 0.99)
    epochs = 5
    batch_size = 200

    model = PMA_GroupingModel(

    ).to(device)

    model.load_state_dict(torch.load('settransformer.pth', weights_only=True))

    opt = torch.optim.Adam(model.parameters(), lr=learn_rate, betas=betas)
    loss_fn = nn.CrossEntropyLoss(ignore_index=4)

    def map_one_border(N_sites, border_idx):
        map = np.zeros((2, N_sites)).astype("bool")
        map[0, :border_idx] = 1
        map[1, border_idx:] = 1
        return map

    mappings = np.array([map_one_border(N_sites, i) for i in range(1, N_sites + 1)])
    groupings = np.argmax(mappings, axis=1)

    try:
        train(model, epochs, opt, loss_fn, groupings, 330, 10)
    except KeyboardInterrupt:
        print("KeyboardInterrupt recieved")

    torch.save(model.state_dict(), 'settransformer.pth')

    gen(model, groupings, 10)
