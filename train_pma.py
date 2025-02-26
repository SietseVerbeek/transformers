import argparse
import os
from time import process_time

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from models.PMATransformer import PMA_GroupingModel
from models.transformer import Transformer
from pma import Config, _results_dir
from utils.data import get_set_partition_batch
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
    global current_epoch, logs

    for current_epoch in range(current_epoch, current_epoch + epochs):
        epoch_start_time = process_time()

        print("=" * 30, f"starting epoch {current_epoch}", "=" * 30)
        logs["logs"].append("=" * 30 + f"starting epoch {current_epoch}" + "=" * 30)

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
        logs["loss"].append(train_loss)

        print("training loss: ", train_loss)
        logs["logs"].append(f"training loss:  {train_loss}")
        print("epoch time: ", process_time() - epoch_start_time)
        logs["logs"].append(f"epoch time: {process_time() - epoch_start_time}")

        save_checkpoint(model, optimizer, logs, current_epoch, "crash_checkpoint.pth")

    print(f"total training time: {process_time() - start_time}s")
    logs["logs"].append(f"total training time: {process_time() - start_time}s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    c = Config(args.config)

    batch_size = 200

    device = "cuda"

    model_file = _results_dir(f"model_{c.model_id}.params")
    checkpoint_file = _results_dir(f"model_{c.model_id}__id_{c.train_id}.pth")

    logs = {"loss": [], "val_loss": [], "logs": []}

    if os.path.isfile(model_file):
        print(f"Model loaded from file {model_file}")
        model = PMA_GroupingModel.from_file(model_file).to(device)

    else:
        print(f"Creating new model file at {model_file}")
        model = PMA_GroupingModel(
            embed_dim=c.embed_dim,
            enc_layers=c.enc_layers,
            enc_heads=c.enc_heads,
            pma_heads=c.pma_heads,
            dec_heads=c.dec_heads,
            queries=c.queries,
        ).to(device)

        model.to_file(model_file)

    opt = torch.optim.Adam(model.parameters(), lr=c.learn_rate, betas=c.betas)
    loss_fn = nn.CrossEntropyLoss(ignore_index=4)

    model, opt, current_epoch, logs = load_checkpoint(checkpoint_file, model, opt)

    def map_one_border(N_sites, border_idx):
        map = np.zeros((2, N_sites)).astype("bool")
        map[0, :border_idx] = 1
        map[1, border_idx:] = 1
        return map

    mappings = np.array([map_one_border(c.N_sites, i) for i in range(1, c.N_sites + 1)])
    groupings = np.argmax(mappings, axis=1)

    try:
        train(model, c.epochs, opt, loss_fn, groupings, 330, c.set_size)
    except KeyboardInterrupt:
        print("KeyboardInterrupt recieved")

    os.remove("crash_checkpoint.pth")
    os.remove("crash_checkpoint.pth.logs")
    save_checkpoint(model, opt, logs, current_epoch, checkpoint_file)
