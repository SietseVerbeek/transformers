import argparse
import os
from time import process_time

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from models.transformer import Transformer
from partition import Config, _results_dir
from utils.data import get_simple_partition_batch
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
        src, tgt = get_simple_partition_batch(batch_size, set_size, groupings)
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, pad_idx=4, device=device
        )

        logits = model(
            src,
            tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

        optimizer.zero_grad()

        # pred shape must be (batch_size, #classes, seq length)
        logits = logits.permute(0, 2, 1)
        loss = loss_fn(logits, tgt_out)

        loss.backward()
        optimizer.step()

        average_loss += loss.item() / (batch_size * num_batches)

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
    logs["logs"].append(f"total training time: {process_time() - start_time}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(args.config)

    model_id = config.model_id
    train_id = config.train_id

    # params are only used when there is no model file with model_id
    num_tokens = config.num_tokens
    d_model = config.d_model
    padding_idx = config.padding_idx
    num_heads = config.num_heads
    num_encoder_layers = config.num_encoder_layers
    num_decoder_layers = config.num_decoder_layers
    dropout = config.dropout

    N_sites = config.N_sites

    learn_rate = config.learn_rate
    betas = config.betas
    epochs = config.epochs
    batch_size = config.batch_size

    device = "cuda"

    model_file = _results_dir(f"model_{model_id}.params")
    checkpoint_file = _results_dir(f"model_{model_id}__id_{train_id}.pth")

    logs = {"loss": [], "val_loss": [], "logs": []}

    if os.path.isfile(model_file):
        print(f"Model loaded from file {model_file}")
        model = Transformer.from_file(model_file).to(device)

    else:
        print(f"Creating new model file at {model_file}")
        model = Transformer(
            num_tokens=num_tokens,
            d_model=d_model,
            padding_idx=padding_idx,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        ).to(device)

        model.to_file(model_file)

    opt = torch.optim.Adam(model.parameters(), lr=learn_rate, betas=betas)
    loss_fn = nn.CrossEntropyLoss(ignore_index=4)

    model, opt, current_epoch, logs = load_checkpoint(checkpoint_file, model, opt)

    def map_one_border(N_sites, border_idx):
        map = np.zeros((2, N_sites)).astype("bool")
        map[0, :border_idx] = 1
        map[1, border_idx:] = 1
        return map

    mappings = np.array([map_one_border(N_sites, i) for i in range(1, N_sites)])
    groupings = np.argmax(mappings, axis=1)

    try:
        train(model, epochs, opt, loss_fn, groupings, 330, 20)
    except KeyboardInterrupt:
        print("KeyboardInterrupt recieved")
        current_epoch -= 1

    os.remove("crash_checkpoint.pth")
    save_checkpoint(model, opt, logs, current_epoch, checkpoint_file)
