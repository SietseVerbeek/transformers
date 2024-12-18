import argparse
import os
from time import process_time

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.transformer import Transformer
from partition import Config, _results_dir
from utils.datasets import PartitionDataset
from utils.fs import load_checkpoint, save_checkpoint
from utils.masking import create_mask


def _data_dir(filename: str):
    return f"data/partition_sets/{filename}"


def sequence_train_epoch(
    model: Transformer,
    optimizer: Optimizer,
    loss_fn: _Loss,
    dataloader: DataLoader,
    device="cuda",
):
    model.train()
    average_loss = 0

    for src, tgt in dataloader:
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

        average_loss += loss.item() / len(dataloader)

    return average_loss


def sequence_validation_epoch(
    model: Transformer, loss_fn: _Loss, dataloader: DataLoader, device="cuda"
):
    model.eval()
    average_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
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

            # pred shape must be (batch_size, #classes, seq length)
            logits = logits.permute(0, 2, 1)
            loss = loss_fn(logits, tgt_out)
            average_loss += loss.item() / len(dataloader)

    return average_loss


def train(
    model: Transformer,
    epochs: int,
    optimizer: Optimizer,
    loss_fn: _Loss,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    device="cuda",
):
    start_time = process_time()
    global current_epoch, logs

    for current_epoch in range(current_epoch, current_epoch + epochs):
        epoch_start_time = process_time()

        print("=" * 30, f"starting epoch {current_epoch}", "=" * 30)

        train_loss = sequence_train_epoch(
            model, optimizer, loss_fn, train_dataloader, device
        )
        logs["loss"].append(train_loss)

        validate_loss = sequence_validation_epoch(
            model, loss_fn, validation_dataloader, device
        )
        print("training loss: ", train_loss)
        print("validation loss: ", validate_loss)
        print("epoch time: ", process_time() - epoch_start_time)

        save_checkpoint(model, optimizer, logs, current_epoch, "crash_checkpoint.pth")

    print("total training time: ", process_time() - start_time, "s")


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
    train_data_id = config.train_data_id
    val_data_id = config.val_data_id

    learn_rate = config.learn_rate
    betas = config.betas
    epochs = config.epochs
    batch_size = config.batch_size

    device = "cuda"

    model_file = _results_dir(f"model_{model_id}.params")
    checkpoint_file = _results_dir(f"model_{model_id}__id_{train_id}.pth")

    train_datafile = _data_dir(f"N_{N_sites}_id_{train_data_id}.npz")
    val_datafile = _data_dir(f"N_{N_sites}_id_{val_data_id}.npz")

    logs = {"loss": [], "logs": []}

    dataset = PartitionDataset(train_datafile)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        pin_memory_device="cuda",
    )

    val_dataset = PartitionDataset(val_datafile, validation=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

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

    try:
        train(
            model,
            epochs,
            opt,
            loss_fn,
            dataloader,
            dataloader,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt recieved")
        current_epoch -= 1

    os.remove("crash_checkpoint.pth")
    save_checkpoint(model, opt, logs, current_epoch, checkpoint_file)
