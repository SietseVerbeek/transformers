import os
from time import process_time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from models.transformer import Transformer
from utils.datasets import MCMDataset
from utils.fs import info_dir, results_dir
from utils.masking import create_mask


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
    logs: list[str],
    device="cuda",
):
    start_time = process_time()

    for i in range(1, epochs + 1):
        epoch_start_time = process_time()

        print("=" * 30, f"starting epoch {i}", "=" * 30)
        logs.extend(("=" * 30, f"starting epoch {i}", "=" * 30, "\n"))

        train_loss = sequence_train_epoch(
            model, optimizer, loss_fn, train_dataloader, device
        )

        validate_loss = sequence_validation_epoch(
            model, loss_fn, validation_dataloader, device
        )
        print("training loss: ", train_loss)
        print("validation loss: ", validate_loss)
        print("epoch time: ", process_time() - epoch_start_time)
        logs.extend(
            (
                f"training loss: {train_loss}\n",
                f"validation loss: {validate_loss}\n",
                f"epoch time: {process_time() - epoch_start_time}\n",
            )
        )

    print("total training time: ", process_time() - start_time, "s")
    logs.extend((f"total training time: {process_time() - start_time} s\n"))


def main(
    d_model: int,
    data_file: str,
    epochs: int,
    filename: str,
    learn_rate: float,
    num_decoder_layers: int,
    num_encoder_layers: int,
    num_heads: int,
    use_cuda: bool,
):
    info_file = info_dir(f"{filename}.info")
    weights_file = info_dir(f"{filename}.pth")
    tgts_file = info_dir(f"{filename}.npz")

    log_file = results_dir(f"{filename}.log")
    logs = []

    dataset = MCMDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=500, shuffle=True)

    unique_tgts, tgt_counts = dataset.unique_tgts
    tgt_frac = tgt_counts / len(dataset)
    np.savez(tgts_file, unique_tgts=unique_tgts, tgt_frac=tgt_frac)

    validation_dataset = MCMDataset(data_file, train=False)
    val_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

    device = "cuda" if use_cuda else "cpu"

    model = Transformer(
        num_tokens=5,
        d_model=d_model,
        padding_idx=4,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=0.1,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=learn_rate, betas=(0.9, 0.98))
    loss_fn = nn.CrossEntropyLoss(ignore_index=4)

    if os.path.isfile(weights_file):
        model.load_state_dict(torch.load(weights_file, weights_only=True))

        print(f"model weights loaded from {weights_file}")
        logs.extend(
            (
                "\n",
                "=" * 80 + "\n",
                f"Starting training with weights loaded from {weights_file}\n",
            )
        )

    else:
        logs.extend(
            ("=" * 80 + "\n", "Starting training with newly initialised weights\n")
        )

    logs.extend(
        (
            f"datafile: {data_file}\n",
            f"training sequence count: {len(dataset)}\n",
            f"validation sequence count: {len(validation_dataset)}\n",
        )
    )

    # generation is dependent on this file, change with care
    if not os.path.isfile(info_file):
        with open(info_file, "w+") as file:
            lines = [
                "Sequence generation Transformer model info\n",
                f"Model dimensions: d_model = {d_model}\n",
                f"Attention heads: num_heads = {num_heads}\n",
                f"Encoder layers: num_encoder_layers = {num_encoder_layers}\n",
                f"Decoder layers: num_decoder_layers = {num_decoder_layers}\n",
                "=" * 80 + "\n\n",
            ]
            file.writelines(lines)

    try:
        train(model, epochs, opt, loss_fn, dataloader, val_dataloader, logs, device)

    except KeyboardInterrupt:
        logs.append("training stopped with KeyboardInterrupt\n")
        print("training stopped due to KeyboardInterrupt")

    torch.save(model.state_dict(), weights_file)

    with open(log_file, "a+") as file:
        file.writelines(logs)

    print(f"weights saved in {weights_file}")
    print(f"logs written to {log_file}")


if __name__ == "__main__":
    from config import (
        D_MODEL,
        DATA_FILE,
        EPOCHS,
        FILENAME,
        LEARN_RATE,
        NUM_DECODER_LAYERS,
        NUM_ENCODER_LAYERS,
        NUM_HEADS,
        USE_CUDA,
    )

    main(
        D_MODEL,
        DATA_FILE,
        EPOCHS,
        FILENAME,
        LEARN_RATE,
        NUM_DECODER_LAYERS,
        NUM_ENCODER_LAYERS,
        NUM_HEADS,
        USE_CUDA,
    )
