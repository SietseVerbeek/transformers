import argparse
from time import process_time
import os

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from encoder import Config, _results_dir
from models.encoder import SequenceEncoder
from utils.datasets import MCMEncoderDataset
from utils.fs import load_checkpoint, save_checkpoint


def train_epoch(
    model: SequenceEncoder,
    optimizer: Optimizer,
    loss_fn: _Loss,
    dataloader: DataLoader,
    device="cuda",
) -> float:
    model.train()
    average_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        logits = model(src)
        optimizer.zero_grad()

        # pred shape must be (batch_size, #classes, seq length)
        logits = logits.permute(0, 2, 1)
        loss = loss_fn(logits, tgt)

        loss.backward()
        optimizer.step()

        average_loss += loss.item() / len(dataloader)

    return average_loss


def train(
    model: SequenceEncoder,
    epochs: int,
    optimizer: Optimizer,
    loss_fn: _Loss,
    train_dataloader: DataLoader,
    device="cuda",
):
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")

    start_time = process_time()
    global current_epoch

    for current_epoch in range(current_epoch, current_epoch + epochs):
        epoch_start_time = process_time()

        print("=" * 30, f"starting epoch {current_epoch}", "=" * 30)

        train_loss = train_epoch(model, optimizer, loss_fn, train_dataloader, device)
        logs["loss"].append(train_loss)

        print("training loss: ", train_loss)
        print("epoch time: ", process_time() - epoch_start_time)

    print("total training time: ", process_time() - start_time, "s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(args.config)

    model_id = config.model_id
    train_id = config.train_id

    num_tokens = config.num_tokens
    d_model = config.d_model
    num_heads = config.num_heads
    num_layers = config.num_layers
    num_hidden = config.num_hidden

    N_sites = config.N_sites
    train_data_id = config.train_data_id

    learn_rate = config.learn_rate
    betas = (config.beta1, config.beta2)
    epochs = config.epochs
    batch_size = config.batch_size

    device = "cuda"

    model_file = _results_dir(f"model_{model_id}.params")
    checkpoint_file = _results_dir(f"model_{model_id}__id_{train_id}.pth")

    train_datafile = f"data/N_sites_{N_sites}/sites_{N_sites}__id_{train_data_id}"

    logs = {"loss": [], "logs": []}

    dataset = MCMEncoderDataset(train_datafile)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        pin_memory_device="cuda",
    )

    if os.path.isfile(model_file):
        print(f"Model loaded from file {model_file}")
        model = SequenceEncoder.from_file(model_file).to(device)

    else:
        model = SequenceEncoder(
            num_tokens=num_tokens,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            num_hidden=num_hidden,
        ).to("cuda")

    opt = torch.optim.Adam(model.parameters(), lr=learn_rate, betas=betas)
    loss_fn = nn.CrossEntropyLoss()

    model, opt, current_epoch, logs = load_checkpoint(checkpoint_file, model, opt)

    try:
        train(
            model,
            epochs,
            opt,
            loss_fn,
            dataloader,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt recieved")
        current_epoch -1

    save_checkpoint(model, opt, logs, current_epoch, checkpoint_file)
