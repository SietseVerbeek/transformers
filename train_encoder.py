from time import process_time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import entropy
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from models.encoder import SequenceEncoder
from utils.datasets import MCMEncoderDataset, load_from_txt
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


def sequence_validation_epoch(
    model: SequenceEncoder, loss_fn: _Loss, dataloader: DataLoader, device="cuda"
):
    model.eval()
    average_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            logits = model(src)

            # pred shape must be (batch_size, #classes, seq length)
            logits = logits.permute(0, 2, 1)
            loss = loss_fn(logits, tgt)
            average_loss += loss.item() / len(dataloader)

    return average_loss


def train(
    model: SequenceEncoder,
    epochs: int,
    optimizer: Optimizer,
    loss_fn: _Loss,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    device="cuda",
    losses: list[float] = [],
):
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")

    start_time = process_time()
    global current_epoch

    for current_epoch in range(current_epoch, current_epoch + epochs):
        epoch_start_time = process_time()

        print("=" * 30, f"starting epoch {current_epoch}", "=" * 30)
        logs.extend(("=" * 30, f"starting epoch {current_epoch}", "=" * 30, "\n"))

        train_loss = train_epoch(model, optimizer, loss_fn, train_dataloader, device)
        losses.append(train_loss)

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


if __name__ == "__main__":
    logs = []

    learn_rate = 10e-4
    betas = (0.9, 0.99)

    weights_file = "3_3_mcm_1.pth"

    model = SequenceEncoder(
        num_tokens=3, d_model=32, num_heads=4, num_layers=6, num_hidden=400
    ).to("cuda")
    # model.load_state_dict(torch.load(weights_file, weights_only=True))

    opt = torch.optim.Adam(model.parameters(), lr=learn_rate, betas=betas)
    loss_fn = nn.CrossEntropyLoss()

    model, opt, current_epoch, loss = load_checkpoint(model, opt, "encoder.pth")
    data_file = "data/N_sites_6/sites_3_3__id_1__map_id_5"

    dataset = MCMEncoderDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=500, shuffle=True)

    validation_dataset = MCMEncoderDataset(data_file, train=False)
    val_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

    try:
        train(
            model,
            15,
            opt,
            loss_fn,
            dataloader,
            val_dataloader,
            losses=loss,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt recieved")

    torch.save(model.state_dict(), 'encoder.pth')
    save_checkpoint(model, opt, loss, current_epoch, 'encoder.pth')

    model.eval()

    def generate(num_sequences, seq_length, model=model):
        src = torch.full((num_sequences, 1), 2, device="cuda")
        for i in range(seq_length):
            logits = model(src)
            probs = torch.softmax(logits[:, -1], -1).cumsum(-1)

            throw = torch.rand(num_sequences, 1).to("cuda")
            idx = torch.searchsorted(probs, throw)

            src = torch.cat((src, idx), dim=-1)

        return src[:, 1:]

    seq_length = 6

    num_sequences = 500
    batches = 20

    total_sequences = num_sequences * batches
    out = np.ndarray((total_sequences, seq_length))

    for i in range(batches):
        print("gen", i, end="\r")
        out[i * num_sequences : (i + 1) * num_sequences] = generate(
            num_sequences, seq_length
        ).cpu()

    configs_unique, configs_counts = np.unique(
        load_from_txt(data_file), axis=0, return_counts=True
    )
    in_dist = configs_counts / configs_counts.sum()
    in_dist[in_dist == 0] = 10e-10

    out_mask = np.any(np.all(out[..., None, :] == configs_unique, axis=-1), axis=-1)

    out_unique, _ = np.unique(out[out_mask], axis=0, return_counts=True)
    out_counts = np.sum(np.all(out[..., None, :] == configs_unique, axis=-1), axis=0)
    out_dist = out_counts / out_counts.sum()
    out_dist[out_dist == 0] = 10e-10

    plt.plot(in_dist, label="in")
    plt.plot(out_dist, label="generated")
    plt.title(
        f"error frac: {np.sum(~out_mask) / total_sequences}, {entropy(in_dist, out_dist)}"
    )
    plt.legend()
    plt.show()

    # for i, data in enumerate(dataloader):
    #     if i == 3:
    #         break
    #     src, tgt = data
    #     src, tgt = src.to("cuda"), tgt.to("cuda")
    #     print(src)
    #     print(model(src))
    #     print(tgt)
