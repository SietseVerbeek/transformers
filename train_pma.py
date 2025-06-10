import argparse
import os
from time import process_time
import wandb

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from models.PMATransformer import PMA_GroupingModel
from pma import Config, _results_dir
from utils.data import get_set_partition_batch
from utils.fs import load_checkpoint, save_checkpoint
from utils.fwht import gen_spin_model_batch
from utils.masking import create_mask


def sequence_train_epoch(
    model: PMA_GroupingModel,
    optimizer: Optimizer,
    loss_fn: _Loss,
    gen_func,
    batch_size: int,
    num_batches: int,
    set_size: int,
    groupings: npt.NDArray[np.int_],
    device="cuda",
):
    model.train()
    average_loss = 0

    for i in range(num_batches):
        src, tgt = gen_func(batch_size, set_size, groupings)
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
    model: PMA_GroupingModel,
    max_epochs: int,
    optimizer: Optimizer,
    loss_fn: _Loss,
    gen_func,
    groupings: npt.NDArray[np.int_],
    num_batches: int,
    set_size: int,
    device="cuda",
):
    start_time = process_time()
    global current_epoch, logs

    while not converged(logs["loss"], delta=1e-3) and current_epoch < max_epochs:
        epoch_start_time = process_time()

        print("=" * 30, f"starting epoch {current_epoch}", "=" * 30)
        logs["logs"].append("=" * 30 + f"starting epoch {current_epoch}" + "=" * 30)

        train_loss = sequence_train_epoch(
            model,
            optimizer,
            loss_fn,
            gen_func,
            batch_size,
            num_batches,
            set_size,
            groupings,
            device=device,
        )
        logs["loss"].append(train_loss)

        print("training loss: ", train_loss)
        logs["logs"].append(f"training loss:  {train_loss}")
        run.log({"loss": train_loss})
        print("epoch time: ", process_time() - epoch_start_time)
        logs["logs"].append(f"epoch time: {process_time() - epoch_start_time}")
        run.log({"epoch_time": process_time() - epoch_start_time})

        save_checkpoint(model, optimizer, logs, current_epoch, checkpoint_file)

        current_epoch += 1

    print(f"total training time: {process_time() - start_time}s")
    logs["logs"].append(f"total training time: {process_time() - start_time}s\n")


def converged(losses: list[float], delta=1e-4, patience: int = 5) -> bool:
    if len(losses) < patience:
        return False
    if max(losses[-patience:]) - min(losses[-patience:]) < delta:
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    c = Config(args.config)

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="sietse-verbeek-university-of-amsterdam",
        # Set the wandb project where this run will be logged.
        project="testing",
        # Track hyperparameters and run metadata.
        config={
            "model_id": c.model_id,
            "id": c.train_id,
            "learning_rate": c.learn_rate,
            "N_sites": c.N_sites,
            "set_size": c.set_size,
            "batch_size": c.batch_size,
            "learn_rate": c.learn_rate,
            "betas": c.betas,
            "max_epochs": c.max_epochs,
            "train_beta_temps": c.train_beta_temps,
            "permuted": c.permuted,
        },
    )

    batch_size = c.batch_size

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

    def labelings_one_border(length):
        return np.array([[0] * (length - i) + [1] * i for i in range(length)])

    groupings = labelings_one_border(c.N_sites)

    def generate_func(batch_size, set_size, groupings):
        temps = c.train_beta_temps
        src_list, tgt_list = [], []
        for temp in temps:
            src, tgt = gen_spin_model_batch(temp, batch_size // len(temps), set_size, groupings)
            src_list.append(src)
            tgt_list.append(tgt)
        src = torch.cat(src_list)
        tgt = torch.cat(tgt_list)

        if c.permuted:
            perm = torch.stack(
                [torch.randperm(src.size()[-1], device=device) for _ in range(src.size()[0])]
            )
            tgt = torch.gather(tgt, 1, perm)

            perm = perm.unsqueeze(1).expand(-1, src.size()[1], -1)
            src = torch.gather(src, 2, perm)
        return src, tgt

    try:
        train(
            model=model,
            max_epochs=c.max_epochs,
            optimizer=opt,
            loss_fn=loss_fn,
            gen_func=generate_func,
            groupings=groupings,
            num_batches=330,
            set_size=c.set_size,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt recieved")

    save_checkpoint(model, opt, logs, current_epoch, checkpoint_file, write_logs=True)
    run.finish()
