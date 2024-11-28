import os
from typing import Tuple
from torch import nn
import torch
from torch.optim.optimizer import Optimizer


def info_dir(filename: str) -> str:
    return f"info/{filename}"


def results_dir(filename: str) -> str:
    return f"results/{filename}"


def save_checkpoint(
    model: nn.Module, optimizer: Optimizer, loss_log: list, epoch: int, filename: str
) -> None:
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_log": loss_log,
    }
    torch.save(state, filename)


# https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3
def load_checkpoint(
    model: nn.Module, optimizer: Optimizer, filename: str
) -> Tuple[nn.Module, Optimizer, int, list]:

    start_epoch = 0
    loss_log = []

    if os.path.isfile(filename):

        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, weights_only=True)

        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_log = checkpoint["loss_log"]

        print(
            "=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"])
        )

    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, loss_log
