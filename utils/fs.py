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
    model: nn.Module, optimizer: Optimizer, logs: dict[str, list], epoch: int, filename: str
) -> None:
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "logs": logs,
    }
    torch.save(state, filename)


# https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3
def load_checkpoint(
    model: nn.Module, optimizer: Optimizer | None, filename: str
) -> Tuple[nn.Module, Optimizer, int, dict[str, list]]:

    start_epoch = 0
    logs = {
        "loss": [],
        "logs": []
    }

    if os.path.isfile(filename):

        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, weights_only=True)

        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        logs = checkpoint["logs"]

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])

        print(
            "=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"])
        )

    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, logs
