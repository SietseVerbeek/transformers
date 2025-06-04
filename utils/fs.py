import os
from os.path import isfile
from pathlib import Path
from typing import Tuple
from torch import nn
import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import numpy.typing as npt


def info_dir(filename: str) -> str:
    return f"info/{filename}"


def results_dir(filename: str) -> str:
    return f"results/{filename}"


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    logs: dict[str, list],
    epoch: int,
    filename: str,
    write_logs: bool = False,
) -> None:
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "logs": logs,
    }
    torch.save(state, filename)
    if write_logs:
        with open(filename + ".logs", "a") as file:
            file.write('\n'.join(logs["logs"]))


# https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3
def load_checkpoint(
    filename: str, model: nn.Module | None = None, optimizer: Optimizer | None = None
) -> Tuple[nn.Module, Optimizer, int, dict[str, list]]:
    start_epoch = 0
    logs = {"loss": [], "val_loss": [], "logs": []}

    with open(filename + ".logs", "a") as file:
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            file.write("=> loading checkpoint '{}'\n".format(filename))
            checkpoint = torch.load(filename, weights_only=True)

            start_epoch = checkpoint["epoch"] + 1
            logs = checkpoint["logs"]

            if model:
                model.load_state_dict(checkpoint["state_dict"])

            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer"])

            print(
                "=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"])
            )
            file.write(
                "=> loaded checkpoint '{}' (epoch {})\n".format(filename, checkpoint["epoch"])
            )

        else:
            print("=> no checkpoint found at '{}'".format(filename))
            file.write("=> no checkpoint found at '{}'\n".format(filename))

    return model, optimizer, start_epoch, logs


def load_from_txt(file_name: str) -> npt.NDArray[np.uint8]:
    """
    Load spin configurations from file. Must be text file with spin
    configurations saved as rows. Lines starting with # are considered
    to be comments and skipped.

    Arguments:
        file_name - name of spin configurations file

    Output:
        data - numpy array containing spin configurations
    """

    with open(file_name, "r") as file:
        data = [
            list(map(int, line.strip()))
            for line in file
            if not line.strip().startswith("#") and line.strip()
        ]

    return np.array(data, dtype=np.uint8)

def save_configurations(
            configurations: npt.NDArray,
            file_name: str,
            header=""
        ) -> None:
    """
    Save MCM spin configurations in text and npy file.

    Arguments:
        configurations - numpy array with integer representation of 
                spin configurations
        N - number of sites
        file_name - name of file to save

    Output:
        None - Writes files: {filename}
    """

    if isfile(file_name):
        print(f"File {file_name} already exists")
        raise FileExistsError

    np.savetxt(file_name, configurations, fmt="%s", delimiter="", header=header)

