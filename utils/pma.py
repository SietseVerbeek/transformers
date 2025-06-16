import argparse
import os

from models.PMATransformer import PMA_GroupingModel
from pma import Config, _results_dir
from utils.fs import load_checkpoint


def pma_from_config(device: str = "cuda"):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    c = Config(args.config)
    model_file = _results_dir(f"model_{c.model_id}.params")
    checkpoint_file = _results_dir(f"model_{c.model_id}__id_{c.train_id}.pth")

    if os.path.isfile(model_file):
        model = PMA_GroupingModel.from_file(model_file).to(device)
    else:
        raise FileNotFoundError(f"model file with id {model_file} not found")

    model, _, _, logs = load_checkpoint(checkpoint_file, model, None)

    model.eval()

    return model, c, logs
