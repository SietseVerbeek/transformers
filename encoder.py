import yaml

from utils.fs import results_dir as main_results_dir


def _results_dir(filename: str) -> str:
    filename = f"encoder/{filename}"
    return main_results_dir(filename)


class Config:
    def __init__(self, filename) -> None:
        with open(filename, "r") as conf_file:
            config = yaml.safe_load(conf_file.read())

        model = config["model"]

        self.model_id = model["model_id"]
        self.train_id = model["train_id"]

        self.num_tokens = model["num_tokens"]
        self.d_model = model["d_model"]
        self.num_heads = model["num_heads"]
        self.num_layers = model["num_layers"]
        self.num_hidden = model["num_hidden"]

        data = config["data"]
        self.N_sites = data["N_sites"]
        self.train_data_id = data["train_data_id"]

        training = config["training"]
        self.learn_rate = training["learn_rate"]
        self.beta1 = training["beta1"]
        self.beta2 = training["beta2"]
        self.epochs = training["epochs"]
        self.batch_size = training["batch_size"]
