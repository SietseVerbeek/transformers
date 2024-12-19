import yaml

from utils.fs import results_dir as main_results_dir


def _results_dir(filename: str):
    str_ = f"partition/{filename}"
    return main_results_dir(str_)


class Config:
    def __init__(self, filename: str) -> None:
        with open(filename, "r") as conf_file:
            config = yaml.safe_load(conf_file.read())

        model = config["model"]
        self.model_id = model["model_id"]
        self.train_id = model["train_id"]

        # params are only used when there is no model file with model_id
        self.num_tokens = model["num_tokens"]
        self.d_model = model["d_model"]
        self.padding_idx = model["padding_idx"]
        self.num_heads = model["num_heads"]
        self.num_encoder_layers = model["num_encoder_layers"]
        self.num_decoder_layers = model["num_decoder_layers"]
        self.dropout = model["dropout"]

        data = config["data"]
        self.N_sites = data["N_sites"]
        self.train_data_id = data["train_data_id"]
        self.val_data_id = data["val_data_id"]
        self.test_data_id = data["test_data_id"]

        training = config["training"]
        self.learn_rate = training["learn_rate"]
        self.betas = (training["beta1"], training["beta2"])
        self.epochs = training["epochs"]
        self.batch_size = training["batch_size"]
