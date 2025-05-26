import yaml

from utils.fs import results_dir as main_results_dir


def _results_dir(filename: str):
    str_ = f"pma/{filename}"
    return main_results_dir(str_)


class Config:
    def __init__(self, filename: str) -> None:
        with open(filename, "r") as conf_file:
            config = yaml.safe_load(conf_file.read())

        info = config["info"]
        self.model_id = info["model_id"]
        self.train_id = info["train_id"]

        # params are only used when there is no model file with model_id
        model = config["model"]
        self.embed_dim = model["embed_dim"]
        self.enc_layers = model["enc_layers"]
        self.enc_heads = model["enc_heads"]
        self.pma_heads = model["pma_heads"]
        self.dec_heads = model["dec_heads"]
        self.queries = model["queries"]

        training = config["training"]
        self.N_sites = training["N_sites"]
        self.set_size = training["set_size"]
        self.batch_size = training["batch_size"]
        self.learn_rate = training["learn_rate"]
        self.betas = (training["beta1"], training["beta2"])
        self.max_epochs = training["max_epochs"]
        self.train_beta_temps = training["train_beta_temps"]
        self.beta_temp = training["beta_temp"]
