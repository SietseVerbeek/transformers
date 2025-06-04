import numpy as np
from pathlib import Path
import numpy.typing as npt

from utils.sbm import hgsbm
from utils.fwht import generate_data, int_to_binary_array
from utils.fs import save_configurations

def sample_models(q: int, m: int, model_count:int=50, mu_count:int=20):

    count = 0
    mu_list, delta_mu = np.linspace(0, 1, mu_count, retstep=True)
    models = np.ndarray((mu_count, model_count), dtype=object)

    for i, mu in enumerate(mu_list):
        for j in range(model_count):

            edges, mu_r = hgsbm(q, m, mu, m - 1, 2, prob=False, verbose=False)
            count += 1  
            while abs(mu_r - mu) > delta_mu:
                count += 1  
                edges, mu_r = hgsbm(q, m, mu, m - 1, 2, prob=False, verbose=False)

            models[i, j] = [np.power(2,edge).sum() for edge in edges]

    return models, mu_list

def _gen_data(models: npt.NDArray, mu_list: npt.NDArray, beta_list: npt.NDArray, N: int, n_samples: int):

    Path(f"data/sbm/{N}").mkdir(parents=True, exist_ok=True)
    states = int_to_binary_array(np.arange(2 ** N), N)

    for model_arr, mu in zip(models, mu_list):
        for beta in beta_list:
            for i, model in enumerate(model_arr):

                dir = f"data/sbm/{N}/mu_{mu:.2f}/beta_{beta:.2f}"
                Path(dir).mkdir(parents=True, exist_ok=True)

                data = states[generate_data(model, np.ones(len(model)) * beta, N, np.arange(2 ** N), n_samples)]

                filename = dir + f"/{i}"
                save_configurations(data, filename)


if __name__ == "__main__":

    Path("data/sbm").mkdir(parents=True, exist_ok=True)

    q = 2
    m = 5
    N = q * m
    datasets_per_model = 10
    samples_per_dataset = 50
    n_samples = datasets_per_model * samples_per_dataset

    models, mu_list = sample_models(q, m, model_count=5, mu_count=5)
    beta_list = np.linspace(.1, 1, 10, endpoint=True)
    _gen_data(models, mu_list, beta_list, N, n_samples)
