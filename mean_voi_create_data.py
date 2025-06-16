import numpy as np

from utils.fwht import gen_spin_model_batch


def labelings_one_border(length):
    return np.array([[0] * (length - i) + [1] * i for i in range(length)])


if __name__ == "__main__":
    N = 10
    beta_count = 20
    samples = 1000
    set_size = 50

    groupings = labelings_one_border(N)
    betas = np.linspace(0.1, 1, beta_count, endpoint=True)

    data_dict = {}

    for i, grouping in enumerate(groupings):

        border_pos = N - i

        for j, beta in enumerate(betas): 
            src, tgt = gen_spin_model_batch(beta, samples, set_size, grouping[None, ...], device="cpu")

            dict_id = f"{border_pos}_{beta:.2f}_"

            data_dict[dict_id + "src"] = src.numpy()
            data_dict[dict_id + "tgt"] = tgt.numpy()

    np.savez(f"results/pma/dataset__N_{N}__beta_{beta_count}__set_50__samples_{samples}", **data_dict)
