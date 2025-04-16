import numpy.typing as npt
import numpy as np
from numba import njit
import torch


@njit(fastmath=True)
def fwht(u):
    v = u.copy()
    h = 1
    lv = len(v)
    while h < lv:
        for i in range(0, lv, h * 2):
            for j in range(i, i + h):
                x = v[j]
                y = v[j + h]
                v[j] = x + y
                v[j + h] = x - y
        h *= 2
    return v


def generate_data(model, pars, n, states, N=100000):
    """
    model:   list/array of integer values corresponding to interactions
    pars:    list/array of real valued parameters of same length as model
    n:       number of variables
    states:  list/array of all 2**n states in desired format
    N:       number of data samples to return
    """

    if len(model) != len(pars):
        raise ValueError("number of parameters should match number of interactions")

    g = np.zeros(2**n)
    g[model] = pars
    u = np.exp(fwht(g))
    p = u / np.sum(u)
    data = np.random.choice(states, N, p=p)

    return data


def int_to_binary_array(configurations: npt.NDArray, N: int) -> npt.NDArray[np.uint8]:
    """
    Converts numpy array of integers into a 2d array where the second axis
    is the binary representation of the integers with width N

    Arguments:
        configurations - numpy array of integers
        N   - width of binary representation (amount of sites of MCM)

    Output:
        binary - binary representations of integers
            (len(configurations), N) numpy array
    """

    binary = np.zeros((configurations.shape[0], N), dtype=np.uint8)

    for i in range(N):
        binary[:, N - i - 1] = (configurations >> i) & 1

    return binary


def pairwise_interactions(labels):
    """
    Creates integer representations of pairwise interactions between members of a group
    defined in the groupings variable

    Arguments:
        groupings: boolean array

    Output:
        interactions: integer array
    """

    interactions = []

    N_sites = labels.shape[-1]
    num_groups = max(labels) + 1
    idx_groups = [[] for i in range(num_groups)]

    for idx, group in enumerate(labels):
        idx_groups[group].append(idx)

    for group in idx_groups:
        for i in range(len(group) - 1):  # Iterate over neighbor pairs
            pair_mask = (1 << (N_sites - 1 - group[i])) | (1 << (N_sites - 1 - group[i + 1]))
            interactions.append(pair_mask)

    return np.array(interactions)


def gen_spin_model_batch(
    weight: float,
    batch_size: int,
    set_size: int,
    groupings: npt.NDArray[np.bool_],
    device="cuda",
):
    N_sites = groupings.shape[-1]
    N_maps = groupings.shape[0]
    configs_per_map = batch_size // N_maps
    batch_size = configs_per_map * N_maps

    tgt = torch.tensor(
        np.repeat(groupings, configs_per_map, axis=0), dtype=torch.long, device=device
    )
    src = torch.empty((batch_size, set_size, N_sites), dtype=torch.long, device=device)

    model = np.arange(2**N_sites)
    states = int_to_binary_array(model, N_sites)

    for idx in range(N_maps):
        weights = np.zeros(2**N_sites)
        interactions = pairwise_interactions(groupings[idx])
        for i in interactions:
            weights[interactions] = weight

        choices = generate_data(model, weights, N_sites, model, configs_per_map * set_size)
        src[idx * configs_per_map : (idx + 1) * configs_per_map] = torch.tensor(
            states[choices].reshape(-1, set_size, N_sites)
        )

    permutation = torch.randperm(src.size()[0])

    return src[permutation], tgt[permutation]

if __name__ == "__main__":
    # n = 6
    # model = np.arange(2**n)
    # states = int_to_binary_array(model, n)
    # g = np.zeros_like(model)

    # for i in range(n - 1):
    #     if i != 3:
    #         g[3 << i] = 3

    # choices = generate_data(model, g, n, model)
    # out = states[choices]
    # print(np.unique(out, axis=0, return_counts=True))
    # for i in np.unique(out):
    #     print(bin(i))

    # Example input
    groups = np.array([
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ])

    print(gen_spin_model_batch(10, 12, 5, groups))
    # print(pairwise_interactions(groups))
