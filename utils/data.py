from typing import Dict

import torch
from torch.distributions.dirichlet import Dirichlet
import numpy.typing as npt
import numpy as np


def get_partition_batch(
    batch_size: int,
    set_size: int,
    groupings: npt.NDArray[np.int_],
    model_dists: Dict[int, Dirichlet],
):
    N_sites = groupings.shape[-1]
    configs_per_map = batch_size // groupings.shape[0]
    batch_size = configs_per_map * groupings.shape[0]

    icc_sizes = np.apply_along_axis(np.bincount, axis=1, arr=groupings)

    tgt = torch.tensor(np.repeat(groupings, configs_per_map, axis=0), dtype=torch.long)
    src = torch.empty((batch_size, N_sites * set_size), dtype=torch.long)

    for idx in range(groupings.shape[0]):

        left_icc = icc_sizes[idx, 0]
        left_dist = model_dists[left_icc]

        right_icc = icc_sizes[idx, 1]
        right_dist = model_dists[right_icc]

        left_models = left_dist.sample((configs_per_map, )).cumsum(dim=-1)
        right_models = right_dist.sample((configs_per_map, )).cumsum(dim=-1)

        throws = torch.rand((2, configs_per_map, set_size))

        left_ints = torch.searchsorted(left_models, throws[0])
        right_ints = torch.searchsorted(right_models, throws[1])

        left_binary = (left_ints.unsqueeze(-1) >> torch.arange(left_icc - 1, -1, -1)) & 1
        right_binary = (right_ints.unsqueeze(-1) >> torch.arange(right_icc - 1, -1, -1)) & 1

        src[idx * configs_per_map: (idx + 1) * configs_per_map] = torch.cat((left_binary, right_binary), dim=-1).reshape(configs_per_map, -1)

    return src, tgt

if __name__ == "__main__":
    from time import perf_counter
    def map_one_border(N_sites, border_idx):

        map = np.zeros((2, N_sites)).astype('bool')
        map[0, :border_idx] = 1
        map[1, border_idx:] = 1
        return map

    N_sites = 6
    mappings = np.array([map_one_border(N_sites, i) for i in range(1, N_sites)])
    groupings = np.argmax(mappings, axis=1)

    print(groupings)

    dists: Dict[int, Dirichlet] = dict()

    for N in range(1, 20):
        dists.update([(N, Dirichlet(torch.full((2 ** N, ), .5)))])

    start = perf_counter()
    get_partition_batch(300, 20, groupings, dists)
    print(perf_counter() - start)
