from time import perf_counter

import numpy as np
import numpy.typing as npt
from numba import njit
import torch


@njit
def normalized_vi(results, truth):
    n = len(results)
    k1 = np.max(results) + 1
    k2 = np.max(truth) + 1

    # contingency matrix counts overlapping variables in communities
    contingency = np.zeros((k1, k2), dtype=np.int32)
    for i in range(n):
        contingency[results[i], truth[i]] += 1

    # row and col sums give cardinality of a community
    row_sums = np.sum(contingency, axis=1)
    col_sums = np.sum(contingency, axis=0)

    mi = 0.0
    joint_entropy = 0.0

    for i in range(k1):
        for j in range(k2):
            intersection = contingency[i, j]
            if intersection > 0:
                pij = intersection / n
                pi = row_sums[i] / n
                pj = col_sums[j] / n
                mi += pij * np.log(pij / (pi * pj))
                joint_entropy -= pij * np.log(pij)

    if joint_entropy == 0:
        return 0

    return 1.0 - (mi / joint_entropy)


@njit
def normalized_vi_loop(arr1, arr2):
    num_clusterings = arr1.shape[0]
    output = np.zeros(num_clusterings, dtype=np.float64)

    for i in range(num_clusterings):
        output[i] = normalized_vi(arr1[i], arr2[i])

    return output


def torch_check(input):
    if isinstance(input, torch.Tensor):
        if input.is_cuda:
            return input.cpu().numpy()
        return input.numpy()
    return input


def batch_normalized_vi(results, truth):
    results = torch_check(results)
    truth = torch_check(truth)

    return normalized_vi_loop(results, truth)


# compile functions
l1 = np.array([[0, 0], [0, 1]])
l2 = np.array([[0, 1], [0, 1]])

normalized_vi_loop(l1, l2)

if __name__ == "__main__":

    def labelings_one_border(length):
        return np.array([[0] * (length - i) + [1] * i for i in range(length)])

    N = 20
    groupings = labelings_one_border(N)
    groupings = groupings.repeat(1000, axis=0)
    test = np.repeat(
        np.array([0, 1] * 10)[None, ...], groupings.shape[0] * 1000, axis=0
    )

    start = perf_counter()
    out = batch_normalized_vi(groupings, test)
    print(out.shape)
    print("time", perf_counter() - start)
