import numpy as np
from torch.utils.data import DataLoader

from utils.datasets import PartitionDataset


dataset = PartitionDataset("data/partition_sets/N_6_id_0.npz")
dataloader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True,
    pin_memory=True,
    pin_memory_device="cuda",
)

for src, tgt in dataloader:
    print("src")
    print(np.unique(src, return_counts=True, axis=0)[1])
    print("tgt")
    print(np.unique(tgt, return_counts=True, axis=0))
