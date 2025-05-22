from time import perf_counter
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from utils.fs import load_from_txt


class FullyPairwiseDataset(Dataset):
    def __init__(
        self,
        labelings: npt.NDArray[np.uint8],
        betas: npt.NDArray[np.float64],
        set_size: int,
        samples_per_epoch: int = 1000,
        validation=False,
    ) -> None:
        super().__init__()
        start_t = perf_counter()
        
        self.labelings = labelings
        self.betas = betas
        self.set_size = set_size
        self.samples_per_epoch = samples_per_epoch

        N_sites = labelings.shape[-1]

        dirs_array = np.apply_along_axis(
            lambda row: "".join(row.astype(str)),
            axis=1,
            arr=labelings,
        )
        self.data = [[]] * len(dirs_array)
        data_dir = "data/fully_pairwise/val/" if validation else "data/fully_pairwise/" 
        for i, dir in enumerate(dirs_array):
            for j, beta in enumerate(betas):
                filename = data_dir + f"{N_sites}/{dir}/{beta:.2f}"
                samples = load_from_txt(filename)
                self.data[i].append(samples.astype('int'))

        # print(f"data ({len(self.data) * len(self.data[0]) * self.data[0][0].shape}) loaded in {perf_counter() - start_t}")

    def __getitem__(self, index) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        label_idx = np.random.randint(self.labelings.shape[0])
        beta_idx = np.random.randint(self.betas.shape[0])
        choices = np.random.randint(self.data[label_idx][beta_idx].shape[0], size=(self.set_size))
        src = self.data[label_idx][beta_idx][choices]
        return src, self.labelings[label_idx]

    def __len__(self) -> int:
        return self.samples_per_epoch


if __name__ == "__main__":
    border = 5
    labelings = np.array(
        [
            [0] * border + [1] * (20 - border),
            [0] * (border + 1) + [1] * (20 - border - 1),
            [0] * (border + 2) + [1] * (20 - border - 2),
        ]
    )
    betas = np.array([1, 0.9])
    ds = FullyPairwiseDataset(labelings, betas, 20, 3)
