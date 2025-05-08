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
        sample_size: int = 1000,
        validation=False,
    ) -> None:
        super().__init__()
        N_sites = labelings.shape[-1]
        self.data = np.ndarray(
            (len(labelings) * len(betas) * sample_size, set_size, N_sites),
            dtype=np.int_,
        )
        self.targets = np.repeat(labelings, len(betas) * sample_size, axis=0)

        dirs_array = np.apply_along_axis(
            lambda row: "".join(row.astype(str)),
            axis=1,
            arr=labelings,
        )
        data_dir = "data/fully_pairwise/val/" if validation else "data/fully_pairwise/" 
        for i, dir in enumerate(dirs_array):
            for j, beta in enumerate(betas):
                print(i, j)
                filename = data_dir + f"{N_sites}/{dir}/{beta:.2f}"
                samples = load_from_txt(filename)

                choices = np.random.randint(
                    samples.shape[0], size=(sample_size, set_size)
                )
                start_idx = (i * len(betas) + j) * sample_size
                end_idx = (i * len(betas) + j + 1) * sample_size
                self.data[start_idx:end_idx] = samples[choices]

    def __getitem__(self, index) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return self.data.shape[0]


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
