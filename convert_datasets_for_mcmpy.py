from pathlib import Path
import numpy as np

from utils.fs import save_configurations
from utils.test_datasets import get_borders_betas, src_tgt_from_dict

def get_dat_dir(split, is_permuted):

    if is_permuted:
        N = split[4]
        beta_count = split[7]
        set_size = split[10]
        samples = split[13]
        return N, f"data/pma/mcmpy/dataset_permuted__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}/"

    N = split[3]
    beta_count = split[6]
    set_size = split[9]
    samples = split[12]

    return N, f"data/pma/mcmpy/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}/"

if __name__ == "__main__":

    data_dir = Path("data/pma")
    mcmpy_dir = data_dir / "mcmpy"
    mcmpy_dir.mkdir(parents=True, exist_ok=True)

    for file in data_dir.glob("*.npz"):

        split = file.stem.split("_")
        N, dat_dir = get_dat_dir(split, "permuted" in split)
        Path(dat_dir).mkdir(parents=True, exist_ok=True)

        np_file = np.load(file)
        borders, betas = get_borders_betas(np_file)

        i = 0
        for beta in betas:
            for border in borders:
                src, tgt = src_tgt_from_dict(np_file, border, beta)
                for s, t in zip(src, tgt):

                    border_pos = np.argmax(t)
                    border_pos = N if border_pos == 0 else border_pos

                    print(f"{border_pos}_{beta:.2f}_{i}.dat")
                    i += 1
                    save_configurations(s, dat_dir + f"{border_pos}_{beta:.2f}_{i}.dat")
