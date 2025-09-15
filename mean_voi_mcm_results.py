from pathlib import Path
from time import perf_counter
import numpy as np
from utils.test_datasets import get_borders_betas, get_dataset
from mcmpy import Data, MCM, MCMSearch
from collections import defaultdict

from utils.tests import batch_normalized_mi, batch_normalized_vi


def mcmpy_array_to_groupings(array):
    return np.argmax(array, axis=0)


if __name__ == "__main__":
    N = 10
    beta_count = 20
    samples = 1000
    set_size = 50

    def labelings_one_border(length):
        return np.array([[0] * (length - i) + [1] * i for i in range(length)])

    targets = labelings_one_border(N)

    dat_dir = Path(
        f"data/mcmpy/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}/"
    )

    searcher = MCMSearch()

    output_set = defaultdict(list)
    tgt_set = defaultdict(list)

    for file in sorted(dat_dir.iterdir())[:10]:

        split = file.name.split("_")
        border_idx = int(split[0])
        beta = float(split[1])

        data = Data(str(file), N, 2)
        mcm = searcher.exhaustive(data)
        output = mcmpy_array_to_groupings(mcm.array)

        output_set[beta].append(output)
        tgt_set[beta].append(border_idx)

    betas = sorted(output_set.keys())

    voi_measurements = np.empty((len(betas), samples))
    mean_voi = np.empty(len(betas))
    std_voi = np.empty(len(betas))

    nmi_measurements = np.empty((len(betas), samples))
    mean_nmi = np.empty(len(betas))
    std_nmi = np.empty(len(betas))

    for i, beta in enumerate(betas):
        vois = batch_normalized_vi(np.array(output_set[beta]), targets[tgt_set[beta]])
        nmis = batch_normalized_mi(np.array(output_set[beta]), targets[tgt_set[beta]])

        voi_measurements[i] = vois
        nmi_measurements[i] = nmis

        mean_voi[i] = vois.mean()
        std_voi[i] = vois.std()

        mean_nmi[i] = nmis.mean()
        std_nmi[i] = nmis.std()

    results_path = Path("results/mcmpy/")
    results_path.mkdir(parents=True, exist_ok=True)
    out_file = results_path / f"dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}"

    np.savez(
        out_file,
        betas=betas,
        mean_voi=mean_voi,
        std_voi=std_voi,
        mean_nmi=mean_nmi,
        std_nmi=std_nmi,
        vois = voi_measurements,
        nmis = nmi_measurements,
    )
