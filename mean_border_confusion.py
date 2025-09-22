import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

from utils.pma import pma_from_config
from utils.test_datasets import (
    get_borders_betas,
    tgt_output_from_dict,
)


if __name__ == "__main__":
    device = "cuda"
    model, c, logs = pma_from_config(device)

    N = c.N_sites

    beta_count = 5
    samples = 200
    set_size = 25

    file = np.load(
        f"results/pma/predictions/dataset__N_{N}__beta_{beta_count}__set_{set_size}__samples_{samples}__{c.model_id}__{c.train_id}.npz"
    )

    borders, betas = get_borders_betas(file)

    # collect confusion matrices
    conf_mats = []
    for beta in betas:
        true_labels = []
        pred_labels = []

        for border in borders:
            tgt, out = tgt_output_from_dict(file, border, beta)
            out_borders = np.argmax(out, axis=-1)

            true_labels.extend([border % N] * len(out_borders))
            pred_labels.extend(out_borders)

        cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(N))
        conf_mats.append(cm)

    # ---- PLOT ----
    n_betas = len(betas)
    ncols = min(3, n_betas)
    nrows = int(np.ceil(n_betas / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, beta, cm in zip(axes, betas, conf_mats):
        sns.heatmap(cm, ax=ax, cmap="Blues", cbar=False)
        ax.set_title(f"$\\beta$={beta}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    # remove unused axes
    for ax in axes[len(betas):]:
        ax.remove()

    fig.suptitle(
        f"Confusion matrices\n"
        f"ds:{N, beta_count, samples, set_size}, shown: {logs['samples_shown']} $\\beta$ {c.train_beta_temps}",
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"results/pma/confusion_matrices__{c.model_id}__{c.train_id}")
    plt.close()

