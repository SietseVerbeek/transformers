import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import entropy

from models.transformer import Transformer
from utils.fs import info_dir, results_dir
from utils.masking import generate_square_subsequent_mask


def generate_naked_sequences(
    model: Transformer, k: int, max_length, SOS_token=2, EOS_token=3, device="cuda"
):
    model.eval()
    tgt = torch.full((k, 1), SOS_token, dtype=torch.int64, device=device)
    src = torch.empty(k, 2, dtype=torch.int64, device=device)
    src[:] = torch.tensor([SOS_token, EOS_token])

    for _ in range(max_length):
        tgt_mask = (generate_square_subsequent_mask(tgt.size(1))).to(device)

        logits = model(src, tgt, tgt_mask=tgt_mask).detach()
        probabilities = torch.softmax(logits[:, -1, :], -1).cumsum(-1)
        throw = torch.rand(k, 1).to(device)
        idx = torch.searchsorted(probabilities, throw)

        # Concatenate previous input with predicted best word
        tgt = torch.cat((tgt, idx), dim=-1)

    return tgt


def generate_naked_sequences_dynamic(
    model: Transformer, k: int, max_length, SOS_token=2, EOS_token=3, device="cuda"
):
    tgt = torch.full((k, 1), SOS_token, dtype=torch.int64, device=device)
    src = torch.empty(k, 2, dtype=torch.int64, device=device)
    src[:] = torch.tensor([SOS_token, EOS_token])

    EOS_tensor = torch.full((k, 1), EOS_token, dtype=torch.int64, device=device)

    for _ in range(max_length):
        tgt_mask = (generate_square_subsequent_mask(tgt.size(1))).to(device)

        logits = model(src, tgt, tgt_mask=tgt_mask).detach()
        probabilities = torch.softmax(logits[:, -1, :], -1).cumsum(-1)
        throw = torch.rand(k, 1).to(device)
        idx = torch.searchsorted(probabilities, throw)

        # Concatenate previous input with predicted best word
        tgt = torch.cat((tgt, idx), dim=-1)
        src = torch.cat((tgt, EOS_tensor.view(k, 1)), dim=-1)

    return tgt


def plot_prob_density_overlap(
    filename: str, tgt_distribution: npt.NDArray, generated_distribution: npt.NDArray
):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(generated_distribution, label="generated")
    ax1.plot(tgt_distribution, label="source")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(generated_distribution)
    ax2.set_title("generated")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(tgt_distribution)
    ax3.set_title("source")

    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    from config import FILENAME, SEQUENCES_PER_TRIAL, TRIALS, USE_CUDA

    info_filename = info_dir(f"{FILENAME}.info")
    weights_filename = info_dir(f"{FILENAME}.pth")
    tgts_filename = info_dir(f"{FILENAME}.npz")

    static_fig_filename = results_dir(f"{FILENAME}.jpg")
    dynamic_fig_filename = results_dir(f"{FILENAME}_dynamic.jpg")
    log_file = results_dir(f"{FILENAME}.log")

    total_sequences = SEQUENCES_PER_TRIAL * TRIALS

    tgt_file = np.load(tgts_filename)

    tgts = tgt_file["unique_tgts"]
    tgt_frac = tgt_file["tgt_frac"]
    generate_length = tgts.shape[-1] - 1

    print(f"generating {total_sequences} sequences")
    logs = [
        "\n",
        "=" * 80 + "\n",
        f"Generating {total_sequences} sequences of length {generate_length}\n",
    ]

    logs.extend(
        (
            "target distribution:\n",
            f"{tgt_frac}\n",
            f"normalization: {np.sum(tgt_frac)} (should equal 1)\n",
        )
    )

    with open(info_filename, "r") as file:
        lines = file.readlines()[1:]
        d_model = int(lines[0].split()[-1])
        num_heads = int(lines[1].split()[-1])
        num_encoder_layers = int(lines[2].split()[-1])
        num_decoder_layers = int(lines[3].split()[-1])

    device = "cuda" if USE_CUDA else "cpu"
    model = Transformer(
        num_tokens=5,
        d_model=d_model,
        padding_idx=4,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load(weights_filename, weights_only=True))

    dynamic = np.empty((total_sequences, generate_length + 1))
    static = np.empty((total_sequences, generate_length + 1))

    for i in range(TRIALS):
        print(i * SEQUENCES_PER_TRIAL, end="\r")
        dynamic[i * SEQUENCES_PER_TRIAL : (i + 1) * SEQUENCES_PER_TRIAL] = (
            generate_naked_sequences_dynamic(
                model, SEQUENCES_PER_TRIAL, generate_length, device=device
            )
            .to("cpu")
            .numpy()
        )
        static[i * SEQUENCES_PER_TRIAL : (i + 1) * SEQUENCES_PER_TRIAL] = (
            generate_naked_sequences(
                model, SEQUENCES_PER_TRIAL, generate_length, device=device
            )
            .to("cpu")
            .numpy()
        )

    mask_dynamic = np.any(np.all(dynamic[..., None, :] == tgts, axis=-1), axis=-1)
    mask_static = np.any(np.all(static[..., None, :] == tgts, axis=-1), axis=-1)

    counts_static = np.sum(np.all(static[..., None, :] == tgts, axis=-1), axis=0)
    counts_dynamic = np.sum(np.all(dynamic[..., None, :] == tgts, axis=-1), axis=0)

    fracs_static = counts_static / np.sum(mask_static)
    fracs_dynamic = counts_dynamic / np.sum(mask_dynamic)

    plot_prob_density_overlap(static_fig_filename, tgt_frac, fracs_static)
    plot_prob_density_overlap(dynamic_fig_filename, tgt_frac, fracs_dynamic)

    logs.extend(
        (
            "\n",
            "dynamic:\n",
            f"{fracs_dynamic}\n",
            f"KL-divergence {entropy(tgt_frac, fracs_dynamic)}\n",
            f"error frac: {np.sum(~mask_dynamic) / (total_sequences)}\n",
        )
    )

    logs.extend(
        (
            "\n",
            "static:\n",
            f"{fracs_static}\n",
            f"KL-divergence {entropy(tgt_frac, fracs_static)}\n",
            f"error frac: {np.sum(~mask_dynamic) / (total_sequences)}\n",
        )
    )

    rng = np.random.default_rng()
    random_data = rng.integers(2, size=static.shape)
    random_data[:, 0] = 2
    random_data[:, -1] = 3
    counts_random = np.sum(np.all(random_data[..., None, :] == tgts, axis=-1), axis=0)
    frac_random = counts_random / random_data.shape[0]

    logs.extend(
        (
            "\n",
            "random data:\n",
            f"{frac_random}\n",
            f"KL-divergence {entropy(tgt_frac, frac_random)}\n",
        )
    )

    with open(log_file, "a+") as file:
        file.writelines(logs)

    print(f"figure saved to {static_fig_filename}")
    print(f"logs written to {log_file}")
