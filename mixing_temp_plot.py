import argparse
import matplotlib.pyplot as plt
import numpy as np

from pma import Config


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    c = Config(args.config)

    filename = f"results/pma/voi_{c.model_id}__id_{c.train_id}"
    npzfile = np.load(filename + ".npz")

    beta = npzfile["beta"]
    mu = npzfile["mu"]
    voi = npzfile["voi"]

    x, y = np.meshgrid(mu, beta)

    plt.pcolormesh(x, y, voi)
    plt.xlabel("$\\mu$")
    plt.ylabel("$\\beta$")
    plt.colorbar()
    plt.savefig(filename)
