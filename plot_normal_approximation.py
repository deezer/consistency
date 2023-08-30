import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import tqdm
from src.embeddings import Embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distribution", type=str, required=True, choices=['normal', 'uniform', 'beta'],
                        help="Type of distribution to be studied")

    args = parser.parse_args()

    if args.distribution == "normal":
        params = {"name": "normal", "x_range": np.linspace(-20, 80, 5000),
                  "title": r"Normal approximation for $\mathcal{N}(0.5,1)$", "y_lim": 0.4}

    elif args.distribution == "uniform":
        params = {"name": "uniform", "x_range": np.linspace(-1, 40, 5000),
                  "title": r"Normal approximation for Uniform(0,1)", "y_lim": 1.3}
    elif args.distribution == "beta":
        params = {"name": "beta", "x_range": np.linspace(-1, 40, 5000),
                  "title": r"Normal approximation for Beta(2,2)", "y_lim": 1.8}
    n_points = 1000
    d_range = [2, 10, 32, 64, 128]
    LABEL_SIZE = 12
    AXES_SIZE = 18
    TITLE_SIZE = 18
    sns.set()
    print(f"{params['title']} with d = {','.join([str(d) for d in d_range])}")
    for d in tqdm.tqdm(d_range):
        if args.distribution == "normal":
            vectors = np.random.normal(loc=0.5, scale=1, size=(n_points, d))
        elif args.distribution == "uniform":
            vectors = np.random.uniform(low=0, high=1, size=(n_points, d))
        elif args.distribution == "beta":
            vectors = np.random.beta(a=2, b=2, size=(n_points, d))
        vectors = torch.Tensor(vectors)
        vectors = Embeddings(vectors, sims=True)
        other_dist = vectors.get_other_dist(params["x_range"])
        sns.lineplot(x=params["x_range"], y=other_dist, label="d=%d" % d)
        other_sampled = vectors.sample_other()
        if d == 2:
            sns.histplot(other_sampled, bins=20, stat="density")
        else:
            sns.histplot(other_sampled, bins=40, stat="density")
        plt.legend(loc="upper right", fontsize=LABEL_SIZE)
        plt.ylabel("density", fontsize=AXES_SIZE)
        plt.xlabel(r"$s(X, Y)$", fontsize=AXES_SIZE)
        plt.title(params["title"], fontsize=TITLE_SIZE)
        plt.ylim(top=params["y_lim"])
        plt.savefig(f"plots/s(X,Y)_{params['name']}.png")
