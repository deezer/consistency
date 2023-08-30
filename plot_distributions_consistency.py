import numpy as np
import torch
from src.utils import get_device
from src.embeddings import Embeddings
from src.consistency import Consistency
import argparse
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

d = 128
n_small = 10 ** 3
n_large = 10 ** 6
span = range(2, 51)  # possible size of subsets to compute average


def prepare_consistency(vectors):
    dev = get_device()
    vectors = torch.Tensor(vectors)
    embeddings = Embeddings(vectors)
    consistency = Consistency(embeddings)
    consistency.embeddings.vectors = consistency.embeddings.vectors.to(dev)
    return consistency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distribution", type=str, required=True, choices=['normal', 'rademacher', 'uniform'],
                        help="Type of distribution to be studied")

    args = parser.parse_args()

    if args.distribution == "normal":
        vectors_small = np.random.normal(loc=0, scale=1, size=(n_small, d))
        vectors_large = np.random.normal(loc=0, scale=1, size=(n_large, d))
        params = {"name": "normal", "title": r"$X_{i,j} \sim \mathcal{N}(0,1)$"}
    elif args.distribution == "rademacher":
        vectors_small = 2 * np.random.randint(0, 2, size=(n_small, d))-1
        vectors_large = 2 * np.random.randint(0, 2, size=(n_large, d))-1
        params = {"name": "rademacher", "title": r"$X_{i,j} \sim Rademacher()$"}
    elif args.distribution == "uniform":
        vectors_small = np.random.uniform(-1, 1, size=(n_small,d))
        vectors_large = np.random.uniform(-1, 1, size=(n_large,d))
        params = {"name": "uniform", "title": r"$X_{i,j} \sim Uniform(-1,1)$"}

    print(rf"{params['title']}")

    consistency_small = prepare_consistency(vectors_small)
    consistency_large = prepare_consistency(vectors_large)

    print(f"Simulate consistency for N = {str(n_small)}")
    values_simulated_small = [consistency_small.avg_precision(k) for k in tqdm.tqdm(span)]
    print(f"Simulate consistency for N = {str(n_large)}")
    values_simulated_large = [consistency_large.avg_precision(k) for k in tqdm.tqdm(span)]
    print(f"Compute consistency for N = {str(n_small)}")
    values_computed_small = np.array(
        [consistency_small.compute_expected_precision_alt(k, x_min=0, x_max=100) for k in tqdm.tqdm(span)])
    print(f"Compute consistency for N = {str(n_large)}")
    values_computed_large = np.array(
        [consistency_large.compute_expected_precision_alt(k, x_min=0, x_max=100) for k in tqdm.tqdm(span)])

    sns.set()
    sns.lineplot(x=span, y=values_computed_small, label="N = 1 000 (computed)", color="darkblue")
    sns.lineplot(x=span, y=values_simulated_small, label="N = 1 000 (simulated)", color="darkorange")
    sns.lineplot(x=span, y=values_computed_large, label="N = 1 000 000 (computed)", color="deepskyblue")
    sns.lineplot(x=span, y=values_simulated_large, label="N = 1 000 000 (simulated)", color="goldenrod")

    LABEL_SIZE = 12
    AXES_SIZE = 18
    TITLE_SIZE = 18
    plt.legend(fontsize=LABEL_SIZE)
    plt.xlabel("k", fontsize=AXES_SIZE)
    plt.ylabel(r"$Consistency_{k}(\mathcal{X})$", fontsize=AXES_SIZE)
    plt.title(params['title'], fontsize=TITLE_SIZE)
    plt.savefig(f"plots/consistency_{params['name']}.png")