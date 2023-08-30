import torch
import numpy as np
import pandas as pd
from src.embeddings import Embeddings
from src.consistency import Consistency
from src.utils import get_device
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_embeddings", type=str, required=True, choices=['svd', 'als'],
                        help="Type of embeddings to be studied")

    args = parser.parse_args()

    dev = get_device()
    data_df = pd.read_parquet("data/song_embeddings.parquet")
    if args.source_embeddings == "svd":
        vectors = torch.Tensor(np.stack(data_df.features_svd.to_numpy()))
        params = {"name": "TT-SVD"}
    if args.source_embeddings == "als":
        vectors = torch.Tensor(np.stack(data_df.features_mf.to_numpy()))
        params = {"name": "UT-ALS"}

    # center embeddings and compute consistency
    vectors_centered = vectors - vectors.mean(dim=0)
    embeddings = Embeddings(vectors_centered)
    embeddings.vectors = embeddings.vectors.to(dev)
    consistency = Consistency(embeddings)
    span = range(2, 51)
    print(f"Computing consistency for {params['name']}")
    values_empirical = [consistency.avg_precision(k) for k in tqdm.tqdm(span)]

    # Create embeddings sampled from normal distribution
    N_svd = vectors.shape[0]
    d_svd = vectors.shape[1]
    normal_vectors = torch.Tensor(np.random.normal(loc=0, scale=1, size=(N_svd, d_svd)))
    normal_embeddings = Embeddings(normal_vectors)
    normal_embeddings.vectors = normal_embeddings.vectors.to(dev)
    normal_consistency = Consistency(normal_embeddings)
    # compute theoretical consistency for normal embeddings
    print(f"Computing theoretical consistency for normal distribution")

    valuesTheoretical = [normal_consistency.compute_expected_precision_alt(k, 0, 150) for k in tqdm.tqdm(span)]
    if args.source_embeddings == "als":
        valuesTheoretical[0] = normal_consistency.compute_expected_precision(2, 0, 150)

    # Plot results
    sns.lineplot(x=span, y=valuesTheoretical, label=r"$\mathcal{N}(0,1)$")
    sns.lineplot(x=span, y=values_empirical, label=params["name"], color="coral")

    LABEL_SIZE = 12
    AXES_SIZE = 18
    TITLE_SIZE = 18
    plt.legend(fontsize=LABEL_SIZE)
    plt.xlabel("k", fontsize=AXES_SIZE)
    plt.ylabel(r"$Consistency_{k}(\mathcal{X})$", fontsize=AXES_SIZE)
    plt.title(params["name"], fontsize=TITLE_SIZE)
    plt.savefig(f"plots/consistency_{params['name']}.png")