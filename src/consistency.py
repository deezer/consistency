import torch
import numpy as np
import math
import scipy
from scipy import special, integrate
from src.utils import get_device
import matplotlib.pyplot as plt


class Consistency:
    # A class to compute consistency scores of a set of embeddings {X_1,...,X_N}
    # Both theoretical values using i.i.d hypothesis as well as actual values can be computed

    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings
        self.dev = get_device()

    def avg_precision(self, k):
        # Compute empirical precision for random subsets of size k
        subsets_idx = self.sample_subsets(k)
        avg_emb = self.avg_embeddings(subsets_idx)
        nearest_neighbours = self.get_top_k(avg_emb, k)
        precision = self.precision(subsets_idx, nearest_neighbours)
        return precision

    def sample_subsets(self, p_size, n_users=10 ** 3):
        # Randomly sample N subsets of the same size
        N = self.embeddings.N
        if N >= 10 ** 5:
            # for very large values of N we sample with replacement for efficiency, as the chance of actually sampling
            # the same vector multiple times is very low
            subsets = torch.LongTensor(np.random.choice(N, size=(n_users, p_size), replace=True)).to(self.dev)
        else:
            subsets = torch.zeros((n_users, p_size)).long().to(self.dev)
            for i in range(n_users):
                subsets[i] = torch.LongTensor(np.random.choice(N, size=p_size, replace=False)).to(self.dev)
        return subsets

    def avg_embeddings(self, subsets_idx):
        # Compute average embedding of a subset
        return self.embeddings.vectors[subsets_idx].mean(dim=1)

    def get_top_k(self, agg_embedding, p_size):
        # find k nearest neighbours for each average vector
        scores = agg_embedding.matmul(self.embeddings.vectors.T)
        top_clusters = scores.topk(k=p_size, dim=1)[1]
        return top_clusters

    def precision(self, subsets, top_clusters):

        n_users = subsets.shape[0]
        p_size = subsets.shape[1]
        n_common = 0
        for i in range(n_users):
            n_common += len(np.intersect1d(subsets[i].cpu(), top_clusters[i].cpu()))
        return n_common / (p_size * n_users)

    def mu_diff(self, k):
        # E[<MU, U_in> - <MU, U_out>]
        return self.embeddings.d * self.embeddings.sigma ** 2 / k

    def sigma_diff(self, k):
        # STD[<MU, U_in> - <MU, U_out>]

        d = self.embeddings.d
        mu = self.embeddings.mu
        sigma = self.embeddings.sigma
        gamma = self.embeddings.skew
        kappa = self.embeddings.kurtosis
        return math.sqrt(d * ((2 * (
                k - 1) + kappa) * sigma ** 4 + 2 * k ** 2 * sigma ** 2 * mu ** 2 + 2 * k * gamma * mu * sigma ** 3)) / k

    def mu_mean(self, k):
        # E[MU]
        return self.embeddings.mu

    def sigma_mean(self, k):
        # STD[MU]
        return self.embeddings.sigma / math.sqrt(k)

    def mu_in(self, k):
        # E[<MU, U_in>]
        d = self.embeddings.d
        mu = self.embeddings.mu
        sigma = self.embeddings.sigma
        return d * mu ** 2 + d * sigma ** 2 / k

    def sigma_in(self, k):
        # STD[<MU, U_in>]

        V = self.embeddings
        d = V.d
        mu = V.mu
        sigma = V.sigma
        gamma = V.skew
        kappa = V.kurtosis
        return math.sqrt(d * mu ** 2 * sigma ** 2 * k * (k + 3) + d * sigma ** 4 * (
                kappa + k - 2) + 4 * d * mu * gamma * sigma ** 3) / k

    def mu_out(self, k):
        # E[<MU, U_out>]

        d = self.embeddings.d
        mu = self.embeddings.mu
        return d * mu ** 2

    def sigma_out(self, k):
        # STD[<MU, U_out>]

        sigma = self.embeddings.sigma
        mu = self.embeddings.mu
        d = self.embeddings.d
        return math.sqrt(d * sigma ** 2 * ((k + 1) * mu ** 2 + sigma ** 2) / k)

    def get_diff_scores(self, k, n_subsets=1000):
        # assess f_in - f_out with simulations

        subsets = self.sample_subsets(k, n_subsets)
        avg_embedding = self.avg_embeddings(subsets)
        in_choice = np.array([np.random.choice(subsets[u], size=k, replace=True) for u in range(n_subsets)])
        in_scores = torch.bmm(self.embeddings.vectors[in_choice], avg_embedding.unsqueeze(2))
        out_choice = np.array(
            [np.random.choice(np.setdiff1d(range(n_subsets), subsets[u]), size=k, replace=True) for u in
             range(n_subsets)])
        out_scores = torch.bmm(self.embeddings.vectors[out_choice], avg_embedding.unsqueeze(2))

        return in_scores - out_scores

    def get_in_scores(self, k, n_subsets=1000):
        # assess f_in with simulations
        subsets = self.sample_subsets(k, n_subsets)
        avg_embedding = self.avg_embeddings(subsets)
        in_scores = torch.bmm(self.embeddings.vectors[subsets], avg_embedding.unsqueeze(2))
        return in_scores

    def get_out_scores(self, k, n_subsets=1000):
        # assess f_out with simulations
        subsets = self.sample_subsets(k, n_subsets)
        other_choice = np.array(
            [np.random.choice(np.setdiff1d(range(n_subsets), subsets[u]), k) for u in range(n_subsets)])
        avg_embedding = self.avg_embeddings(subsets)
        out_scores = torch.bmm(self.embeddings.vectors[other_choice], avg_embedding.unsqueeze(2))
        return out_scores

    def get_in_dist(self, k):
        # required to plot f_in
        x = self.embeddings.x
        return self.f_in(x, k)

    def get_out_dist(self, k):
        # required to plot f_out
        x = self.embeddings.x
        return self.f_out(x, k)

    def f_in(self, x, k):
        # p.d.f of <MU, U_in>
        return scipy.stats.norm.pdf(x, self.mu_in(k), self.sigma_in(k))

    def f_in_alt(self, x, k):
        # p.d.f of <MU, U_in>
        x = (x - self.mu_in(k)) / self.sigma_in(k)
        return scipy.stats.norm.pdf(x, 0, 1)

    def f_out(self, x, k):
        # p.d.f of <MU, U_out>
        return scipy.stats.norm.pdf(x, self.mu_out(k), self.sigma_out(k))

    def F_in(self, x, k):
        # c.d.f of <MU, U_in>
        return scipy.stats.norm.cdf(x, self.mu_in(k), self.sigma_in(k))

    def F_out(self, x, k):
        # c.d.f of <MU, U_out>

        return scipy.stats.norm.cdf(x, self.mu_out(k), self.sigma_out(k))

    def f_i_in(self, x, i, k):
        # p.d.f of ith highest value of <MU, U_in>

        N = k
        F = self.F_in(x, k)
        f = self.f_in(x, k)
        dist = scipy.special.comb(N, i) * i * F ** (N - i) * (1 - F) ** (i - 1) * f
        return dist

    def f_j_out(self, x, j, k):
        # p.d.f of jth highest value of <MU, U_in>

        N = self.embeddings.N - k
        F = self.F_out(x, k)
        f = self.f_out(x, k)
        dist = scipy.special.comb(N, j) * j * F ** (N - j) * (1 - F) ** (j - 1) * f
        return dist

    def F_i_in(self, x, i, k):
        # c.d.f of ith order statistic of <MU, U_in>

        N = k
        F = self.F_in(x, k)
        els = np.array([special.comb(N, j) * F ** j * (1 - F) ** (N - j) for j in range(N - i + 1, N + 1)])
        return np.sum(els, axis=0)

    def F_j_out(self, x, j, k):
        # c.d.f of jth order statistic of <MU, U_in>

        N = self.embeddings.N - k
        F = self.F_out(x, k)
        els = np.array([special.comb(N, i) * F ** i * (1 - F) ** (N - i) for i in range(N - j + 1, N + 1)])
        return np.sum(els, axis=0)

    def top_i_in_dist(self, i, k):
        # distribution of f_i_in
        x = self.embeddings.x_tight
        return self.f_i_in(x, i, k)

    def top_j_out_dist(self, j, k):
        # distribution of f_j_out
        x = self.embeddings.x_tight
        return self.f_j_out(x, j, k)

    def simu_jth_out(self, j, k):
        # empirical distribution of f_j_out
        subsets = self.sample_subsets(k)
        avg_emb = self.avg_embeddings(subsets)
        scores = avg_emb.matmul(self.embeddings.vectors.T)
        scores.scatter_(dim=1, value=-10 ** 3, index=subsets)
        values = scores.topk(k=j, dim=1, largest=True)[0][:, -1]
        return values

    def simu_jth_in(self, j, k):
        # empirical distribution of f_i_int

        subsets = self.sample_subsets(k)
        avg_emb = self.avg_embeddings(subsets)
        scores = avg_emb.matmul(self.embeddings.vectors.T)
        scores_subsets = scores.gather(dim=1, index=subsets)
        values = scores_subsets.topk(k=j, dim=1, largest=True)[0][:, -1]
        return values

    def compute_expected_precision(self, k, x_min, x_max):
        # Compute consistency using formula with F_i_in and f_j_out
        proba = np.sum(
            [integrate.quad(lambda x: (1 - self.F_i_in(x, i, k)) * self.f_j_out(x, k - i + 1, k), x_min, x_max) for i in
             range(1, k + 1)]) / k
        return proba

    def compute_expected_precision_alt(self, k, x_min, x_max):
        # Compute consitsency using formula with F_j_out and f_i_in

        proba = integrate.quad(
            lambda x: np.array([self.f_i_in(x, i, k) * self.F_j_out(x, k - i + 1, k) for i in range(1, k + 1)]).sum(
                axis=0), x_min, x_max)[0] / k
        return proba

    def plot_diff(self, k):
        # plot p.d.d of <MU, U_in> - <MU, U_out>
        plt.hist(self.get_diff_scores(k).squeeze().reshape(-1), density=True)
        plt.plot(self.embeddings.x - 20,
                 scipy.stats.norm.pdf(self.embeddings.x - 20, self.mu_diff(k), self.sigma_diff(k)))
