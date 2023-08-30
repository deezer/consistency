import scipy
import math
import numpy as np
import matplotlib.pyplot as plt


class Embeddings:
    # A class to study a set of embeddings {X_1,...,X_N}
    def __init__(self, vectors, sims=False) -> None:
        self.vectors = vectors
        self.N = self.vectors.shape[0]  # number of embeddings
        self.d = self.vectors.shape[1]  # dimension
        self.mu = self.vectors.mean()  # empirical mean
        if self.mu < 0.05:
            self.mu = 0
        self.sigma = self.vectors.std()  # empirical standard deviation
        self.skew = scipy.stats.skew(self.vectors).mean()  # empirical skewness
        self.kurtosis = scipy.stats.kurtosis(self.vectors, fisher=False).mean()  # empirical kurtosis
        self.mu_other = self.cmp_mu_other()
        self.sigma_other = self.cmp_sigma_other()
        self.mu_self = self.cmp_mu_self()
        self.sigma_self = self.cmp_sigma_self()
        self.cov_self_other = self.cmp_cov_self_other()
        self.cov_other = self.cmp_cov_other()
        self.cov_self_other = self.cmp_cov_self_other()
        self.cov_other = self.cmp_cov_other()
        if sims:
            # WARNING : this will compute X*X.T so only do this if N is small
            self.sims = self.cmp_sims()
        self.x = self.cmp_x()
        self.x_tight = self.cmp_x_tight()

    def cmp_mu_other(self):
        # Compute E[<X_i,X_j>] (i!=j)
        return self.d * self.mu ** 2

    def cmp_sigma_other(self):
        # Compute STD[<X_i,X_j>] (i!=j)
        d = self.d
        sigma = self.sigma
        mu = self.mu
        return math.sqrt(d * (sigma ** 2 * (sigma ** 2 + 2 * mu ** 2)))

    def cmp_mu_self(self):
        # Compute E[<X_i,X_i>]
        return self.d * (self.sigma ** 2 + self.mu ** 2)

    def cmp_sigma_self(self):
        # Compute STD[<X_i,X_i>]
        d = self.d
        sigma = self.sigma
        mu = self.mu
        gamma = self.skew
        kappa = self.kurtosis
        return math.sqrt(
            d * (4 * mu ** 2 * sigma ** 2 + 4 * mu * gamma * sigma ** 3 + (kappa - 1) * sigma ** 4))

    def cmp_cov_self_other(self):
        # compute Cov(<X_i,X_i>,<X_i,X_j>) (i!=j)
        return 2 * self.d * self.sigma ** 2 * self.mu ** 2

    def cmp_cov_other(self):
        # compute Cov(<X_i,X_j>,<X_i,X_k>) (i!=j!=k)
        return self.d * self.sigma ** 2 * self.mu ** 2

    def cmp_sims(self):
        # compute inner product between every pair fo vectors
        return self.vectors.matmul(self.vectors.T)

    def cmp_x(self):
        # compute scale of x-axis for plots
        return np.linspace(self.mu_other - 5 * self.sigma_other, self.mu_other + 20 * self.sigma_other, 5000)

    def cmp_x_tight(self):
        # compute scale of x-axis for plots (tighter than cmp_x)
        return np.linspace(self.mu_other - 2 * self.sigma_other, self.mu_other + 5 * self.sigma_other, 5000)

    def sample_other(self):
        # sample from <X_i,X_j> (i!=j)
        triu = self.sims.triu(diagonal=1)
        triu = triu[triu != 0]
        other_samples = np.random.choice(triu, self.N)
        return other_samples

    def get_self_dist(self, x):
        # get pdf of the normal approximation of <X_i,X_i>
        self_dist = scipy.stats.norm.pdf(x, self.mu_self, self.sigma_self)
        return self_dist

    def get_other_dist(self, x):
        # get pdf of the normal approximation of <X_i,X_j> (i!=j)

        other_dist = scipy.stats.norm.pdf(x, self.mu_other, self.sigma_other)
        return other_dist

    def plot_other_vs_self(self):
        # Plot normal approximation for both <X_i,X_i> and <X_i,X_j> as well as histograms of empirical distributions
        other_dist = self.get_other_dist(self.x)
        self_dist = self.get_self_dist(self.x)
        other_samples = self.sample_other()
        plt.hist(self.sims.diag(), density=True, bins=20, label="self_simu")
        plt.plot(self.x, self_dist, label="self_formula")
        plt.hist(other_samples, density=True, bins=20, label="other_simu")
        plt.plot(self.x, other_dist, label="other_formula")
        plt.legend()
        plt.title("distribution of dot product in 128 dimensions for normal distribution")
        plt.show()
