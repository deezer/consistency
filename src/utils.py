import torch
import math
import scipy

def get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    return dev


# Covariance utils

def cov_self_other(d, mu_u, sigma_u, gamma_u, mu_v):
    # covariance between <u,u> and <u,v>
    return d * mu_v * (gamma_u * sigma_u ** 3 + 2 * mu_u * sigma_u ** 2)


def cov_other_other(d, sigma_u, mu_v, mu_w):
    # covariance between <u,v> and <u,w>
    return d * sigma_u ** 2 * mu_v * mu_w


def sigma_dot_product_other(d, mu_u, sigma_u, mu_v, sigma_v):
    return math.sqrt(d * ((mu_u ** 2 + sigma_u ** 2) * (mu_v ** 2 + sigma_v ** 2) - mu_u ** 2 * mu_v ** 2))


def sigma_dot_product_self(d, mu, sigma, gamma, kappa):
    return math.sqrt(d * (4 * mu ** 2 * sigma ** 2 + 4 * mu * gamma * sigma ** 3 + (kappa - 1) * sigma ** 4))


# proba that s_in > s_out
def compute_proba(d, k, mu, sigma, gamma, kappa):
    return (1 / 2) * (1 - scipy.special.erf(-math.sqrt(d * sigma ** 2 / (
            2 * ((2 * (k - 1) + kappa) * sigma ** 2 + 2 * k * mu * gamma * sigma + 2 * k ** 2 * mu ** 2)))))
