import numpy as np
import SparseGP as sgp

def get_neighbor_order(X, y, idx):
    x0 = X[idx]
    if len(X.shape) == 1 : xdist = np.abs(X - x0)
    else                 : xdist = np.abs( np.sum(X - x0, 1) )
    neighbor_order = np.argsort(xdist)[1:]

    return neighbor_order


def get_neighborhood(X, y, idx):
    nidx = get_neighbor_order(X, y, idx)

    Xsorted = X[nidx]
    ysorted = y[nidx]

    return Xsorted, ysorted


def calculate_neighbor_errors(ysorted, y0):
    ysum = np.cumsum(ysorted)
    ybar = ysum / np.arange(1, len(ysorted) + 1)
    e = ybar - y0
    return e


def fitk_spincom_incorrect(X, y, idx):
    y0 = y[idx]
    Xsorted, ysorted = get_neighborhood(X, y, idx)
    e = calculate_neighbor_errors(ysorted, y0)
    k = np.argmin(e) + 1

    return k


def fitk_spincom(X, y, idx):
    y0 = y[idx]
    Xsorted, ysorted = get_neighborhood(X, y, idx)
    e = calculate_neighbor_errors(ysorted, y0)
    k = np.argmin(np.abs(e)) + 1

    return k


def initialize_k_generalization(X, y, h, sigma):
    k0 = fitk_spincom(X, y, 0)
    k1 = fitk_spincom(X, y, 1)

    g = sgp.SparseGP(X[0], k0, h, sigma)
    g.update(X[1], k1)
    ridx = np.arange(2, len(X))

    return g, ridx


def add_most_novel_training_datum(g, X, y, ridx):
    gamma = g.novelty(X, ridx)
    idxnew = np.argmin(gamma)
    knew = fitk_spincom(X, y, idxnew)
    g.update(X[idxnew], knew)
    idx_ = np.searchsorted(ridx, idxnew)
    ridx = np.delete(ridx, idx_)


def generalizek_spincom(X, y, m, h, sigma):
    g, ridx = initialize_k_generalization(X, y, h, sigma)
    for _ in range(2, m):
        add_most_novel_training_datum(g, X, y, ridx)

    return g


def choose_k_from_random_subset(Krand, ysorted, y0):
    error_min = np.Inf
    for k in Krand:
        error = np.mean(ysorted[:k]) - y0
        if error < error_min:
            error_min = error
            kbest = k

    return kbest


def fitk_adaknn(X, y, idx, alpha, kmax):
    y0 = y[idx]
    Xsorted, ysorted = get_neighborhood(X, y, idx)
    Krand = np.random.choice(np.arange(kmax), alpha)
    kbest = choose_k_from_random_subset(Krand, ysorted, y0)
    return kbest