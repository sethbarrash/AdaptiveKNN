import numpy as np

def candidate_subset(X, ridx, r):
    if r < len(ridx): 
        idx_eval_novelty = np.random.choice(len(ridx), r, False)
        idx_eval_novelty = np.sort(idx_eval_novelty)
        ridx = ridx[idx_eval_novelty]
    Xr = np.atleast_2d(X[ridx])
    return Xr

def find_most_novel_training_datum(g, X, ridx, r = 100):
    Xr    = candidate_subset(X, ridx, r)
    gamma = g.novelty(Xr)
    idx   = np.argmin(gamma)
    return ridx[idx]


def update_inactive_set(ridx, idx):
    idx_ = np.searchsorted(ridx, idx)
    ridx = np.delete(ridx, idx_)
    return ridx


class SparseGaussianProcessDataSelector:

    def __init__(self, sogp):
        self.sogp = sogp

    def fit(self, X):
        self._fit_X = X
        self.ridx = np.arange(2, len(X)) # Assume the SOGP is fed the first two rows of X upon initialization

    def select(self, X, r):
        idx = find_most_novel_training_datum(self.sogp, self._fit_X, self.ridx, r)
        update_inactive_set(self.ridx, idx)
        return idx