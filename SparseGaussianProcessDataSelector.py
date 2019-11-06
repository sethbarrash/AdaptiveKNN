from adaptiveknn import *

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