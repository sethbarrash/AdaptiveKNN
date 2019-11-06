import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from SparseGaussianProcess import SparseGaussianProcess
from SparseGaussianProcessDataSelector import SparseGaussianProcessDataSelector

class SAKNeighborsRegressor:

    def __init__(self, h, m, r, sigma):
        self.h = h
        self.m = m
        self.r = r
        self.sigma = sigma

    def fit(self, X, y):
        self._fit_X = X
        self._fit_y = y
        self.sogp = SparseGaussianProcess(X[0], y[0], self.h, self.sigma)
        self.sogp.update(X[1], y[1])
        self.selector = SparseGaussianProcessDataSelector(self.sogp)

    def learn_next_instance(self):
        ## Actively select the next training instance whose k-value to label
        idx = self.selector.select(self._fit_X, self.r)
        ## Update the sparse Gaussian process with the newly labeled datum
        self.sogp.update(self._fit_X[idx], self._fit_y[idx])

    def predict(self, X):
        yhat = np.zeros((len(X),))
        for i in range(len(X)):
            xi = np.atleast_2d(X[i])
            ku = self.sogp.predict(xi)
            knr = KNeighborsRegressor(n_neighbors = ku)
            knr.fit(self._fit_X, self._fit_y)
            yhat[i] = knr.predict(xi)[0]
        return yhat

