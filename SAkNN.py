import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from SparseGaussianProcess import SparseGaussianProcess
from SparseGaussianProcessDataSelector import SparseGaussianProcessDataSelector

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

def calculate_neighbor_errors(ysorted, y0, task):
    if task == 'r':
        ysum = np.cumsum(ysorted)
        ybar = ysum / np.arange(1, len(ysorted) + 1)
        e = ybar - y0
        return e
    else:
        return ysorted == y0

def fitk_spincom(X, y, idx, task):
    y0 = y[idx]
    Xsorted, ysorted = get_neighborhood(X, y, idx)
    e = calculate_neighbor_errors(ysorted, y0, task)
    if task == 'r':
        k = np.argmin(np.abs(e)) + 1
    else: 
        idx_correct = np.where(e)[0]
        k = np.random.choice(idx_correct) + 1

    return k



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
        ki = fitk_spincom(self._fit_X, self._fit_y, idx, 'r')
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



class SAKNeighborsClassifier:

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
        ki = fitk_spincom(self._fit_X, self._fit_y, idx, 'c')
        self.sogp.update(self._fit_X[idx], self._fit_y[idx])

    def predict(self, X):
        yhat = np.zeros((len(X),))
        for i in range(len(X)):
            xi = np.atleast_2d(X[i])
            ku = self.sogp.predict(xi)
            knr = KNeighborsClassifier(n_neighbors = ku)
            knr.fit(self._fit_X, self._fit_y)
            yhat[i] = knr.predict(xi)[0]
        return yhat
