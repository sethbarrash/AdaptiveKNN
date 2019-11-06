import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from SparseGaussianProcess import SparseGaussianProcess
from SparseGaussianProcessDataSelector import SparseGaussianProcessDataSelector

class SAKNeighborsBase:

    def __init__(self, h, r, sigma):
        self.h = h
        self.r = r
        self.sigma = sigma

    def fit(self, X, y):
        self._fit_X = X
        self._fit_y = y
        self.sogp = SparseGaussianProcess(X[0], y[0], self.h, self.sigma)
        self.sogp.update(X[1], y[1])
        self.selector = SparseGaussianProcessDataSelector(self.sogp)

    def _get_neighbor_order(self, idx):
        x0 = self._fit_X[idx]
        if len(self._fit_X.shape) == 1 : xdist = np.abs(self._fit_X - x0)
        else                 : xdist = np.abs( np.sum(self._fit_X - x0, 1) )
        neighbor_order = np.argsort(xdist)[1:]

        return neighbor_order

    def _get_neighborhood(self, idx):
        nidx = self._get_neighbor_order(idx)

        Xsorted = self._fit_X[nidx]
        ysorted = self._fit_y[nidx]

        return Xsorted, ysorted




class SAKNeighborsRegressor(SAKNeighborsBase):

    def __init__(self, h, r, sigma):
        super().__init__(h, r, sigma)

    def _calculate_neighbor_errors(self, ysorted, y0, task):
        if task == 'r':
            ysum = np.cumsum(ysorted)
            ybar = ysum / np.arange(1, len(ysorted) + 1)
            e = ybar - y0
            return e
        else:
            return ysorted == y0

    def fitk(self, idx, task):
        y0 = self._fit_y[idx]
        Xsorted, ysorted = self._get_neighborhood(idx)
        e = self._calculate_neighbor_errors(ysorted, y0, task)
        if task == 'r':
            k = np.argmin(np.abs(e)) + 1
        else: 
            idx_correct = np.where(e)[0]
            k = np.random.choice(idx_correct) + 1
        return k

    def learn_next_instance(self):
        ## Actively select the next training instance whose k-value to label
        idx = self.selector.select(self._fit_X, self.r)
        ## Update the sparse Gaussian process with the newly labeled datum
        ki = self.fitk(idx, 'r')
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



class SAKNeighborsClassifier(SAKNeighborsBase):

    def __init__(self, h, r, sigma):
        super().__init__(h, r, sigma)

    def _calculate_neighbor_errors(self, ysorted, y0, task):
        if task == 'r':
            ysum = np.cumsum(ysorted)
            ybar = ysum / np.arange(1, len(ysorted) + 1)
            e = ybar - y0
            return e
        else:
            return ysorted == y0

    def fitk(self, idx, task):
        y0 = self._fit_y[idx]
        Xsorted, ysorted = self._get_neighborhood(idx)
        e = self._calculate_neighbor_errors(ysorted, y0, task)
        if task == 'r':
            k = np.argmin(np.abs(e)) + 1
        else: 
            idx_correct = np.where(e)[0]
            k = np.random.choice(idx_correct) + 1
        return k

    def learn_next_instance(self):
        ## Actively select the next training instance whose k-value to label
        idx = self.selector.select(self._fit_X, self.r)
        ## Update the sparse Gaussian process with the newly labeled datum
        ki = self.fitk(idx, 'c')
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
