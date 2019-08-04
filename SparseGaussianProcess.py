import numpy as np
from sklearn.gaussian_process.kernels import RBF

## Three challenges to vectorization:
## 1. Inputs to kernels must be 2d
## 2. KBinv must be initialized properly
## 3. Overflow errors must be anticipated


class SparseGaussianProcess:

    def __init__(self, x0, y0, r, noise):
        self._alpha = np.atleast_2d(y0 / noise)
        self._C     = np.atleast_2d(-1./ noise)
        self._KB    = np.zeros((1, 1))
        self._Q     = np.ones((1, 1))
        self.x      = np.atleast_2d(x0)
        self.noise  = noise
        self.kernel = RBF(r)
        self.t      = 1


    def predict(self, x):
        k = self.kernel(self.x, x)
        return k.T @ self._alpha


    def novelty(self, xr):
        k = self.kernel(self.x, xr)
        gamma = np.ones((len(xr),)) - np.diagonal(k.T @ self._Q @ k)
        return gamma


    def _calculate_b1(self, x, y):
        yhat = self.predict(x).squeeze()
        return (y - yhat) / self.noise


    def _calculate_b2(self, x, y):
        return - 1. / self.noise


    def _update_KB(self, x):
        KBnew = np.zeros((self.t + 1, self.t + 1))
        KBnew[:self.t, :self.t] = self._KB
        KBnew[self.t,  :self.t] = KBnew[:self.t, self.t] = self.kernel(self.x, x).squeeze()

        self._KB = KBnew


    def _update_C_alpha(self, x, y):
        b1 = self._calculate_b1(x, y)
        b2 = self._calculate_b2(x, y)

        k  = self.kernel(np.atleast_2d(self.x), x).squeeze()
        k  = np.append(k, 0)
        e  = np.zeros_like(k)
        e[-1] = 1

        C  = np.append(self._C, np.zeros((self.t, 1)),     axis = 1)
        C  = np.append(C,       np.zeros((1, self.t + 1)), axis = 0)
        Cke = C @ k + e
        C  += b2 * np.outer(Cke, Cke)
        self._C = C

        alpha = np.vstack((self._alpha, 0.))
        alpha += b1 * Cke[:, np.newaxis]
        self._alpha = alpha

    def _update_Q(self, x):
        k   = self.kernel(self.x, x)
        Qk  = self._Q @ k
        Qke = np.vstack((Qk, 1.))

        Qnew    = np.zeros((self.t + 1, self.t + 1))
        Qnew[:self.t, :self.t] = self._Q
        Qnew   += (1. - k.T @ Qk) * np.outer(Qke, Qke)
        self._Q = Qnew

    def update(self, x, y):
        self._update_C_alpha(x, y)
        self._update_KB(x)
        self._update_Q(x)
        self.x = np.vstack((self.x, x))
        self.t += 1