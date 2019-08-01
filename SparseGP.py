import numpy as np

## Three challenges to vectorization:
## 1. Inputs to kernels must be 2d
## 2. KBinv must be initialized properly
## 3. Overflow errors must be anticipated


class SparseGP:

    def __init__(self, x0, y0, r, noise):
        self._alpha = np.zeros((1,))
        self._C     = np.zeros((1, 1))
        self._KB    = np.zeros((1, 1))
        self._KBinv = np.zeros((1, 1))
        self.x      = np.array([x0])
        self.noise  = noise
        self.r      = r
        self.t      = 1


    def K0(self, x0, x1):
        return np.exp(- np.sum((x1 - x0) ** 2) / self.r)


    def k(self, x):
        x_length = len(x)
        if x_length == 1:
            k = np.zeros((self.t,))
            for i in range(self.t):
                k[i] = self.K0(self.x[i], x)
            return k
        else :
            k = np.zeros((self.t, x_length))
            for i in range(self.t):
                for j in range(x_length):
                    k[i, j] = self.K0(self.x[i], x[j])
            return k



    def a(self, x):
        k = self.k(x)
        return np.inner(k, self._alpha)


    def novelty_individual(self, xr):
        k = self.k(xr)
        gamma = k.T @ self._KBinv @ k
        return gamma


    def novelty(self, X, ridx):
        gamma = np.zeros_like(ridx)
        gamma_idx = 0

        for l in ridx:
            try:
                gamma[gamma_idx] = self.novelty_individual(X[l])
            except(OverflowError):
                gamma = np.zeros_like(ridx)
                gamma[gamma_idx] = 1.
                return gamma
            gamma_idx += 1

        return gamma


    def _calculate_b1(self, x, y):
        return (y - self.a(x)) / self.noise


    def _calculate_b2(self, x, y):
        return - 1. / self.noise


    # def _update_alpha(self, x, y):
    #     b1 = self._calculate_b1(x, y)
    #     k  = self.k(x)
    #     k  = np.append(k, 0)
    #     e  = np.zeros_like(k)
    #     e[-1] = 1

    #     alpha = np.append(self._alpha, 0)
    #     alpha += b1 * (self._C @ k + e)
    #     self._alpha = alpha


    def _update_KB(self, x):
        KBnew = np.zeros((self.t + 1, self.t + 1))
        KBnew[:self.t, :self.t] = self._KB
        KBnew[self.t, :self.t] = KBnew[:self.t, self.t] = self.k(x)
        
        self._KB = KBnew


    # def _update_C(self, x, y):
    #     b2 = self._calculate_b2(x, y)
    #     k  = self.k(x)
    #     k  = np.append(k, 0)
    #     e  = np.zeros_like(k)
    #     e[-1] = 1

    #     C  = np.append(self._C, np.zeros((self.t, 1)),     axis = 1)
    #     C  = np.append(self._C, np.zeros((1, self.t + 1)), axis = 0)
    #     Cke = C @ k + e
    #     C  += b2 * np.outer(Cke, Cke)
    #     self._C = C


    def _update_C_alpha(self, x, y):
        b1 = self._calculate_b1(x, y)
        b2 = self._calculate_b2(x, y)

        k  = self.k(x)
        k  = np.append(k, 0)
        e  = np.zeros_like(k)
        e[-1] = 1

        C  = np.append(self._C, np.zeros((self.t, 1)),     axis = 1)
        C  = np.append(C,       np.zeros((1, self.t + 1)), axis = 0)
        Cke = C @ k + e
        C  += b2 * np.outer(Cke, Cke)
        self._C = C

        alpha = np.append(self._alpha, 0)
        alpha += b1 * Cke
        self._alpha = alpha


    def update(self, x, y):
        self._update_C_alpha(x, y)
        self._update_KB(x)
        self._KBinv = np.linalg.inv(self._KB)
        self.x = np.append(self.x, x)
        self.t += 1