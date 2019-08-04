import unittest
import numpy as np
import SparseGaussianProcess as sgp
from copy import deepcopy
from adaptiveknn import *

from sklearn.gaussian_process.kernels import RBF

# class Test_ktree(unittest.TestCase):



class Test_sparseGaussianProcess(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.X = np.array([[1., 2., 4.],
                           [2., 2., 5.]])
        self.y = np.array([[2.5], [3.5]])
        self.x0 = np.atleast_2d(self.X[0])
        self.x1 = np.atleast_2d(self.X[1])
        self.xnew = np.array([[4., 6., 7.]])
        self.ynew = np.array([4.5])
        self.sgp = sgp.SparseGaussianProcess(self.x0, self.y[0], 1., 4.)
        self.kernel = RBF(1.)

    def test_sgp_init(self):
        np.testing.assert_array_equal(self.sgp.x, self.x0)
        np.testing.assert_array_equal(self.sgp._alpha, np.atleast_2d(self.y[0, 0] / self.sgp.noise))

    def test_sgp_predict(self):
        yhat = self.sgp.predict(self.x1)
        alpha1 = self.y[0] / self.sgp.noise
        K1 = self.kernel(self.x0, self.x1)
        yhat_true = alpha1 * K1
        np.testing.assert_array_equal(yhat, yhat_true)

    def test_sgp_calculate_b1(self):
        b1_true = np.atleast_1d(self.ynew / self.sgp.noise)
        b1 = self.sgp._calculate_b1(self.xnew, self.ynew)
        np.testing.assert_array_almost_equal(b1, b1_true, 3)

    def test_sgp_update_KB(self):
        sgp_copy = deepcopy(self.sgp)
        sgp_copy._update_KB(self.x1)

        k = self.kernel(self.x0, self.x1)
        KBtrue = np.zeros((2, 2))
        KBtrue[0, 1] = KBtrue[1, 0] = k

        np.testing.assert_array_equal(sgp_copy._KB, KBtrue)

    def test_sgp_update_C_alpha(self):
        sgp_copy = deepcopy(self.sgp)
        sgp_copy._update_C_alpha(self.x1, self.y[1])

        yhat_2 = self.sgp.predict(self.x1).squeeze()
        b1_1 = self.y[0, 0] / self.sgp.noise
        b1_2 = (self.y[1, 0] - yhat_2) / self.sgp.noise
        b2 = -1. / self.sgp.noise
        k2 = self.kernel(self.x0, self.x1).squeeze()


        alpha_true = np.array([[b1_1 + b1_2*b2*k2], [b1_2]])
        C_true     = np.array([[b2**3 * k2**2 + b2, b2**2 * k2],
                               [b2**2 * k2,         b2]])

        np.testing.assert_array_equal(sgp_copy._alpha, alpha_true)
        np.testing.assert_array_equal(sgp_copy._C,     C_true)

    def test_sgp_update_Q(self):
        sgp_copy = deepcopy(self.sgp)
        sgp_copy._update_Q(self.x1)


        k2 = self.kernel(self.x0, self.x1).squeeze()
        Q_true = np.zeros((2, 2))
        Q_true[0, 0] = 1.
        Q_true += (1 - k2**2) * np.array([[k2**2, k2],
                                          [k2,    1.]])

        np.testing.assert_array_equal(sgp_copy._Q, Q_true)

    def test_sgp_update(self):
        sgp_copy = deepcopy(self.sgp)
        sgp_copy.update(self.x1, self.y[1, 0])

        ## Test that KB is correct
        k = self.kernel(self.x0, self.x1)
        KBtrue = np.zeros((2, 2))
        KBtrue[0, 1] = KBtrue[1, 0] = k

        np.testing.assert_array_equal(sgp_copy._KB, KBtrue)

        ## Test that alpha and c are correct
        yhat_2 = self.sgp.predict(self.x1).squeeze()
        b1_1 = self.y[0, 0] / self.sgp.noise
        b1_2 = (self.y[1, 0] - yhat_2) / self.sgp.noise
        b2 = -1. / self.sgp.noise
        k2 = self.kernel(self.x0, self.x1).squeeze()


        alpha_true = np.array([[b1_1 + b1_2*b2*k2], [b1_2]])
        C_true     = np.array([[b2**3 * k2**2 + b2, b2**2 * k2],
                               [b2**2 * k2,         b2]])

        np.testing.assert_array_equal(sgp_copy._alpha, alpha_true)
        np.testing.assert_array_equal(sgp_copy._C,     C_true)

        ## Test that Q is correct
        k2 = self.kernel(self.x0, self.x1).squeeze()
        Q_true = np.zeros((2, 2))
        Q_true[0, 0] = 1.
        Q_true += (1 - k2**2) * np.array([[k2**2, k2],
                                          [k2,    1.]])

        np.testing.assert_array_equal(sgp_copy._Q, Q_true)

        ## Test that x is correct
        np.testing.assert_array_equal(sgp_copy.x, self.X)

        ## Test that t is correct
        self.assertEqual(sgp_copy.t, 2)



class Test_spincom(unittest.TestCase):

    def test_get_neighbor_order_0(self):
        x = np.power(2., np.arange(5))
        y = np.power(2., np.arange(5))
        nidx = get_neighbor_order(x, y, 2)
        np.testing.assert_array_equal(nidx, [1, 0, 3, 4])

    def test_get_neighbor_order_1(self):
        X = np.power([[2.], [2.]], np.arange(5)).T
        y = np.power(2., np.arange(5))
        nidx = get_neighbor_order(X, y, 2)
        np.testing.assert_array_equal(nidx, [1, 0, 3, 4])

    def test_get_neighborhood_0(self):
        x = np.power(2., np.arange(5))
        y = np.power(2., np.arange(5))

        xsorted, ysorted = get_neighborhood(x, y, 2)

        np.testing.assert_array_equal(xsorted, np.array([2., 1., 8., 16.]))
        np.testing.assert_array_equal(ysorted, np.array([2., 1., 8., 16.]))

    def test_get_neighborhood_1(self):
        X = np.power([[2.], [2.]], np.arange(5)).T
        y = np.power(2., np.arange(5))

        Xsorted, ysorted = get_neighborhood(X, y, 2)

        Xsortedtrue = np.array([[2.,  2.],
                                [1.,  1.],
                                [8.,  8.],
                                [16., 16.]])
        np.testing.assert_equal(Xsorted, Xsortedtrue)
        np.testing.assert_equal(ysorted, np.array([2., 1., 8., 16.]))

    def test_get_neighbor_errors_0(self):
        ysorted = np.arange(5)
        y0 = 2.
        etrue = [-2., -1.5, -1., -0.5, 0.]

        e = calculate_neighbor_errors(ysorted, y0)
        np.testing.assert_array_equal(e, etrue)



class TestTrainTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        np.random.seed(55455)
        self.Xtrain = np.random.normal(0, 1, (50, 5))
        self.ytrain = np.random.normal(0, 1, (50,))
        self.Xtest  = np.random.normal(0, 1, (10, 5))
        self.ytest  = np.random.normal(0, 1, (10,))

    def test_knnTrainTest(self):
        np.random.seed(55455)
        Xtrain = np.random.normal(0, 1, (50, 5))
        ytrain = np.random.normal(0, 1, (50,))
        Xtest  = np.random.normal(0, 1, (10, 5))
        ytest  = np.random.normal(0, 1, (10,))

        errors, trainTime, testTime = knnTrainTest(Xtrain, ytrain, Xtest, ytest, 1)

        self.assertEqual(len(errors), len(ytest))
        self.assertLess(len(errors.squeeze().shape), 2)
        self.assertFalse(np.any(np.isnan(errors)))
        self.assertGreater(trainTime, 0)
        self.assertGreater(testTime,  0)


    def test_spincomTrainTest(self):
        np.random.seed(55455)
        n = 50
        Xtrain = np.random.normal(0, 1, (n, 5))
        ytrain = np.random.normal(0, 1, (n,))
        Xtest  = np.random.normal(0, 1, (10, 5))
        ytest  = np.random.normal(0, 1, (10,))

        hyperparams = {"m" : 20, "h" : 1., "sigma" : 4.}
        errors, ktest, trainTime, testTime = spincomTrainTest(Xtrain, ytrain, Xtest, ytest, hyperparams)

        self.assertEqual(len(errors), len(ytest))
        self.assertLess(len(errors.squeeze().shape), 2)
        self.assertFalse(np.any(np.isnan(errors)))
        self.assertEqual(len(ktest), len(ytest))
        self.assertFalse(np.any(np.isnan(ktest)))
        self.assertFalse(np.any(np.logical_or(ktest > n, ktest < 1)))
        self.assertTrue(np.issubdtype(ktest.dtype, np.integer))
        self.assertGreater(trainTime, 0)
        self.assertGreater(testTime,  0)

    def test_adaknnTrainTest(self):
        np.random.seed(55455)
        n = 50
        Xtrain = np.random.normal(0, 1, (n, 5))
        ytrain = np.random.normal(0, 1, (n,))
        Xtest  = np.random.normal(0, 1, (10, 5))
        ytest  = np.random.normal(0, 1, (10,))

        hyperparams = {"alpha" : 10, "kmax" : 25}
        errors, ktest, trainTime, testTime = adaknnTrainTest(Xtrain, ytrain, Xtest, ytest, hyperparams)

        self.assertEqual(len(errors), len(ytest))
        self.assertLess(len(errors.squeeze().shape), 2)
        self.assertFalse(np.any(np.isnan(errors)))
        self.assertEqual(len(ktest), len(ytest))
        self.assertFalse(np.any(np.isnan(ktest)))
        self.assertFalse(np.any(np.logical_or(ktest > n, ktest < 1)))
        self.assertTrue(np.issubdtype(ktest.dtype, np.integer))
        self.assertGreater(trainTime, 0)
        self.assertGreater(testTime,  0)

    def test_ktreeTrainTest(self):
        np.random.seed(55455)
        n = 50
        Xtrain = np.random.normal(0, 1, (n, 5))
        ytrain = np.random.normal(0, 1, (n,))
        Xtest  = np.random.normal(0, 1, (10, 5))
        ytest  = np.random.normal(0, 1, (10,))

        hyperparams = {"rho1" : 1e-5, "rho2" : 1e-5, "k" : 5, "sigma" : 1e-5}
        errors, ktest, trainTime, testTime = ktreeTrainTest(Xtrain, ytrain, Xtest, ytest, hyperparams)

        self.assertEqual(len(errors), len(ytest))
        self.assertLess(len(errors.squeeze().shape), 2)
        self.assertFalse(np.any(np.isnan(errors)))
        self.assertEqual(len(ktest), len(ytest))
        self.assertFalse(np.any(np.isnan(ktest)))
        self.assertFalse(np.any(np.logical_or(ktest > n, ktest < 1)))
        self.assertTrue(np.issubdtype(ktest.dtype, np.integer))
        self.assertGreater(trainTime, 0)
        self.assertGreater(testTime,  0)



@unittest.skip("Computationally expensive tests")
class TestOptimizeHyperparameters(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        np.random.seed(55455)
        self.Xtrain = np.random.normal(0, 1, (50, 5))
        self.ytrain = np.random.normal(0, 1, (50,))
        self.Xtest  = np.random.normal(0, 1, (10, 5))
        self.ytest  = np.random.normal(0, 1, (10,))


    def test_optimizek(self):
        ks = np.power(2., np.arange(6)).astype("int")
        k = optimize_k(self.Xtrain, self.ytrain, self.Xtest, self.ytest, ks)[0]
        self.assertIn(k, ks)


    def test_alphas_kmaxs_to_index_0(self):
        alphas = np.array([10, 20, 30, 40])
        kmaxs  = np.array([25, 50, 75])
        alphas_idx = np.array([10, 10, 10, 20, 20, 20, 30, 30, 40, 40])
        kmaxs_idx  = np.array([25, 50, 75, 25, 50, 75, 50, 75, 50, 75])
        idx_true = pd.MultiIndex.from_arrays([alphas_idx, kmaxs_idx])
        idx = alphas_kmaxs_to_index(alphas, kmaxs)
        pd.testing.assert_index_equal(idx, idx_true)

    def test_alphas_kmaxs_to_index_1(self):
        np.random.seed(55455)
        alphas = np.random.randint(1, 100, (100,))
        kmaxs  = np.random.randint(1, 100, (100,))
        idx = alphas_kmaxs_to_index(alphas, kmaxs)

        for i in range(len(idx)):
            pair = idx[i]
            self.assertLessEqual(pair[0], pair[1])

    def test_optimize_sigma(self):
        h = 1.
        m = 20
        sigmas = np.power(2., np.arange(5))
        sigma = optimize_sigma(self.Xtrain, self.ytrain, self.Xtest, self.ytest,
            sigmas, m, h)[0]
        self.assertIn(sigma, sigmas)

    def test_optimize_hyperparams_adaknn(self):
        alphas = np.array([10, 20, 30, 40])
        kmaxs  = np.array([25, 50, 75])
        alpha, kmax = optimize_hyperparams_adaknn(self.Xtrain, self.ytrain, 
            self.Xtest, self.ytest, alphas, kmaxs)[:2]
        self.assertIn(alpha, alphas)
        self.assertIn(kmax, kmaxs)

    def test_optimize_hyperparams_ktree(self):
        rhos1  = np.power(10., np.arange(-5, -2))
        rhos2  = np.power(10., np.arange(-5, -2))
        ks     = np.power(2.,  np.arange(5)).astype(int)
        sigmas = np.power(10., np.arange(-5, -2))
        hyperparam_grid = {
            "rhos1"  : rhos1,
            "rhos2"  : rhos2,
            "ks"     : ks,
            "sigmas" : sigmas,
        }
        rho1, rho2, k, sigma = optimize_hyperparams_ktree(self.Xtrain,
            self.ytrain, self.Xtest, self.ytest, hyperparam_grid)[:4]
        self.assertIn(rho1,  rhos1)
        self.assertIn(rho2,  rhos2)
        self.assertIn(k,     ks)
        self.assertIn(sigma, sigmas)



class TestUtilities(unittest.TestCase):

    def test_atleast_2d_T_0(self):
        x = 5
        x = atleast_2d_T(x)
        xtrue = np.array([[5]])
        np.testing.assert_array_equal(x, xtrue)

    def test_atleast_2d_T_1(self):
        x = np.array([1, 2])
        x = atleast_2d_T(x)
        xtrue = np.array([[1],
                          [2]])
        np.testing.assert_array_equal(x, xtrue)

    def test_atleast_2d_T_2(self):
        x = np.array([[1],
                      [2]])
        xtrue = np.array([[1],
                          [2]])
        x = atleast_2d_T(x)
        np.testing.assert_array_equal(x, xtrue)

    def test_atleast_2d_T_3(self):
        x = np.array([[1, 2],
                      [2, 3]])
        xtrue = np.array([[1, 2],
                           [2, 3]])
        x = atleast_2d_T(x)
        np.testing.assert_array_equal(x, xtrue)


    def test_chisel_k_0(self):
        n = 100
        k = 0
        k = chisel_k(k, n)
        ktrue = 1
        self.assertEqual(k, ktrue)

    def test_chisel_k_1(self):
        n = 100
        k = 1.75
        k = chisel_k(k, n)
        ktrue = 1
        self.assertEqual(k, ktrue)

    def test_chisel_k_2(self):
        n = 100
        k = 110
        k = chisel_k(k, n)
        ktrue = 100
        self.assertEqual(k, ktrue)



class Test_real_data_tests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        np.random.seed(55455)
        self.Xtrain = np.random.normal(0, 1, (50, 5))
        self.ytrain = np.random.normal(0, 1, (50,))
        self.Xvalid = np.random.normal(0, 1, (20, 5))
        self.yvalid = np.random.normal(0, 1, (20,))
        self.Xtest  = np.random.normal(0, 1, (10, 5))
        self.ytest  = np.random.normal(0, 1, (10,))
        self.data_partition = ("Air", self.Xtrain, self.Xvalid, self.Xtest,
            self.ytrain, self.yvalid, self.ytest)

    def test_real_data_trial_knn(self):
        ks = np.power(2., np.arange(5)).astype(int)
        new_row = real_data_trial_knn(self.data_partition, ks)
        dataset, k, error, hyperparameterTime, trainTime, testTime = new_row.values

        try:
            k_is_int = np.issubdtype(k.dtype, np.integer)
        except(AttributeError):
            k_is_int = type(k) == int

        self.assertTrue(dataset)
        self.assertTrue(k_is_int)
        self.assertGreater(k, 0)
        self.assertLess(k, len(self.Xtrain))
        self.assertGreaterEqual(error, 0.)
        self.assertGreater(hyperparameterTime, 0.)
        self.assertGreater(trainTime, 0.)
        self.assertGreater(testTime, 0.)

    def test_real_data_trial_spincom(self):
        sigmas = np.power(2., np.arange(5))
        new_row = real_data_trial_spincom(self.data_partition, sigmas, 10, 1.)
        dataset, sigma, m, h, error, hyperparameterTime, trainTime, testTime = new_row.values

        self.assertTrue(dataset)
        self.assertGreater(sigma, 0)
        self.assertLess(sigma, len(self.Xtrain))
        self.assertGreater(m, 0)
        self.assertGreater(h, 0)
        self.assertGreaterEqual(error, 0.)
        self.assertGreater(hyperparameterTime, 0.)
        self.assertGreater(trainTime, 0.)
        self.assertGreater(testTime, 0.)


    def test_real_data_trial_adaknn(self):
        alphas = np.array([10, 20, 30, 40])
        kmaxs  = np.array([25, 50, 75])
        new_row = real_data_trial_adaknn(self.data_partition, alphas, kmaxs)
        dataset, alpha, kmax, error, hyperparameterTime, trainTime, testTime = new_row.values

        try:
            alpha_is_int = np.issubdtype(alpha.dtype, np.integer)
        except(AttributeError):
            alpha_is_int = type(alpha) == int

        try:
            kmax_is_int  = np.issubdtype(kmax.dtype,  np.integer)
        except(AttributeError):
            kmax_is_int = type(kmax) == int

        self.assertTrue(dataset)
        self.assertTrue(kmax_is_int)
        self.assertGreater(kmax, 1)
        self.assertTrue(alpha_is_int)
        self.assertGreater(alpha, 1)
        self.assertLess(alpha, kmax)
        self.assertGreaterEqual(error, 0.)
        self.assertGreater(hyperparameterTime, 0.)
        self.assertGreater(trainTime, 0.)
        self.assertGreater(testTime, 0.)


    def test_real_data_trial_ktree(self):
        rhos1  = np.power(10., np.arange(-5, -2))
        rhos2  = np.power(10., np.arange(-5, -2))
        ks     = np.power(2.,  np.arange(5)).astype(int)
        sigmas = np.power(10., np.arange(-5, -2))
        hyperparam_grid = {
            "rhos1"  : rhos1,
            "rhos2"  : rhos2,
            "ks"     : ks,
            "sigmas" : sigmas,
        }
        new_row = real_data_trial_ktree(self.data_partition, hyperparam_grid)
        dataset, rho1, rho2, k, sigma, error, hyperparameterTime, trainTime, testTime = new_row.values

        try:
            k_is_int  = np.issubdtype(k.dtype,  np.integer)
        except(AttributeError):
            k_is_int = type(k) == int

        self.assertTrue(dataset)
        self.assertGreater(rho1, 0.)
        self.assertGreater(rho2, 0.)
        self.assertTrue(k_is_int)
        self.assertGreater(k, 1)
        self.assertGreater(sigma, 0.)
        self.assertGreaterEqual(error, 0.)
        self.assertGreater(hyperparameterTime, 0.)
        self.assertGreater(trainTime, 0.)
        self.assertGreater(testTime, 0.)





if __name__ == "__main__":
    unittest.main()