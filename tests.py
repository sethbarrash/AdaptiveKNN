import unittest
import numpy as np
import SparseGP as sgp
from adaptiveknn import *

from sklearn.gaussian_process.kernels import RBF

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