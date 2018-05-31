# -*- coding: utf8 -*-

import sys
import unittest
import numpy.testing as npt
import numpy as np
from random import randint
sys.path.append('../tsf')
from tsf.windows import DinamicWindow
from tsf.tsf_tools import _dinamic_window_delegate, incremental_variance


def var(samples):
    return np.var(samples)


def stddev(samples):
    return np.sqrt(np.var(samples))


class TestDinamicWindow(unittest.TestCase):
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])
    metrics = ['mean', 'variance', 'mean', var, stddev]

    # White
    def test_single_maker(self):
        ratio = 0.1
        expected = [[1, 0],
                    [1.5, 0.25],
                    [2, 0.6666667],
                    [3, 0.6666667],
                    [4, 0.6666667],
                    [5, 0.6666667],
                    [6, 0.6666667],
                    [7, 0.6666667],
                    [8, 0.6666667]]

        result = _dinamic_window_delegate(self.data[0], incremental_variance, metrics=self.metrics[0:2], ratio=ratio,
                                          horizon=1)

        # Test data
        npt.assert_allclose(result, expected)

    def test_multi_maker(self):

        ratio = 0.1
        expected = [[1, 0, 11, 0, 21, 0],
                    [1.5, 0.25, 11.5, 0.25, 21.5, 0.25],
                    [2, 0.6666667, 12, 0.6666667, 22, 0.6666667],
                    [3, 0.6666667, 13, 0.6666667, 23, 0.6666667],
                    [4, 0.6666667, 14, 0.6666667, 24, 0.6666667],
                    [5, 0.6666667, 15, 0.6666667, 25, 0.6666667],
                    [6, 0.6666667, 16, 0.6666667, 26, 0.6666667],
                    [7, 0.6666667, 17, 0.6666667, 27, 0.6666667],
                    [8, 0.6666667, 18, 0.6666667, 28, 0.6666667]]

        dw = DinamicWindow(ratio=ratio, stat='variance', horizon=1)
        Xt = dw.transform(X=[], y=self.data)

        # Test data
        npt.assert_allclose(Xt, expected)

    def test_incremental_variance(self):
        ratio = 0.1
        for _ in range(1, 10):
            # Completely random problem
            horizon = randint(1, 30)
            series_number = randint(1, 100)
            series_length = randint(100, 500)

            metrics_number = randint(1, len(self.metrics))
            random_data = np.random.rand(series_number, series_length)

            metrics = self.metrics[0:metrics_number]

            dw = DinamicWindow(ratio=ratio, stat=var, metrics=metrics, horizon=horizon)
            var_function_result = dw.transform(X=[], y=random_data)
            dw = DinamicWindow(ratio=ratio, stat='variance', metrics=metrics, horizon=horizon)
            incremental_variance_result = dw.transform(X=[], y=random_data)

            # Test
            npt.assert_allclose(var_function_result, incremental_variance_result)

    def test_shape(self):
        ratio = 0.1
        for _ in range(1, 10):
            # Completely random problem
            horizon = randint(1, 30)
            series_number = randint(1, 100)
            series_length = randint(100, 500)

            metrics_number = randint(1, len(self.metrics))
            random_data = np.random.rand(series_number, series_length)

            metrics = self.metrics[0:metrics_number]

            dw = DinamicWindow(ratio=ratio, stat='variance', metrics=metrics, horizon=horizon)
            Xt = dw.transform(X=[], y=random_data)

            # Test shape
            self.assertEquals(Xt.shape[0], random_data.shape[1] - horizon)
            self.assertEquals(Xt.shape[1], random_data.shape[0] * metrics_number)


if __name__ == "__main__":
    unittest.main()