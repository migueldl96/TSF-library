# -*- coding: utf8 -*-

import sys
import unittest
import numpy.testing as npt
import numpy as np
sys.path.append('../tsf')
from tsf_windows import DinamicWindow
from tsf_tools import _dinamic_window_delegate, incremental_variance


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
        expected = [[1.5, 0.25],
                    [2, 0.6666667],
                    [3, 0.6666667],
                    [4, 0.6666667],
                    [5, 0.6666667],
                    [6, 0.6666667],
                    [7, 0.6666667],
                    [8, 0.6666667]]

        result = _dinamic_window_delegate(self.data[0], incremental_variance, metrics=self.metrics[0:2], ratio=ratio)

        # Test data
        npt.assert_allclose(result, expected)

    def test_multi_maker(self):

        ratio = 0.1
        expected = [[1.5, 0.25, 11.5, 0.25, 21.5, 0.25],
                    [2, 0.6666667, 12, 0.6666667, 22, 0.6666667],
                    [3, 0.6666667, 13, 0.6666667, 23, 0.6666667],
                    [4, 0.6666667, 14, 0.6666667, 24, 0.6666667],
                    [5, 0.6666667, 15, 0.6666667, 25, 0.6666667],
                    [6, 0.6666667, 16, 0.6666667, 26, 0.6666667],
                    [7, 0.6666667, 17, 0.6666667, 27, 0.6666667],
                    [8, 0.6666667, 18, 0.6666667, 28, 0.6666667]]

        dw = DinamicWindow(ratio=ratio, stat='variance')
        Xt = dw.transform(X=[], y=self.data)

        # Test data
        npt.assert_allclose(Xt, expected)

    def test_shape(self):
        ratio = 0.1
        for index in range(1, len(self.metrics)):
            metrics = self.metrics[0:index]
            dw = DinamicWindow(ratio=ratio, stat='variance', metrics=metrics)
            Xt = dw.transform(X=[], y=self.data)

            # Test shape
            self.assertEquals(Xt.shape[0], self.data.shape[1] - 2)
            self.assertEquals(Xt.shape[1], self.data.shape[0] * len(metrics))


if __name__ == "__main__":
    unittest.main()