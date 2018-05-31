# -*- coding: utf8 -*-

import sys
import unittest
import numpy.testing as npt
import numpy as np
sys.path.append('../tsf')
from tsf.windows import ClassChange
from tsf.tsf_tools import _classchange_window_delegate
from random import randint


def var(samples):
    return np.var(samples)


def stddev(samples):
    return np.sqrt(np.var(samples))


class TestClassChange(unittest.TestCase):
    data = np.array([[0, 0, 0, 1, 2, 2, 2, 1, 1, 1],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])
    metrics = ['mean', 'variance', 'mean', var, stddev]

    def test_multi_maker(self):
        expected = [[11, 0, 21,  0],
                    [11.5, 0.25,21.5, 0.25],
                    [12, 0.6666667, 22, 0.6666667],
                    [14, 0, 24, 0],
                    [15, 0, 25, 0],
                    [15.5, 0.25, 25.5, 0.25],
                    [16, 0.6666667, 26, 0.6666667],
                    [18, 0, 28, 0],
                    [18.5, 0.25, 28.5, 0.25]]
        result = _classchange_window_delegate(self.data[0], self.data[1:], metrics=['mean', 'variance'], horizon=1)
        # Test data
        npt.assert_allclose(result, expected)

    def test_shape(self):
        # Test shape
        for _ in range(1, 50):
            # Completely random problem
            series_length = randint(50, 500)
            exogs_number = np.random.randint(1, 4)
            endog_data = np.random.randint(3, size=series_length)
            exogs_data = np.random.rand(exogs_number, series_length)

            metrics_number = randint(1, len(self.metrics))
            metrics = self.metrics[0:metrics_number]

            horizon = randint(1, 30)
            random_data = np.insert(exogs_data, 0, endog_data, 0)

            cc = ClassChange(metrics=metrics, horizon=horizon)
            result = cc.transform(X=[], y=random_data)

            self.assertEquals(result.shape[0], series_length - horizon)
            self.assertEquals(result.shape[1], exogs_number * metrics_number)


if __name__ == '__main__':
    unittest.main()
