# -*- coding: utf8 -*-

import unittest
import numpy.testing as npt
import numpy as np
from tsf.pipeline import TSFPipeline
from tsf.windows import SimpleAR, DinamicWindow, ClassChange
from sklearn.linear_model import LassoCV
from random import randint


class TestPipeline(unittest.TestCase):
    data = np.array([[0, 0, 0, 1, 2, 2, 2, 1, 1, 1],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])
    metrics = ['mean', 'variance', 'mean']

    # Just transformation test
    def test_transform(self):
        expected_x = [[0, 0, 0, 11, 12, 13, 21, 22, 23, 0, 0, 12, 0.66666667, 22, 0.66666667, 12, 0.66666667, 22, 0.66666667],
                    [0, 0, 1, 12, 13, 14, 22, 23, 24, 1, 0, 13, 0.66666667, 23, 0.66666667, 14, 0, 24, 0],
                    [0, 1, 2, 13, 14, 15, 23, 24, 25, 2, 0, 14, 0.66666667, 24, 0.66666667, 15, 0, 25, 0],
                    [1, 2, 2, 14, 15, 16, 24, 25, 26, 2, 0, 15, 0.66666667, 25, 0.66666667, 15.5, 0.25, 25.5, 0.25],
                    [2, 2, 2, 15, 16, 17, 25, 26, 27, 2, 0, 16, 0.66666667, 26, 0.66666667, 16, 0.66666667, 26, 0.66666667],
                    [2, 2, 1, 16, 17, 18, 26, 27, 28, 1, 0, 17, 0.66666667, 27, 0.66666667, 18, 0, 28, 0],
                    [2, 1, 1, 17, 18, 19, 27, 28, 29, 1, 0, 18, 0.66666667, 28, 0.66666667, 18.5, 0.25, 28.5, 0.25]]
        expected_y = [1, 2, 2, 2, 1, 1, 1]

        pipe = TSFPipeline([('ar', SimpleAR(n_prev=3)),
                            ('dw', DinamicWindow()),
                            ('cc', ClassChange())])
        X_t, y_t = pipe.transform(X=[], y=self.data)

        npt.assert_allclose(expected_x, X_t)
        npt.assert_allclose(expected_y, y_t)

    def test_fit(self):
        lineal_serie = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        train = lineal_serie[:10]
        test = lineal_serie[10:]
        pipe = TSFPipeline([('ar', SimpleAR(n_prev=2)),
                            ('lasso', LassoCV())])
        pipe.fit(X=[], y=train)

        predicted = pipe.predict(test)

        npt.assert_allclose(pipe.offset_y(test, predicted), predicted, atol=0.05, rtol=0)


if __name__ == '__main__':
    unittest.main()