# -*- coding: utf8 -*-

import unittest
import numpy.testing as npt
import numpy as np
from tsf.pipeline import TSFPipeline
from tsf.grid_search import TSFGridSearch
from tsf.windows import SimpleAR, DinamicWindow, ClassChange
from sklearn.tree import DecisionTreeClassifier


class TestGridSearch(unittest.TestCase):
    data = np.array([[0, 0, 0, 1, 2, 2, 2, 1, 1, 1, 0, 0, 2, 0, 0, 1, 2, 0, 2, 1],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]])
    metrics = ['mean', 'variance', 'mean']

    # Just transformation test
    def test_fit(self):
        pipe = TSFPipeline([('ar', SimpleAR(n_prev=3)),
                            ('dw', DinamicWindow()),
                            ('cc', ClassChange()),
                            ('DecisionTree', DecisionTreeClassifier())])
        params = [
            {
                'ar__n_prev': [1, 2, 3]
            },
            {
                'dw__ratio': [0.1, 0.2]
            },
            {
                'criterion': ["gini", "entropy"]
            }
        ]
        grid = TSFGridSearch(pipe, params)

        # We don't have best_params_ dict until we fit
        self.assertFalse(hasattr(grid, "best_params_"))

        grid.fit(X=[], y=self.data)

        # We already have it
        self.assertTrue(grid.best_params_)


if __name__ == '__main__':
    unittest.main()
