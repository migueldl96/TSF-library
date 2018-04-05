# -*- coding: utf8 -*-

import sys
import unittest
import numpy.testing as npt
import numpy as np
sys.path.append('../tsf')
from tsf_windows import SimpleAR
from tsf_tools import _fixed_window_delegate


class TestSimpleAR(unittest.TestCase):
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])

    def test_single_maker(self):
        expected = [[1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9]]
        result = _fixed_window_delegate(self.data[0], 2)

        # Test shape
        self.assertEquals(result.shape[0], 8)
        self.assertEquals(result.shape[1], 2)

        # Test data
        npt.assert_allclose(result, expected)

    def test_multi_transform(self):
        n_prevs = [1, 2, 3, 4, 5, 6, 7, 8]

        # Test shape
        for n_prev in n_prevs:
            sar = SimpleAR(n_prev)
            Xt = sar.transform(X=[], y=self.data)

            self.assertEquals(Xt.shape[1], n_prev * self.data.shape[0])
            self.assertEquals(Xt.shape[0], self.data.shape[1] - n_prev)

        # Test data for n_prev = 2
        sar = SimpleAR(2)
        expected = [[1, 2, 11, 12, 21, 22],
                  [2, 3, 12, 13, 22, 23],
                  [3, 4, 13, 14, 23, 24],
                  [4, 5, 14, 15, 24, 25],
                  [5, 6, 15, 16, 25, 26],
                  [6, 7, 16, 17, 26, 27],
                  [7, 8, 17, 18, 27, 28],
                  [8, 9, 18, 19, 28, 29]]
        Xt = sar.transform(X=[], y=self.data)

        npt.assert_allclose(Xt, expected)


if __name__ == '__main__':
    unittest.main()
