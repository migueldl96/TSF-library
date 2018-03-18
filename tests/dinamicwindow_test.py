# -*- coding: utf8 -*-

import sys
import unittest
import numpy.testing as npt
import numpy as np
sys.path.append('/Users/migueldiazlozano/Desktop/Ingeniería Informática/TFG/TSF/tsf')
from time_series_forescaster import DinamicWindow
from tsf_tools import _dinamic_window_delegate

class Test_DinamicWindow(unittest.TestCase):
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])

    def test_single_maker(self):
        pass


if __name__ == "__main__":
    unittest.main()