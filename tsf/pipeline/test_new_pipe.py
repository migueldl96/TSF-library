# -*- coding: utf8 -*-

import sys
sys.path.append('/Users/migueldiazlozano/Desktop/Ingeniería Informática/TFG/TSF')
from tsf.pipeline import tsf_pipeline
import tsf.time_series_forescaster

def test_pipe():
    ar = SimpleAR()
    dw = DinamicWindow()

    pipe = tsf_pipeline(['ar', SimpleAR(), ('dw', DinamicWindow()), ('', None)])

if __name__ == "__main__":
    test_pipe()