# -*- coding: utf8 -*-

import numpy as np
import sys
sys.path.append('/Users/migueldiazlozano/Desktop/Ingeniería Informática/TFG/TSF')
from tsf.time_series_forescaster import SimpleAR, DinamicWindow, TimeSeriesForecaster
from tsf.pipeline.tsf_pipeline import tsf_pipeline
from sklearn.linear_model import LassoCV

def run_pipeline_test():
    # Data
    train_data = np.array(range(0, 20, 1))
    test_data  = np.array(range(30, 40, 1))

    # Create pipeline
    pipe = tsf_pipeline([('ar', SimpleAR(n_prev=3)),
                     ('dw', DinamicWindow(stat='variance', ratio=0.1, metrics=['mean', 'variance'])),
                     ('regressor', LassoCV())])

    # Fit pipeline
    pipe.fit(X=[], y=train_data)

    # Predict using Pipeline
    pipe.predict(test_data)


if __name__ == "__main__":
    run_pipeline_test()