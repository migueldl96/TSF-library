# -*- coding: utf8 -*-

import numpy as np
import sys
sys.path.append('/Users/migueldiazlozano/Desktop/Ingeniería Informática/TFG/TSF')
from tsf.time_series_forescaster import SimpleAR, DinamicWindow, TimeSeriesForecaster
from sklearn.pipeline import Pipeline


def run_pipeline_test():
    # Data
    train_data = np.array(range(0, 20, 1))
    test_data  = np.array(range(30, 40, 1))

    print "Train data"
    print train_data

    # Transform the data
    ar = SimpleAR(n_prev=4)
    X_train = ar.fit_transform(X=[], y=train_data)
    X_test = ar.fit_transform(X=[], y=test_data)

    print "AR train inputs:"
    print X_train

    dw = DinamicWindow(stat='variance', ratio=0.1, metrics=['mean', 'variance'])
    X_train = dw.fit_transform(X=X_train, y=train_data)
    X_test = dw.fit_transform(X=X_test, y=test_data)

    print "DW+AR train inputs:"
    print X_train

    model = TimeSeriesForecaster()
    model.fit(X_train, y=train_data)

    print "Real test data:"
    print test_data
    print "Predicted test data:"
    print model.predict(X_test)

if __name__ == "__main__":
    run_pipeline_test()