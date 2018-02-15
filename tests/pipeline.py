# -*- coding: utf8 -*-

import sys
sys.path.append('/Users/x5981md/time-series-forecasting/tsf/')
sys.path.append('/Users/x5981md/time-series-forecasting/tsf/pipeline')
from time_series_forescaster import SimpleAR, DinamicWindow
from tsf_pipeline import tsf_pipeline
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

import click
import numpy as np
import pandas as pd

# Some stat handlers examples
def var_function(samples):
    return np.var(samples)


def mean_function(samples):
    return np.mean(samples)


@click.command()
@click.option('--file', '-f', default=None, required=True, help=u'Fichero de serie temporal')
@click.option('--ratio', '-v', default=0.1, required=False, help=u'Ratio de stat total')
@click.option('--test', '-t', default=0.3, required=False, help=u'Ratio de muestras para test')
def run_pipeline_test(file, ratio, test):
    # Read
    data = pd.read_csv(file, header=None)
    data = data.values.reshape(1, data.values.shape[0])
    data = data[0]

    # Split
    n_data = len(data)
    n_test = int(n_data * test)
    test = data[n_test:]
    train = data[:n_test]

    # Create pipeline
    pipe = tsf_pipeline([('ar', SimpleAR(n_prev=3)),
                     ('dw', DinamicWindow(stat=var_function, ratio=ratio, metrics=['variance', mean_function])),
                     ('regressor', LassoCV(random_state=0))])

    # Fit pipeline
    pipe.fit(X=[], y=train)

    # Predict using Pipeline
    predicted_train = pipe.predict(train)
    predicted_test = pipe.predict(test)

    # MSE
    mse_train = mean_squared_error(pipe.offset_y(train, predicted_train), predicted_train)
    mse_test = mean_squared_error(pipe.offset_y(test, predicted_test), predicted_test)

    print "MSE train: " + str(mse_train)
    print "MSE test: " + str(mse_test)

if __name__ == "__main__":
    run_pipeline_test()