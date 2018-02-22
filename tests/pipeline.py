# -*- coding: utf8 -*-

import sys
sys.path.append('/Users/x5981md/time-series-forecasting/tsf')
sys.path.append('/Users/x5981md/time-series-forecasting/tsf/pipeline')
sys.path.append('/Users/x5981md/time-series-forecasting/tsf/grid_search')
from time_series_forescaster import SimpleAR, DinamicWindow, RangeWindow
from tsf_pipeline import TSFPipeline
from tsf_grid import TSFGridSearchCV
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
@click.option('--files', '-f', type=click.Choice(['temp.txt', 'humidity.txt', 'windDir.txt', 'windSpeed.txt',
                                                 'QNH.txt']), multiple=True)
@click.option('--ratio', '-v', default=0.1, required=False, help=u'Ratio de stat total')
@click.option('--test', '-t', default=0.3, required=False, help=u'Ratio de muestras para test')
def run_pipeline_test(files, ratio, test):

    # Read
    data = read_data(files)

    # Split
    n_data = len(data[0])
    n_train = int(n_data * (1.0-test))
    train = data[:, :n_train]
    test = data[:, n_train:]

    # Create pipeline
    pipe = TSFPipeline([('ar', SimpleAR(n_prev=5)),
                        ('dw', DinamicWindow(ratio=ratio)),
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


def read_data(files):
    data = []
    for file in files:
        path = '../data/' + file
        single_serie = pd.read_csv(path, header=None).values
        single_serie = single_serie.reshape(1, single_serie.shape[0])

        if len(data) == 0:
            data = single_serie
        else:
            data = np.append(data, single_serie, axis=0)

    return data

if __name__ == "__main__":
    run_pipeline_test()