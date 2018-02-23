# -*- coding: utf8 -*-

import sys
sys.path.append('/Users/x5981md/time-series-forecasting/tsf')
sys.path.append('/Users/x5981md/time-series-forecasting/tsf/pipeline')
from time_series_forescaster import SimpleAR, DinamicWindow, RangeWindow
from tsf_pipeline import TSFPipeline
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


# RVR umbralizer
def umbralizer(sample):
    if sample < 1000:
        return 2
    elif 1000 < sample < 1990:
        return 1
    else:
        return 0


@click.command()
@click.option('--files', '-f', type=click.Choice(['temp.txt', 'humidity.txt', 'windDir.txt', 'windSpeed.txt',
                                                 'QNH.txt', 'RVR.txt']), multiple=True)
@click.option('--ratio', '-r', default=0.1, required=False, help=u'Ratio de stat total')
@click.option('--test_r', '-t', default=0.3, required=False, help=u'Ratio de muestras para test')
def run_pipeline_test(files, ratio, test_r):

    # Read
    #data = read_data(files)
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])

    # RVR preprocessing



    # Split
    train, test = split_train_test(data, test_r)

    # Create pipeline
    pipe = TSFPipeline([('ar', SimpleAR(n_prev=2)),
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


def split_train_test(data, test_ratio):
    train_ratio = 1-test_ratio
    if len(data.shape) == 1:
        train_samples = int(len(data) * train_ratio)
        return data[:train_samples], data[train_samples:]
    else:
        train_samples = int(len(data[0]) * train_ratio)
        return data[:, :train_samples], data[:, train_samples:]


if __name__ == "__main__":
    run_pipeline_test()