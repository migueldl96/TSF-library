# -*- coding: utf8 -*-

import sys
sys.path.append('../tsf')
sys.path.append('../tsf/pipeline')
sys.path.append('../tsf/grid_search')
from time_series_forescaster import SimpleAR, DinamicWindow, RangeWindow, ClassChange
from tsf_pipeline import TSFPipeline
from tsf_gridsearchcv import TSFGridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import click
import numpy as np
import pandas as pd


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
@click.option('--n_jobs', '-j', default=-1, required=False, help=u'Número de procesos en paralelo')
def run_pipeline_test(files, ratio, test_r, n_jobs):

    # Read
    data = read_data(files)
    #data = np.array([[1, 2, 43, 4, 5, 6, 7, 8, 9, 10, 44, 65, 67], [11, -2, 13, 14, 15, 16, 99, 18, 19, 20, 44, 65, 67],
    #                [-121, 22, 23, 24, 15, 26, 27, 28, 29, 30, 44, 65, 67]])

    # Split
    train, test = split_train_test(data, test_r)

    # Create pipeline
    pipe = TSFPipeline([('dw', DinamicWindow(ratio=ratio, stat='variance', n_jobs=n_jobs, indexs=None)),
                        ('regressor', LassoCV(random_state=0, n_jobs=n_jobs))])

    # Param grid
    params = [
        {
            'dw__ratio': [0.3, 0.2]
        },
        {
            'random_state': [2, 0, 1]
        }
    ]

    # Cross validation GridSearch
    cv = KFold(n_splits=3, random_state=0)

    # Create and fit TSFGridSearch
    gs = TSFGridSearchCV(pipe, params, cv=cv)
    gs.fit(X=[], y=data)

    # Predict using Pipeline
    predicted_train = gs.predict(train)
    predicted_test = gs.predict(test)

    # MSE
    mse_train = mean_squared_error(pipe.offset_y(train, predicted_train), predicted_train)
    mse_test = mean_squared_error(pipe.offset_y(test, predicted_test), predicted_test)

    print "MSE train: " + str(mse_train)
    print "MSE test: " + str(mse_test)


def read_data(files):
    data = []

    if not files:
        raise ValueError("There is no data. Please use -f to select a file to read.")

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