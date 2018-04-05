# -*- coding: utf8 -*-

import sys
sys.path.append('../tsf')
sys.path.append('../tsf/pipeline')
sys.path.append('../tsf/grid_search')
reload(sys)
sys.setdefaultencoding('utf-8')
from time_series_forescaster import SimpleAR, DinamicWindow, RangeWindow, ClassChange, TSFBaseTransformer
from tsf_pipeline import TSFPipeline
from tsf_gridsearchcv import TSFGridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import random
import numpy as np
import pandas as pd

from sacred import Experiment


def var_function(samples):
    return np.var(samples)


# RVR umbralizer
def umbralizer(sample):
    if sample < 1000:
        return 2
    elif 1000 < sample < 1990:
        return 1
    else:
        return 0


ex = Experiment()


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


def create_pipe(pipe_steps):
    steps = []
    if pipe_steps['cc']:
        steps.append(("cc", ClassChange()))
    if pipe_steps['dw']:
        steps.append(("dw", DinamicWindow()))
    if pipe_steps['ar']:
        steps.append(("ar", SimpleAR()))
    steps.append(("model", LassoCV()))
    return TSFPipeline(steps)


def get_params(pipe_steps, tsf_config):
    params = []
    TSFBaseTransformer.horizon = tsf_config['horizon']
    if pipe_steps['ar']:
        params.append(tsf_config['ar'])
    if pipe_steps['dw']:
        params.append(tsf_config['dw'])
    if 'model' in pipe_steps.keys():
        params.append(tsf_config['model'])

    return params


@ex.config
def configuration():
    pipe_steps = {
        'ar': True,
        'dw': True,
        'cc': True,
        'model': LassoCV()
    }
    tsf_config = {
        'n_jobs': -1,
        'horizon': 1,
        'ar': {
            'ar__n_prev': [1, 2]
        },
        'dw': {
            'dw__ratio': [0.1]
        },
        'model': {
        }
    }

    pipe = create_pipe(pipe_steps)
    params = get_params(pipe_steps, tsf_config)

    test_ratio = 0.3
    files = ["temp.txt"]
    seed = 0


@ex.named_config
def rvr():
    files = ["RVR.txt", "temp.txt", "humidity.txt", "windDir.txt", "windSpeed.txt", "QNH.txt"]

@ex.automain
def main(files, test_ratio, pipe, params, seed):

    # Read the data
    data = read_data(files)

    # Set the seed
    random.seed(seed)

    # Split
    train, test = split_train_test(data, test_ratio)

    # Cross validation GridSearch
    cv = KFold(n_splits=3, random_state=seed)

    # Create and fit TSFGridSearch
    gs = TSFGridSearchCV(pipe, params, cv=cv)
    gs.fit(X=[], y=train)
    print "Best params: " + str(gs.best_params_)

    # Predict using Pipeline
    predicted_train = gs.predict(train)
    predicted_test = gs.predict(test)

    # MSE
    mse_train = mean_squared_error(pipe.offset_y(train, predicted_train), predicted_train)
    mse_test = mean_squared_error(pipe.offset_y(test, predicted_test), predicted_test)

    print "MSE train: " + str(mse_train)
    print "MSE test: " + str(mse_test)