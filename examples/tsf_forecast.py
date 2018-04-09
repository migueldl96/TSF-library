# -*- coding: utf8 -*-

import sys
sys.path.append('../tsf')
sys.path.append('../tsf/pipeline')
sys.path.append('../tsf/grid_search')
reload(sys)
sys.setdefaultencoding('utf-8')
from tsf_windows import SimpleAR, DinamicWindow, RangeWindow, ClassChange, TSFBaseTransformer
from tsf_pipeline import TSFPipeline
from tsf_gridsearchcv import TSFGridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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


def get_estimator(estimator):
    if estimator == 0:
        return LassoCV()
    if estimator == 1:
        return MLPRegressor()
    if estimator == 2:
        return MLPClassifier()


def create_pipe(pipe_steps, estimator):
    steps = []
    if pipe_steps['cc']:
        steps.append(("cc", ClassChange()))
    if pipe_steps['dw']:
        steps.append(("dw", DinamicWindow()))
    if pipe_steps['ar']:
        steps.append(("ar", SimpleAR()))
    if pipe_steps['model']:
        steps.append(("model", get_estimator(estimator)))
    return TSFPipeline(steps)


def get_params(pipe_steps, tsf_config, model_config):
    params = []
    TSFBaseTransformer.horizon = tsf_config['horizon']
    if pipe_steps['ar']:
        params.append(tsf_config['ar'])
    if pipe_steps['dw']:
        params.append(tsf_config['dw'])
    if pipe_steps['cc']:
        params.append(tsf_config['cc'])
    if pipe_steps['model']:
        params.append(model_config['params'])

    return params


@ex.config
def configuration():
    seed = 0

    # ratio of samples for testing
    test_ratio = 0.3

    # files where time series are located
    files = ["temp.txt"]

    # steps of the model
    pipe_steps = {
        'ar': True,     # Standard autorregresive model using fixed window
        'dw': True,     # Dinamic Window based on stat limit
        'cc': False,    # Dinamic Window based on class change (classification oriented)
        'model': True   # Final estimator used for forecasting
    }

    # parameters of the windows model
    tsf_config = {
        'n_jobs': -1,   # Number of parallel process
        'horizon': 1,   # Forecast distance
        'ar': {         # Standar autorregresive parameters
            'ar__n_prev': [1, 2]                        # Number of previous samples to use
        },
        'dw': {         # Dinamic Window based on stat limit parameters
            'dw__stat': ['variance'],                   # Stat to calculate window size
            'dw__metrics': [['mean', 'variance']],      # Stats to resume information of window
            'dw__ratio': [0.1],                         # Stat ratio to limit window size
            'dw__indexs': [None]                        # Indexs of series to be used
        },
        'cc': {         # Dinamic window based on class change parameters
            'cc__metrics': [['mean', 'variance']],      # Stats to resume information of window
            'cc__indexs': [None],                       # Indexs of series to be used
            'cc__umbralizer': [None]
        },
    }

    # parameters of the estimator model
    model_config = {
        'type': 'regression',
        'estimator': 0,
        'params': {

        }
    }


@ex.named_config
def rvr():
    files = ["RVR.txt", "temp.txt", "humidity.txt", "windDir.txt", "windSpeed.txt", "QNH.txt"]


@ex.automain
def main(files, test_ratio, pipe_steps, tsf_config, model_config, seed):

    # Read the data
    data = read_data(files)

    # Set the seed
    random.seed(seed)

    # Umbralizer
    if model_config['type'] == "classification":
        data[0] = map(umbralizer, data[0])

    # Create pipe and set the config
    pipe = create_pipe(pipe_steps, model_config['estimator'])
    params = get_params(pipe_steps, tsf_config, model_config)

    # Split
    train, test = split_train_test(data, test_ratio)

    # Create and fit TSFGridSearch
    gs = TSFGridSearchCV(pipe, params)
    gs.fit(X=[], y=train)
    print "Best params: " + str(gs.best_params_)

    # Predict using Pipeline
    predicted_train = gs.predict(train)
    predicted_test = gs.predict(test)

    # MSE
    mse_train = mean_squared_error(pipe.offset_y(train, predicted_train), predicted_train)
    mse_test = mean_squared_error(pipe.offset_y(test, predicted_test), predicted_test)

    if model_config['type'] == 'regression':
        print "MSE train: " + str(mse_train)
        print "MSE test: " + str(mse_test)
    elif model_config['type'] == 'classification':
        ccr_train = accuracy_score(pipe.offset_y(train, predicted_train), predicted_train)
        ccr_test = accuracy_score(pipe.offset_y(test, predicted_test), predicted_test)
        print "CCR train: " + str(ccr_train)
        print "CCR test: " + str(ccr_test)
