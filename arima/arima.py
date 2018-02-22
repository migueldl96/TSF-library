# -*- coding: utf8 -*-

import pandas as pd
import click
import datetime
from statsmodels.tsa.arima_model import *
from sklearn.metrics import mean_squared_error

@click.command()
@click.option('--data', '-d', default=1, required=True, help=u'Característica a usar:'
                                                                u'1 - Velocidad del viento'
                                                                u'2 - Dirección del viento'
                                                                u'3 - Temperatura'
                                                                u'4 - Humedad'
                                                                u'5 - QNH')
@click.option('--n_prev', '-n', default=4, required=False, help=u'Muestras previas')
@click.option('--test', '-t', default=0.3, required=False, help=u'Ratio de muestras para test')
def run_arima(data, n_prev, test):
    def parser(a, b, c, d):
        return datetime.strptime('-'.join([a, b, c, d]), '%Y-%m-%d-%H')

    # Read
    df = pd.read_csv('../data/niebla.txt', header=None, date_parser=parser, parse_dates={'datetime': [0, 1, 2, 3]}, sep='\t')
    data = df.values[:, data]

    # Split
    n_data = len(data)
    n_test = int(n_data * test)
    test, train = data[n_test:], data[:n_test]

    # Model
    train_data = [x for x in train]
    predictions = list()

    # walk-forward validation
    for t in range(len(test)):
        # fit model
        model = ARIMA(train_data, order=(n_prev,1,0))
        model_fit = model.fit(disp=False)
        # one step forecast
        yhat = model_fit.forecast()[0]
        # store forecast and ob
        predictions.append(yhat)
        train_data.append(test[t])

    # evaluate forecasts
    mse = mean_squared_error(test, predictions)

    print('Test MSE: %.3f' % mse)


if __name__ == "__main__":
    run_arima()