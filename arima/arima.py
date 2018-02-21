import pandas as pd
import click
import datetime
from statsmodels.tsa.arima_model import *
from sklearn.metrics import mean_squared_error

@click.command()
@click.option('--data', '-d', default=None, required=True, help=u'Característica a usar:'
                                                                u'1 - Temperatura'
                                                                u'2 - ')
@click.option('--n_prev', '-n', default=4, required=False, help=u'Muestras previas')
@click.option('--test', '-t', default=0.3, required=False, help=u'Ratio de muestras para test')
def run_arima(file, n_prev, test):
    def parser(a, b, c, d):
        return datetime.strptime('-'.join([a, b, c, d]), '%Y-%m-%d-%H')

    # Read
    df = pd.read_csv(file, header=None, date_parser=parser, parse_dates={'datetime': [0, 1, 2, 3]}, sep='\t')
    data = df.values[:, 1]
    print data
    quit()

    # Split
    n_data = len(data)
    n_test = int(n_data * test)
    test = data[n_test:]
    train = data[:n_test]

    # Model
    arima = ARIMA(train, (n_prev, 0, 0))

    fitted_model = arima.fit(disp=0)

    predicted = fitted_model.predict(start=0, end=len(data))
    print data
    print predicted
    # MSE
    print mean_squared_error(data, predicted[:-1])


if __name__ == "__main__":
    run_arima()