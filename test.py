# -*- coding: utf8 -*-

import sys
import pandas as pd
sys.path.append('/Users/migueldiazlozano/Desktop/Ingeniería Informática/TFG/TSF/tsf')
from tsf.pipeline.tsf_pipeline import tsf_pipeline
from tsf.time_series_forescaster import SimpleAR, DinamicWindow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


def test_pipe():
    ar = SimpleAR()
    dw = DinamicWindow()

    sc = StandardScaler()

    # Read
    data = pd.read_csv("data/temp.txt", header=None)
    data = data.values.reshape(1, data.values.shape[0])
    data = data[0]

    train_n = int(len(data) * 0.7)
    train = data[:train_n]
    test = data[train_n:]

    pipe = tsf_pipeline([('ar', SimpleAR()),
                     ('dw', DinamicWindow()),
                     ('lasso', MLPRegressor())])

    pipe.fit(X=[], y=train)
    X, y = pipe.transform(X=[], y=train)

    y_predict = pipe.predict(y=test)
    print pipe.offset_y(test, y_predict)
    print y_predict
    print "MSE:"
    print mean_squared_error(y_predict, pipe.offset_y(test, y_predict))
if __name__ == "__main__":
    test_pipe()