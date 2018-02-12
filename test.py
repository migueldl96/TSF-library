# -*- coding: utf8 -*-

import sys
sys.path.append('/Users/migueldiazlozano/Desktop/Ingeniería Informática/TFG/TSF/tsf')
from tsf.pipeline.tsf_pipeline import tsf_pipeline
from tsf.time_series_forescaster import SimpleAR, DinamicWindow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler



def test_pipe():
    ar = SimpleAR()
    dw = DinamicWindow()

    sc = StandardScaler()

    train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    test  = range(10, 20, 1)
    print()

    pipe = tsf_pipeline([('ar', SimpleAR()),
                     ('dw', DinamicWindow()),
                     ('lasso', LassoCV())])
    param_grid = [
        {
            'n_prev': [1, 2],
        },
        {
            'ratio': [0.1, 0.2]
        }
    ]
    grid =  GridSearchCV(pipe, param_grid)
    grid.fit(X=[], y=train)

    quit()

    print model

if __name__ == "__main__":
    test_pipe()