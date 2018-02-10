from tsf.time_series_forescaster import SimpleAR, TimeSeriesForecaster, DinamicWindow
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV


def main():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
    tsf = TimeSeriesForecaster()

    pipe = Pipeline([('ar', SimpleAR()),
                     ('dw', DinamicWindow()),
                     ('lasso', tsf)])

    pipe.fit(X=[], y=data)


if __name__ == "__main__":
    main()