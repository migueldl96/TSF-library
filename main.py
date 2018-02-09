from tsf.time_series_forescaster import SimpleAR, TimeSeriesForecaster, DinamicWindow
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
def main():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    pipe = Pipeline(steps=[('ar', SimpleAR(n_prev=2)),
                           ('lasso', LassoCV())])
    pipe.fit(X=[],y=data)


if __name__ == "__main__":
    main()