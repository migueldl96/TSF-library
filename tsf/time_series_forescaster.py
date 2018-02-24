import numpy as np
from tsf_tools import _window_maker_delegate, _range_window_delegate
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import LassoCV


class TimeSeriesForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, base_model=LassoCV()):
        self._model = base_model

    def set_params(self, **params):
        for param, value in params.iteritems():
            if param in self.get_params():
                super(TimeSeriesForecaster, self).set_params(**{param: value})
            else:
                self._model.set_params(**{param: value})
        return self

    def fit(self, X, y=None):
        # We must fit with the same number of targets as the inputs matrix
        y = self.get_targets(X, y)

        return self._model.fit(X, y)

    def predict(self, X, y=None):
        return self._model.predict(X)

    def get_targets(self, X, y):
        samples_x = X.shape[0]
        offset_y = y.shape[0] - samples_x
        return y[offset_y:]


class SimpleAR(BaseEstimator, TransformerMixin):
    def __init__(self, n_prev=5):
        self.n_prev = n_prev

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.fit_transform(X, y=y)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        # Y must be the time serie!
        if y is None:
            raise ValueError("TSF transformers need to receive the time serie data as Y input.")

        # Y must be 2D
        if len(y.shape) == 1:
            y = np.array([y])

        # X must be ndarray-type
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # We have to get the previous 'n_prev' samples for every 'y' value
        for serie in y:
            partial_X = _window_maker_delegate(serie, fixed=self.n_prev)

            # Only 'y' samples offset by 'n_prev' can be forescasted
            serie = serie[self.n_prev:]
            serie = np.array(serie)

            # We already have the data, lets append it to our inputs matrix
            X, serie = append_inputs(X, partial_X, serie)

        return X


class DinamicWindow(BaseEstimator, TransformerMixin):
    def __init__(self, stat='variance', ratio=0.1, metrics=['mean', 'variance']):
        self.stat = stat
        self.ratio = ratio

        # Fit attributes
        if callable(stat):
            self._handler = stat
        else:
            self._handler = None

        # Metrics
        self._valid_metrics = ['mean', 'variance']
        if not hasattr(metrics, "__iter__"):
            raise ValueError("'metrics' param should be iterable.")
        else:
            self.metrics = metrics

    def fit(self, X, y=None):

        # User-defined handler?
        if not callable(self._handler):
            self._handler = self._get_stat_handler(y)

        return self

    def transform(self, X, y=None):
        X = self.fit_transform(X, y=y)
        return X

    def fit_transform(self, X, y=None, **fit_params):

        # Y must be the time serie!
        if y is None:
            raise ValueError("TSF transformers need to receive the time serie data as Y input.\
                                 Please call fit before transforming.")
        # Y must be 2D
        if len(y.shape) == 1:
            y = np.array([y])

        # X must be ndarray-type
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # User-defined handler?
        if not callable(self._handler):
            self._handler = self._get_stat_handler(y)

        # Build database for every output
        for serie in y:

            partial_X = _window_maker_delegate(serie, handler=self._handler, metrics=self.metrics)

            # We begin in the third sample, some stats need at least two samples to work
            serie = y[2:]

            # We already have the data, lets append it to our inputs matrix
            X, serie = append_inputs(X, partial_X, serie)

        return X

    def _get_stat_handler(self, y):
        # Defined stat handlers
        def variance(samples):
            return np.var(samples)

        if self.stat == 'variance':
            return variance
        else:
            raise ValueError("Invalid stat argument for dinamic window. Please use ['variance'] or own stat function.")


class RangeWindow(BaseEstimator, TransformerMixin):
    def __init__(self, metrics=['mean', 'variance']):

        # Metrics
        if not hasattr(metrics, "__iter__"):
            raise ValueError("'metrics' param should be iterable.")
        else:
            self._metrics = metrics
        self._metrics = metrics

        # Fit attributes
        self._dev = None

    def fit(self, X, y=None):
        # Deviation for range
        self._dev = (np.max(y) - np.min(y)) / np.mean(y)
        return self

    def transform(self, X, y=None):
        X = self.fit_transform(X, y)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        # Y must be the time serie!
        if y is None:
            raise ValueError("TSF transformers need to receive the time serie data as Y input.\
                                 Please call fit before transforming.")

        # X must be ndarray-type
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Build database for every output
        for serie in y:
            # Deviation for range
            self._dev = (np.max(serie) - np.min(serie)) / np.var(serie)

            partial_X = _range_window_delegate(serie, self._dev, self._metrics)

            # We already have the data, lets append it to our inputs matrix
            X, serie = append_inputs(X, partial_X, serie)

        return X

    def _in_range(self, value, allowed_range):
        return allowed_range.min() < value < allowed_range.max()


def append_inputs(X, X_new, y):
    # We must append an ndarray-type
    if not isinstance(X_new, np.ndarray):
        X_new = np.array(X_new)

    # Is X empty?
    if X.size == 0:
        X = X_new
    else:
        # They must have same number of samples
        x_samples = X.shape[0]
        x_new_samples = X_new.shape[0]

        if x_samples == x_new_samples:
            X = np.append(X, X_new, axis=1)
        else:
            # If not, we delete first 'dif' rows from the bigger matrix and from the time serie outputs
            bigger, smaller = (X, X_new) if x_samples > x_new_samples else (X_new, X)
            dif = np.abs(x_new_samples-x_samples)
            bigger = np.delete(bigger, range(0, dif, 1), 0)

            # Now we can append
            X = np.append(bigger, smaller, 1)

            # We cant predict first 'dif' values from 'y'
            y = y[dif:]
    return X, y