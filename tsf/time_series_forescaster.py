import numpy as np

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

        # X must be ndarray-type
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # We have to get the previous 'n_prev' samples for every 'y' value
        partial_X = []
        begin = 0
        end = self.n_prev

        while end < len(y):
            n_samples = y[begin:end]
            partial_X.append(n_samples)
            begin = begin + 1
            end = end + 1

        # Only 'y' samples offset by 'n_prev' can be forescasted
        y = y[self.n_prev:]
        y = np.array(y)

        # We already have the data, lets append it to our inputs matrix
        X, y = append_inputs(X, partial_X, y)
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
        self._limit = None

        # Metrics
        self._valid_metrics = ['mean', 'variance']
        if not hasattr(metrics, "__iter__"):
            raise ValueError("'metrics' param should be iterable.")
        else:
            self.metrics = metrics

    def fit(self, X, y=None):

        # User-defined handler?
        if callable(self._handler):
            self._limit = self._handler(y) * self.ratio
        else:
            self._limit, self._handler = self.get_stat_limit(y)

        return self

    def transform(self, X, y=None):
        X = self.fit_transform(X, y=y)
        return X

    def fit_transform(self, X, y=None, **fit_params):

        # User-defined handler?
        if callable(self._handler):
            self._limit = self._handler(y) * self.ratio
        else:
            self._limit, self._handler = self.get_stat_limit(y)

        # Y must be the time serie!
        if y is None:
            raise ValueError("TSF transformers need to receive the time serie data as Y input.\
                                 Please call fit before transforming.")

        # X must be ndarray-type
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Build database for every output
        partial_X = []
        for index, output in enumerate(y[2:]):
            index = index + 2

            # First two previous samples
            pivot = index-2
            samples = y[pivot:index]

            while self._handler(samples) < self._limit and pivot - 1 >= 0:
                pivot = pivot - 1
                samples = y[pivot:index]

            # Once we have the samples, we gather info about them
            samples_info = []
            for metric in self.metrics:
                samples_info.append(self._get_samples_info(samples, metric))
            partial_X.append(samples_info)

        # We begin in the third sample, some stats need at least two samples to work
        y = y[2:]

        # We already have the data, lets append it to our inputs matrix
        X, y = append_inputs(X, partial_X, y)
        return X

    def get_stat_limit(self, y):

        # Define stat handlers
        def variance(samples):
            return np.var(samples)

        if self.stat == 'variance':
            return variance(y) * self.ratio, variance
        else:
            raise ValueError("Invalid stat argument for dinamic window. Please use ['variance'] or own stat function.")

    def _get_samples_info(self, samples, metric):
        # Valid metric?
        if metric not in self._valid_metrics and not callable(metric):
            raise ValueError("Unkown '%s' metric" % metric)

        if callable(metric):
            return metric(samples)
        else:
            return {
                'mean': np.mean(samples),
                'variance': np.var(samples)
            }.get(metric)


class ClassChange(BaseEstimator, TransformerMixin):
    def __init__(self):
        print "Just an init method!"


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
            # If not, we delete firts 'dif' rows from the bigger matrix and from the time serie outputs
            bigger, smaller = (X, X_new) if x_samples > x_new_samples else (X_new, X)
            dif = np.abs(x_new_samples-x_samples)
            bigger = np.delete(bigger, range(0, dif, 1), 0)

            # Now we can append
            X = np.append(bigger, smaller, 1)

            # We cant predict first 'dif' values from 'y'
            y = y[dif:]
    return X, y