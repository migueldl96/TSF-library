import numpy as np
import warnings
from tsf_tools import _fixed_window_delegate, _range_window_delegate, _dinamic_window_delegate, _classchange_window_delegate, incremental_variance
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV


class TSFBaseTransformer(BaseEstimator, TransformerMixin):

    horizon = None

    def __init__(self, indexs=None, n_jobs=-1, metrics=None, horizon=None):
        self.n_jobs = n_jobs
        self.set_horizon(horizon)

        # Involved serie indexs
        if indexs is not None and not hasattr(indexs, "__iter__"):
            raise ValueError("'indexs' parameter should be iterable.")
        self.indexs = indexs
        self.series = None

        # Metrics
        if metrics is None:
            self._metrics = ['mean', 'variance']
        else:
            if not hasattr(metrics, "__iter__"):
                raise ValueError("'metrics' param should be iterable.")
            self._metrics = metrics

    def set_involved_series(self, y):
        y = self.check_consistent_y(y)

        if self.indexs is not None:
            self.series = []
            for index in self.indexs:
                try:
                    self.series.append(y[index])
                except IndexError:
                    warnings.warn("'%d' index out of 'y' range. Max: '%d'. Ignoring this index..."
                                  % (index, y.shape[0]-1))
        else:
            self.series = y

    def set_horizon(self, new_horizon):
        if new_horizon is not None:
            if TSFBaseTransformer.horizon is None:
                TSFBaseTransformer.horizon = new_horizon

            if TSFBaseTransformer.horizon is not None and TSFBaseTransformer.horizon != new_horizon:
                warnings.warn("Different horizon values in TSF Transformers. Swapping from '%d' to '%d'" %
                              (TSFBaseTransformer.horizon, new_horizon), Warning)
                TSFBaseTransformer.horizon = new_horizon

    def check_consistent_y(self, y):
        # Y must be the time serie!
        if y is None:
            raise ValueError("TSF transformers need to receive the time serie data as Y input.")

        # Y must be ndarray-type
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Y must be 2D
        if len(y.shape) == 1:
            y = np.array([y])

        return y

    def check_consistent_X(self, X):
        # X must be ndarray-type
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        return X

    def check_consistent_params(self, X, y):
        y = self.check_consistent_y(y)
        X = self.check_consistent_X(X)

        return X, y

    def append_inputs(self, X, X_new):
        # We must append an ndarray-type
        if not isinstance(X_new, np.ndarray):
            X_new = np.array(X_new)

        # We need a 2D array
        if len(X_new.shape) == 3:
            X_new = X_new.transpose((1, 0, 2)).reshape((X_new.shape[1], X_new.shape[0]*X_new.shape[2]))

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
                dif = np.abs(x_new_samples-x_samples)
                if x_samples > x_new_samples:
                    X = np.delete(X, range(0, dif, 1), 0)
                else:
                    X_new = np.delete(X_new, range(0, dif, 1), 0)

                # Now we can append
                X = np.append(X, X_new, 1)

        return X


class SimpleAR(TSFBaseTransformer):
    def __init__(self, n_prev=5, n_jobs=-1, indexs=None, horizon=None):
        # Init superclass
        super(SimpleAR, self).__init__(indexs=indexs, n_jobs=n_jobs, horizon=horizon)

        self.n_prev = n_prev

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.fit_transform(X, y=y)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        # Involved series
        self.set_involved_series(y)

        # Consistent data
        X, y = self.check_consistent_params(X, self.series)

        # Build database for every output
        partial_X = Parallel(n_jobs=self.n_jobs)(
            delayed(_fixed_window_delegate)(serie, n_prev=self.n_prev, horizon=TSFBaseTransformer.horizon)
            for serie in self.series)

        # We already have the data, lets append it to our inputs matrix
        X = self.append_inputs(X, partial_X)

        return X


class DinamicWindow(TSFBaseTransformer):
    def __init__(self, stat='variance', ratio=0.1, metrics=None, n_jobs=-1, indexs=None, horizon=None):
        # Init superclass
        super(DinamicWindow, self).__init__(indexs=indexs, n_jobs=n_jobs, metrics=metrics, horizon=horizon)

        self.stat = stat
        self.ratio = ratio

        # Fit attributes
        if callable(stat):
            self._handler = stat
        else:
            self._handler = None

    def fit(self, X, y=None):
        # User-defined handler?
        if not callable(self._handler):
            self._handler = self._get_stat_handler(y)

        return self

    def transform(self, X, y=None):
        X = self.fit_transform(X, y=y)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        # Involved series
        self.set_involved_series(y)

        # Consistent params
        X, y = self.check_consistent_params(X, self.series)

        # User-defined handler?
        if not callable(self._handler):
            self._handler = self._get_stat_handler(y)

        # Build database for every output
        partial_X = Parallel(n_jobs=self.n_jobs)(
            delayed(_dinamic_window_delegate)(serie, handler=self._handler, metrics=self._metrics, ratio=self.ratio,
                                              horizon=TSFBaseTransformer.horizon) for serie in self.series)

        # We already have the data, lets append it to our inputs matrix
        X = self.append_inputs(X, partial_X)

        return X

    def _get_stat_handler(self, y):
        if self.stat == 'variance':
            return incremental_variance
        else:
            raise ValueError("Invalid stat argument for dinamic window. Please use ['variance'] or own stat function.")


class RangeWindow(TSFBaseTransformer):
    def __init__(self, metrics=None, n_jobs=-1, indexs=None, horizon=None):
        # Init superclass
        super(RangeWindow, self).__init__(indexs=indexs, n_jobs=n_jobs, metrics=metrics, horizon=horizon)

        # Fit attributes
        self._dev = None

    def fit(self, X, y=None):
        # Deviation for each serie
        self._dev = (np.max(y, axis=1) - np.min(y, axis=1)) / np.mean(y, axis=1)
        return self

    def transform(self, X, y=None):
        X = self.fit_transform(X, y)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        # Involved series
        self.set_involved_series(y)

        # Consistent params
        X, y = self.check_consistent_params(X, self.series)

        # Deviation for each serie
        self._dev = (np.max(self.series, axis=1) - np.min(self.series, axis=1)) / np.mean(self.series, axis=1)

        # Build database for every output
        partial_X = Parallel(n_jobs=self.n_jobs)(
            delayed(_range_window_delegate)(serie, self._dev[index], self._metrics, horizon=TSFBaseTransformer.horizon)
            for index, serie
            in enumerate(self.series))

        X = self.append_inputs(X, partial_X)

        return X

    def _in_range(self, value, allowed_range):
        return allowed_range.min() < value < allowed_range.max()


class ClassChange(TSFBaseTransformer):
    def __init__(self, umbralizer=None, metrics=None, n_jobs=-1, indexs=None, horizon=1):
        # Init superclass
        super(ClassChange, self).__init__(indexs=indexs, n_jobs=n_jobs, metrics=metrics, horizon=horizon)

        self.umbralizer = umbralizer
        self.umbralized_serie = None

    def fit(self, X, y=None):
        if self.umbralizer is not None:
            self.umbralized_serie = map(self.umbralizer, y[0])
        else:
            self.umbralized_serie = y[0]

    def transform(self, X, y=None):
        # We need al least one exog serie
        if len(y.shape) == 1 or y.shape[0] < 2:
            raise ValueError("ClassChange need tor receive one exogenous serie at least. Please"
                             "an 'y' 2D array with at least 2 rows.")
        Xt = self.fit_transform(X, y)

        return Xt

    def fit_transform(self, X, y=None, **fit_params):
        # Involved series
        self.set_involved_series(y)

        # Consistent params
        X, y = self.check_consistent_params(X, self.series)

        # We need al least one exog serie
        if len(y.shape) == 1 or y.shape[0] < 2:
            raise ValueError("ClassChange need to receive one exogenous serie at least. Please use"
                             "an 'y' 2D array with at least 2 rows.")

        # Umbralize endog
        if self.umbralizer is not None:
            self.umbralized_serie = map(self.umbralizer, y[0])
        else:
            self.umbralized_serie = y[0]

        # Get exogs series info
        partial_X = _classchange_window_delegate(self.umbralized_serie, self.series[1:], self._metrics,
                                                 TSFBaseTransformer.horizon)

        X = self.append_inputs(X, partial_X)
        return X