import numpy as np
import warnings
from tsf_tools import _fixed_window_delegate, _range_window_delegate, _dinamic_window_delegate, _classchange_window_delegate, incremental_variance
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin


class TSFBaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for every transform in TSF library. It contains
    basic attributes and methods needed in all the transformations.

    TSFBaseTransformers objects shouldn't be created, as it is only
    used as superclass for all transformers.
    """

    # Static horizon for all transformers
    horizon = None

    def __init__(self, indexs=None, n_jobs=1, metrics=None, horizon=1):
        """
        Base constructor for basic parameters initialization.
        """
        self.n_jobs = n_jobs
        self.set_horizon(horizon)

        # Involved serie indexs
        if indexs is not None and not hasattr(indexs, "__iter__"):
            raise ValueError("'indexs' parameter should be iterable.")
        self.indexs = indexs
        self.series = None

        # Metrics
        if metrics is None:
            self.metrics = ['mean', 'variance']
        else:
            if not hasattr(metrics, "__iter__"):
                raise ValueError("'metrics' param should be iterable.")
            self.metrics = metrics

    def set_involved_series(self, y):
        """
        Select the series to consider in the transformation if `indexs` is not None.

        Parameters
        ----------
        y : array-like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.
        """
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
        """
        Set the horizon value.

        Parameters
        ----------
        new_horizon : int
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.
        """
        if new_horizon is not None:
            if TSFBaseTransformer.horizon is None:
                TSFBaseTransformer.horizon = new_horizon

            if TSFBaseTransformer.horizon is not None and TSFBaseTransformer.horizon != new_horizon:
                warnings.warn("Different horizon values in TSF Transformers. Swapping from '%d' to '%d'" %
                              (TSFBaseTransformer.horizon, new_horizon), Warning)
                TSFBaseTransformer.horizon = new_horizon

    def check_consistent_y(self, y):
        """
        Check `y` parameter for all transformers and adapt it if needed.

        Parameters
        ----------
        y : array_like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.

        Returns
        -------
        y : array_like
            Time series array checked. If matrix, first row is consider endogenous and the rest as exogenous.
        """
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
        """
        Check `X` parameter for all transformers and adapt it if needed.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.

        Returns
        -------
        X : array-like
            Previous data before transformation is appended to inputs matrix checked.
        """
        # X must be ndarray-type
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        return X

    def check_consistent_params(self, X, y):
        """
        Check `X` and `y` parameters for all transformers and adapt it if needed.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.
        y : array_like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.

        Returns
        -------
        X : array-like
            Previous data before transformation is appended to inputs matrix checked.
        y : array_like
            Time series array checked. If matrix, first row is consider endogenous and the rest as exogenous.
        """
        y = self.check_consistent_y(y)
        X = self.check_consistent_X(X)

        return X, y

    def append_inputs(self, X, X_new):
        """
        Append `X_new` to `X` to the right resizing `X` if needed.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.
        X_new : array-like
            New data resulting of the transformation.

        Returns
        -------
        X : array-like
            Array containing the previous and the new data. It may have less rows than previous array
            as some transformers needs previous samples to transform, so the first samples of the series
            are deleted.

        """
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

    def offset_y(self, X, y):
        """
        Calculates and return corresponding outputs given a inputs matrix.

        Parameters
        ----------
        X : array-like
            Inputs matrix
        y : array-like
           Time serie(s)

        Returns
        -------
        y_new : array-like
            Endogenous time serie offset to correspond to inputs matrix X.

        """
        X, y = self.check_consistent_params(X, y)
        if len(y.shape) == 1:
            offset = len(y) - X.shape[0]
            return y[offset:]
        else:
            offset = len(y[0]) - X.shape[0]
            return y[0, offset:]


class SimpleAR(TSFBaseTransformer):
    """
    Simple Autorregresive Model using fixed window sizes.

    The SimpleAR transformer takes `n_prev` previous samples for every
    element on the time serie.


    Parameters
    ----------
    n_prev : int, optional, default: 5
        Number of previous samples to consider in the transformation.
    indexs : array_like, optional, default: None
        Indexs of series to consider in the transformation (multivariate only). If None,
        all series will be considered.
    n_jobs : int, optional, default: 1
        Number of parallel CPU cores to use.
    horizon : str, optional, default: 1
        Distance for forecasting.
    """
    def __init__(self, n_prev=5, n_jobs=1, indexs=None, horizon=1):
        # Init superclass
        super(SimpleAR, self).__init__(indexs=indexs, n_jobs=n_jobs, horizon=horizon)

        self.n_prev = n_prev

    def fit(self, X, y=None):
        """
        Just a compatibility method. It does nothing.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.
        y : array-like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.

        Returns
        -------
        self : object
        """
        return self

    def transform(self, X, y=None):
        """
        Apply the transformation.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.
        y : array-like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.

        Returns
        -------
        X : array-like
            Previous data with the transformation appended for each sample of the time serie.
        """
        X = self.fit_transform(X, y=y)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """
        Apply the transformation.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.
        y : array-like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.

        Returns
        -------
        X : array-like
            Previous data with the transformation appended for each sample of the time serie.
        """
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
    """
    DinamicWindow Model using variable window sizes.

    The DinamicWindow transformers creates windows limited by a `stat` ratio
    over the total of the time serie. This windows is summarize in some `metrics`.

    Parameters
    ----------
    stat : str, 'variance' or callable, default: 'variance'
        Statistical measure to limit the window size. If callable,
        it should receive an array of numbers and return a single value.
    ratio : float, default: 0.1
        Ratio over the total time serie `stat` to limit the window size.
    metrics : array_like, 'variance' or 'mean' or callable, default: ['mean', 'variance']
        Array indicating the metrics to use to summarize the windows content. Predefined
        metrics are 'mean' and 'variance', but callable receiving an array of samples and
        returning a single number can also be used as element of the array.
    indexs : array_like, optional, default: None
        Indexs of series to consider in the transformation (multivariate only). If None,
        all series will be considered.
    n_jobs : int, optional, default: 1
        Number of parallel CPU cores to use.
    horizon : str, optional, default: 1
        Distance for forecasting.

    """
    def __init__(self, stat='variance', ratio=0.1, metrics=None, n_jobs=1, indexs=None, horizon=1):
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
        """
        Set the window size limit.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.
        y : array-like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.

        Returns
        -------
        self : object
        """
        # User-defined handler?
        if not callable(self._handler):
            self._handler = self._get_stat_handler(y)

        return self

    def transform(self, X, y=None):
        """
        Apply the transformation.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.
        y : array-like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.

        Returns
        -------
        X : array-like
            Previous data with the transformation appended for each sample of the time serie.
        """
        X = self.fit_transform(X, y=y)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """
        Apply the transformation.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.
        y : array-like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.

        Returns
        -------
        X : array-like
            Previous data with the transformation appended for each sample of the time serie.
        """
        # Involved series
        self.set_involved_series(y)

        # Consistent params
        X, y = self.check_consistent_params(X, self.series)

        # User-defined handler?
        if not callable(self._handler):
            self._handler = self._get_stat_handler(y)

        # Build database for every output
        partial_X = Parallel(n_jobs=self.n_jobs)(
            delayed(_dinamic_window_delegate)(serie, handler=self._handler, metrics=self.metrics, ratio=self.ratio,
                                              horizon=TSFBaseTransformer.horizon) for serie in self.series)

        # We already have the data, lets append it to our inputs matrix
        X = self.append_inputs(X, partial_X)

        return X

    def _get_stat_handler(self, y):
        if self.stat == 'variance':
            return incremental_variance
        else:
            raise ValueError("Invalid stat argument for dinamic window. Please use ['variance'] or own stat function.")


class ClassChange(TSFBaseTransformer):
    """
    ClassChange Model using variable window sizes.

    The ClassChange model is a classification and multivariate oriented transformer to create
    variable window sizes based on a class change detection. It creates windows from the
    endogenous serie, and then summarize the content of the windows for every exogenous.
    At least one exogenous serie is required to use ClassChange transformer.

    Parameters
    ----------
    umbralizer : func, callable, default: None
        If endogenous serie is continuous, `umbralizer` should be use to umbralize
        the serie so it can be use in ClassChange. None value assume that endogenous
        serie is already umbralized.
    metrics : array_like, 'variance' or 'mean' or callable, default: ['mean', 'variance']
        Array indicating the metrics to use to summarize the windows content. Predefined
        metrics are 'mean' and 'variance', but callable receiving an array of samples and
        returning a single number can also be used as element of the array.
    indexs : array_like, optional, default: None
        Indexs of series to consider in the transformation (multivariate only). If None,
        all series will be considered.
    n_jobs : int, optional, default: 1
        Number of parallel CPU cores to use.
    horizon : str, optional, default: 1
        Distance for forecasting.

    """
    def __init__(self, metrics=None, n_jobs=1, indexs=None, horizon=1):
        # Init superclass
        super(ClassChange, self).__init__(indexs=indexs, n_jobs=n_jobs, metrics=metrics, horizon=horizon)

        self.umbralized_serie = None


    def fit(self, X, y=None):
        """
        Umbralize serie if needed.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.
        y : array-like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.

        Returns
        -------
        self : object
        """
        # Involved series
        self.set_involved_series(y)

        self.umbralized_serie = self.series[0]
        self.series = np.delete(self.series, 0, axis=0)

        return self

    def transform(self, X, y=None):
        """
        Apply the transformation.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.
        y : array-like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.

        Returns
        -------
        X : array-like
            Previous data with the transformation appended for each sample of the time serie.
        """
        # We need al least one exog serie
        if len(y.shape) == 1 or y.shape[0] < 2:
            raise ValueError("ClassChange need tor receive one exogenous serie at least. Please use"
                             "an 'y' 2D array with at least 2 rows.")
        X = self.fit_transform(X, y)

        return X

    def fit_transform(self, X, y=None, **fit_params):
        """
        Apply the transformation.

        Parameters
        ----------
        X : array-like
            Previous data before transformation is appended to inputs matrix.
        y : array-like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.

        Returns
        -------
        X : array-like
            Previous data with the transformation appended for each sample of the time serie.
        """
        # Involved series
        self.set_involved_series(y)

        self.umbralized_serie = self.series[0]
        self.series = np.delete(self.series, 0, axis=0)

        # Consistent params
        X, self.series = self.check_consistent_params(X, self.series)

        # We need al least one exog serie
        if len(y.shape) == 1 or y.shape[0] < 2:
            raise ValueError("ClassChange need to receive one exogenous serie at least. Please use"
                             "an 'y' 2D array with at least 2 rows.")

        # Get exogs series info
        partial_X = _classchange_window_delegate(self.umbralized_serie, self.series, self.metrics,
                                                 TSFBaseTransformer.horizon)

        X = self.append_inputs(X, partial_X)
        return X
