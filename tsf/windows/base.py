from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import numpy as np
from sklearn.externals.joblib import Parallel, delayed


# Unwrapper allows parallelizing
def unwrap_self(window, *kwarg):
    return window.run_preprocessing(*kwarg)


class TSFBaseTransformer(BaseEstimator, TransformerMixin, object):
    """
    Base class for every transform in TSF library. It contains
    basic attributes and methods needed in all the transformations.

    TSFBaseTransformers objects shouldn't be created, as it is only
    used as superclass for all transformers.
    """

    # Static horizon for all transformers
    horizon = None

    def __init__(self, indices=None, n_jobs=1, metrics=None, horizon=1):
        """
        Base constructor for basic parameters initialization.
        """
        self.n_jobs = n_jobs
        self.set_horizon(horizon)

        # Involved serie indices
        if indices is not None and not hasattr(indices, "__iter__"):
            raise ValueError("'indices' parameter should be iterable.")
        self.indices = indices
        self.series = None

        # Metrics
        if metrics is None:
            self.metrics = ['mean', 'variance']
        else:
            if not hasattr(metrics, "__iter__"):
                raise ValueError("'metrics' param should be iterable.")
            self.metrics = metrics

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
        return self.fit_transform(X, y)

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
        partial_X = Parallel(n_jobs=self.n_jobs)(delayed(unwrap_self)(self, serie) for serie in self.series)

        # We already have the data, lets append it to our inputs matrix
        X = self.append_inputs(X, partial_X)

        return X

    def run_preprocessing(self, serie):
        """
        Run the preprocessing algorithm and returns the inputs matrix created for a single time serie.

        Parameters
        ----------
        serie : array-like
            Time serie to be preprocessed.

        Returns
        -------
        X : array-like
            Inputs matrix created by the algorithm.
        """
        pass

    def set_involved_series(self, y):
        """
        Select the series to consider in the transformation if `indices` is not None.

        Parameters
        ----------
        y : array-like
            Time series array. If matrix, first row is consider endogenous and the rest as exogenous.
        """
        y = self.check_consistent_y(y)

        if self.indices is not None:
            self.series = []
            for index in self.indices:
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
            The new forecasting horizon value.
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
        Calculates and return corresponding outputs given a inputs matrix and a time serie.

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