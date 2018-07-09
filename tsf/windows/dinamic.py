from .base import TSFBaseTransformer
import numpy as np
import warnings


class StatAnalysis(TSFBaseTransformer):
    """
    StatAnalysis Model using variable window sizes.

    The StatAnalysis transformers creates windows limited by a `stat` ratio
    over the total of the time serie. The window samples are summarize in some `metrics`.

    Parameters
    ----------
    stat : str, 'variance' or callable, default: 'variance'
        Statistical measure to limit the window size. If callable,
        it should receive an array of numbers and return a single value.
    ratio : float, default: 0.1
        Ratio over the total time serie `stat` to limit the window size.
    metrics : array_like, 'variance' or 'mean' or callable, default: ['mean', 'variance']
        Array indicating the metrics to use to summarize the window content. Predefined
        metrics are 'mean' and 'variance', but callable receiving an array of samples and
        returning a single number can also be used as element of the array.
    indices : array_like, optional, default: None
        Indices of series to consider in the transformation (multivariate only). If None,
        all series will be considered.
    n_jobs : int, optional, default: 1
        Number of parallel CPU cores to use.
    horizon : str, optional, default: 1
        Distance for forecasting.

    """
    def __init__(self, stat='variance', ratio=0.1, metrics=None, n_jobs=1, indices=None, horizon=1):
        # Init superclass
        super(StatAnalysis, self).__init__(indices=indices, n_jobs=n_jobs, metrics=metrics, horizon=horizon)

        self.stat = stat
        self.ratio = ratio

        # Fit attributes
        if callable(stat):
            self._handler = stat
        else:
            self._handler = None

    def fit(self, X, y=None):
        """
        Set the stat function for time serie analysis.

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
        # User-defined handler?
        if not callable(self._handler):
            self._handler = self._get_stat_handler(y)

        return super(StatAnalysis, self).fit_transform(X, y)

    def run_preprocessing(self, serie):
        if self.ratio >= 1:
            warnings.warn("Ratio param is too big (%d). Consider using a ratio <1" %
                          self.ratio, Warning)

        if self._handler.__name__ == "incremental_variance":
            limit = np.var(serie) * self.ratio
        else:
            limit = self._handler(serie) * self.ratio

        partial_X = []

        for index, output in enumerate(serie[self.horizon:]):
            index = index + 1

            # First previous samples
            pivot = index - 1
            samples = serie[pivot:index]
            if self._handler.__name__ == "incremental_variance":
                n = len(samples)
                previous_var = np.var(samples)
                previous_mean = np.mean(samples)
                while pivot - 1 >= 0 and previous_var < limit:
                    n = n + 1
                    previous_var, previous_mean = self._handler(n, previous_mean, previous_var, serie[pivot - 1])
                    if previous_var < limit:
                        pivot = pivot - 1
                        samples = serie[pivot:index]
            else:
                while pivot - 1 >= 0 and self._handler(serie[pivot - 1:index]) < limit:
                    pivot = pivot - 1
                    samples = serie[pivot:index]

            # Once we have the samples, gather info about them
            samples_info = []
            for metric in self.metrics:
                samples_info.append(_get_samples_info(samples, metric))
            partial_X.append(samples_info)

        return np.array(partial_X)

    def _get_stat_handler(self, y):
        if self.stat == 'variance':
            return incremental_variance
        else:
            raise ValueError("Invalid stat argument for StatAnalysis. Please use ['variance'] or own stat function.")


# Incremental variance stat handler
# Source: http://datagenetics.com/blog/november22017/index.html
def incremental_variance(n_data, previous_mean, previous_var, new_value):
    # We need mean
    def incremental_mean(n, previous, new):
        mean = previous + (new - previous) / float(n)
        return mean

    new_mean = incremental_mean(n_data, previous_mean, new_value)
    previous_sn = previous_var * (n_data-1)
    new_sn = previous_sn + (new_value - previous_mean) * (new_value - new_mean)

    return new_sn/float(n_data), new_mean


class ClassChange(TSFBaseTransformer):
    """
    ClassChange Model using variable window sizes.

    The ClassChange model is a classification and multivariate oriented transformer to create
    variable window sizes based on a class change detection. It creates windows from the
    endogenous serie, and then summarize the content of the windows for every exogenous.
    At least one exogenous serie is required to use ClassChange transformer.

    IMPORTANT: ClassChange use is intended for categorical endogenous time series.

    Parameters
    ----------
    metrics : array_like, 'variance' or 'mean' or callable, default: ['mean', 'variance']
        Array indicating the metrics to use to summarize the windows content. Predefined
        metrics are 'mean' and 'variance', but callable receiving an array of samples and
        returning a single number can also be used as element of the array.
    indices : array_like, optional, default: None
        Indices of series to consider in the transformation (multivariate only). If None,
        all series will be considered.
    n_jobs : int, optional, default: 1
        Number of parallel CPU cores to use.
    horizon : str, optional, default: 1
        Distance for forecasting.

    """
    def __init__(self, metrics=None, n_jobs=1, indices=None, horizon=1):
        # Init superclass
        super(ClassChange, self).__init__(indices=indices, n_jobs=n_jobs, metrics=metrics, horizon=horizon)

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
        y = self.check_consistent_y(y)

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

        y = self.check_consistent_y(y)
        # Consistent params
        X, self.series = self.check_consistent_params(X, self.series)

        # We need al least one exog serie
        if len(y.shape) == 1 or y.shape[0] < 2:
            raise ValueError("ClassChange need to receive one exogenous serie at least. Please use"
                             "an 'y' 2D array with at least 2 rows.")

        # Get exogs series info
        partial_X = self.run_preprocessing()

        X = self.append_inputs(X, partial_X)

        return X

    def run_preprocessing(self):
        partial_X = []

        # Info from every output from all exogs series
        for index, output in enumerate(self.umbralized_serie[self.horizon:]):
            index = index + 1
            output_info = []
            pivot = index - 1
            previous = self.umbralized_serie[pivot]

            # Window size
            while pivot > 0 and previous == self.umbralized_serie[pivot - 1]:
                pivot = pivot - 1
            start = pivot
            end = index

            # Info from exog series
            for exog in self.series:
                samples = exog[start:end]
                for metric in self.metrics:
                    output_info.append(_get_samples_info(samples, metric))

            partial_X.append(output_info)

        return np.array(partial_X)


# Auxiliary function for gathering window values information
def _get_samples_info(samples, metric):
    # Valid metric?
    valid_metrics = ['mean', 'variance']
    if metric not in valid_metrics and not callable(metric):
        raise ValueError("Unkown '%s' metric" % metric)

    if callable(metric):
        return metric(samples)
    else:
        return {
            'mean': np.mean(samples),
            'variance': np.var(samples)
        }.get(metric)