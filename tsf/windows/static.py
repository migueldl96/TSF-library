from .base import TSFBaseTransformer
from sklearn.externals.joblib import Parallel, delayed
import numpy as np


class SimpleAR(TSFBaseTransformer):
    """
    Simple Autorregresive Model using fixed window sizes.

    The SimpleAR transformer takes `n_prev` previous samples for every
    element on the time serie.


    Parameters
    ----------
    n_prev : int, optional, default: 5
        Number of previous samples to consider in the transformation.
    indices : array_like, optional, default: None
        Indexs of series to consider in the transformation (multivariate only). If None,
        all series will be considered.
    n_jobs : int, optional, default: 1
        Number of parallel CPU cores to use.
    horizon : str, optional, default: 1
        Distance for forecasting.
    """
    def __init__(self, n_prev=5, n_jobs=1, indices=None, horizon=1):
        # Init superclass
        super(SimpleAR, self).__init__(indices=indices, n_jobs=n_jobs, horizon=horizon)

        self.n_prev = n_prev

    def run_preprocessing(self, serie):
        partial_X = []
        begin = 0
        end = self.n_prev

        while end <= len(serie) - self.horizon:
            n_samples = serie[begin:end]
            partial_X.append(n_samples)
            begin = begin + 1
            end = end + 1

        return np.array(partial_X)