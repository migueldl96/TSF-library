from sklearn.pipeline import Pipeline
import numpy as np

class TSFPipeline(Pipeline):
    """
    Pipeline extension to concatenate TSF transformer before a model is trained.
    """
    @property
    def transform(self):
        if self._final_estimator is not None and hasattr(self._final_estimator, 'transform'):
            self._final_estimator.transform
        return self._transform

    def _transform(self, X, y=None):
        Xt = X
        for name, transform in self.steps:
            if transform is not None and hasattr(transform, 'transform'):
                Xt = transform.transform(Xt, y)

        # We have lost outputs from 'y'...
        if y is not None:
            Yt = self._reshape_outputs(Xt, y)

        return Xt, Yt

    def fit_transform(self, X, y=None, **fit_params):
        last_step = self._final_estimator
        Xt, fit_params = self._fit(X, y, **fit_params)
        Yt = self._reshape_outputs(Xt, y)
        if hasattr(last_step, 'fit_transform'):
            return last_step.fit_transform(Xt, Yt, **fit_params)
        elif last_step is None:
            return Xt, Yt
        else:
            return last_step.fit(Xt, Yt, **fit_params).transform(Xt)

    def fit(self, X=[], y=None, **fit_params):
        Xt, fit_params = self._fit(X, y, **fit_params)

        y = self._check_consistent_y(y)

        if self._final_estimator is not None:
            if len(y.shape) == 1:
                Yt = self._reshape_outputs(Xt, y)
            else:
                Yt = self._reshape_outputs(Xt, y[0])

            self._final_estimator.fit(Xt, Yt, **fit_params)

        return self

    def predict(self, y):
        X = []
        for name, transform in self.steps[:-1]:
            if transform is not None:
                X = transform.transform(X=X, y=y)

        return self.steps[-1][-1].predict(X)

    def score(self, y, sample_weight=None):
        Xt = []
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(X=Xt, y=y)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight

        Yt = y[Xt.shape[0]:]
        return self.steps[-1][-1].score(Xt, Yt, **score_params)

    def offset_y(self, real_y, predicted_y):

        y = self._check_consistent_y(real_y)
        y = self._check_consistent_y(predicted_y)

        if len(real_y.shape) == 1:
            offset = len(real_y) - len(predicted_y)
            return real_y[offset:]
        else:
            offset = len(real_y[0]) - len(predicted_y)
            return real_y[0, offset:]

    def _reshape_outputs(self, X, y):

        y = self._check_consistent_y(y)

        if len(y.shape) == 1:
            offset = len(y) - X.shape[0]
            return y[offset:]
        else:
            offset = len(y[0]) - X.shape[0]
            return y[0, offset:]

    def _check_consistent_y(self, y):
        # Y must be ndarray-type
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Y must be 2D
        if len(y.shape) == 1:
            y = np.array([y])

        return y