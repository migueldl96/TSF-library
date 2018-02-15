from sklearn.pipeline import Pipeline


class TSFPipeline(Pipeline):

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

        if self._final_estimator is not None:
            Yt = self._reshape_outputs(Xt, y)
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
        offset = len(real_y) - len(predicted_y)
        return real_y[offset:]

    def _reshape_outputs(self, X, y=None):
        offset = len(y) - X.shape[0]
        return y[offset:]
