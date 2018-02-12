from sklearn.pipeline import Pipeline


class tsf_pipeline(Pipeline):

    @property
    def transform(self):
        """Apply transforms, and transform with the final estimator

        This also works where final estimator is ``None``: all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
        """
        # _final_estimator is None or has transform, otherwise attribute error
        # XXX: Handling the None case means we can't use if_delegate_has_method
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
        """Fit the model and transform with the final estimator

        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples
        """
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
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        Xt, fit_params = self._fit(X, y, **fit_params)

        if self._final_estimator is not None:
            Yt = self._reshape_outputs(Xt, y)
            self._final_estimator.fit(Xt, Yt, **fit_params)

        return self

    def predict(self, y):
        """Apply transforms to the data, and predict with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_pred : array-like
        """
        X = []
        for name, transform in self.steps[:-1]:
            if transform is not None:
                X = transform.transform(X=X, y=y)

        return self.steps[-1][-1].predict(X)

    def _reshape_outputs(self, X, y=None):
        return y[X.shape[0]:]
