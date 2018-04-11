from sklearn.model_selection import GridSearchCV
from ..pipeline import TSFPipeline
from itertools import product
from sklearn.base import clone


class TSFGridSearch(GridSearchCV):
    def fit(self, X, y=None, groups=None, **fit_params):

        # Partial best score
        best_score = None

        # Partial best params. GridSearch.fit set self.best_params_ with final_estimator best params,
        # this variable is intended to save global best params (transformer + final_estimator)
        best_partial_params = None

        # Pipeline global estimator. GridSearch.fit will iterate over the final_estimator,
        # we need to save global estimator for final setting.
        pipe_estimator = self.estimator

        # Transformers data
        trans_params = self.param_grid[:-1]
        trans_list = self.estimator.steps[:-1]
        trans_list.append(('', None))
        trans_pipe = TSFPipeline(trans_list)

        # Final estimator params
        final_estim_params = self.param_grid[-1]
        final_estimator_name, final_estimator = self.estimator.steps[-1]

        # Generate transformer parameter combinations
        trans_params_dict = { k: v for d in trans_params for k, v in d.items() }
        keys = list(trans_params_dict)
        combos = [dict(zip(keys, p)) for p in product(*trans_params_dict.values())]

        # For every transformer params combinations, we fit a estimator model
        for combo in combos:
            # First step: We get the transformation for each combo
            trans_pipe.set_params(**combo)

            Xt, Yt = trans_pipe.transform(X=[], y=y)

            # Second step: We grid search over the final estimator
            super(TSFGridSearch, self).__init__(final_estimator, final_estim_params, scoring=self.scoring, cv=self.cv)
            super(TSFGridSearch, self).fit(Xt, Yt)

            # Did we find a better model ?
            score = super(TSFGridSearch, self).score(Xt, Yt)
            if best_score is None or score > best_score:
                best_score = score
                best_partial_params = combo
                best_estimator_params = self.best_params_
                best_partial_params.update(best_estimator_params)

                # Little transformation for estimator dict keys
                for key in best_estimator_params.keys():
                    best_partial_params[final_estimator_name + "__" + key] = best_partial_params.pop(key)

        # Set the best estimator and fit it
        self.best_params_ = best_partial_params
        self.best_estimator_ = clone(pipe_estimator).set_params(
            **self.best_params_)
        self.best_estimator_.fit(X=[], y=y)

        return self
