from sklearn.model_selection import GridSearchCV
from tsf_pipeline import TSFPipeline
from itertools import product
from sklearn.base import clone


# TODO: Control de errores
class TSFGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, groups=None, **fit_params):
        best_score = 0

        # Transformers data
        trans_params = self.param_grid[:-1]
        trans_list = self.estimator.steps[:-1]
        trans_list.append(('', None))
        trans_pipe = TSFPipeline(trans_list)

        # Estimator params
        estim_params = self.param_grid[-1]
        estimator_name, estimator = self.estimator.steps[-1]

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
            estimator_gs = GridSearchCV(estimator, estim_params)
            estimator_gs.fit(Xt, Yt)

            # Did we find a best model ?
            score = estimator_gs.score(Xt, Yt)
            if score > best_score:
                best_score = score
                self.best_params_ = combo
                best_estimator_params = estimator_gs.best_params_
                self.best_params_.update(best_estimator_params)

                best_Xt = Xt
                best_Yy = Yt

                # Little transformation for estimator dict keys
                for key in best_estimator_params.keys():
                    self.best_params_[estimator_name + "__" + key] = self.best_params_.pop(key)

        # Set the best estimator
        self.best_estimator_ = clone(self.estimator).set_params(
            **self.best_params_)
        self.best_estimator_.fit(X=best_Xt, y=best_Yy)

        return self
