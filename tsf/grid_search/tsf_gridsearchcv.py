from sklearn.model_selection import GridSearchCV
from tsf_pipeline import TSFPipeline
from itertools import product


class TSFGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, groups=None, **fit_params):
        # Transformers data
        trans_params = self.param_grid[:-1]
        trans_list = self.estimator.steps[:-1]
        trans_list.append(('', None))
        trans_pipe = TSFPipeline(trans_list)

        # Estimator params
        estim_params = self.param_grid[-1]
        estimator = self.estimator.steps[-1][1]

        # Generate transformer parameter combinations
        trans_params_dict = { k: v for d in trans_params for k, v in d.items() }
        keys = list(trans_params_dict)
        combos = [dict(zip(keys, p)) for p in product(*trans_params_dict.values())]

        # For every transformer params combinations, we fit a estimator model
        for combo in combos:
            # First step: We get the transformation for each combo
            trans_pipe.set_params(**combo)
            Xt, Yt = trans_pipe.transform(X=[], y=y)

            print "transformed!"

            # Second step: We grid search over the final estimator
            estimator_gs = GridSearchCV(estimator, estim_params)
            estimator_gs.fit(Xt, Yt)

            print "SCORE: " + str(estimator_gs.score(Xt, Yt))
            quit()