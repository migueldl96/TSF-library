from tsf.windows import SimpleAR, DinamicWindow
from tsf.pipeline import TSFPipeline
from tsf.grid_search import TSFGridSearch
from sklearn.neural_network import MLPRegressor

# Random continous time series
time_series = [[0.2, 0.5, 0.4, 0.32, 0.7, 0.8, 0.91, 0.53, 0.12, -0.26],
               [1.5, 1.54, 1.2, 1.96, 1.43, 1.32, 1.68, 1.23, 1.85, 1.01]]

# Pipeline
pipe = TSFPipeline([('ar', SimpleAR()),
                    ('dw', DinamicWindow()),
                    ('MLP', MLPRegressor())])

# Params grid
params = [
    {
        'ar__n_prev': [1, 2, 3]
    },
    {
        'dw__ratio': [0.1, 0.2]
    },
    {
        'hidden_layer_sizes': [80, 90, 100, 110]
    }
]

# Grid search
grid = TSFGridSearch(pipe, params)

# Fit and best params
grid.fit(X=[], y=time_series)
print grid.best_params_