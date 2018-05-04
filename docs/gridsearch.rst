TSFGridSearch
*************
When dealing with several transformers and estimators there is thousands of possible parameter combinations. Built-in
`GridSearchCV Scikit-learn`_ class helps to optimize these parameters choosing the ones that returns better performance.

TSF Library include a similar mechanism to optimize its transformers parameters (such as ``n_prev`` in *SimpleAR* or ``ratio``
in *DinamicWindow*): **TSFGridsearch**. It decorates original GridSearchCV ``fit`` method and adapt it to TSF Library needs,
therefore the use is identical as the genuine class.

Optimizing hiperparameters
==========================
The full potential of TSFGridSearch is when combining it with :ref:`tsfpipeline`. You can create a sequential list of step
transformations and optimize the parameters. Is this example, we'll use a combination of *SimpleAR* and *DinamicWindow* with
a MLPRegressor:

.. code-block:: python

    from tsf.windows import SimpleAR, DinamicWindow
    from tsf.pipeline import TSFPipeline
    from tsf.grid_search import TSFGridSearch
    from sklearn.neural_network import MLPRegressor

    # Random continous time series
    time_series =   [[0.2, 0.5, 0.4, 0.32, 0.7, 0.8, 0.91, 0.53, 0.12, -0.26],
                     [1.5, 1.54, 1.2, 1.96, 1.43, 1.32, 1.68, 1.23, 1.85, 1.01]]

    # Pipeline
    pipe = TSFPipeline([('ar', SimpleAR()),
                        ('dw', DinamicWindow()),
                        ('MLP', MLPRegressor())])

    # Params grid
    params = 	[
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

``best_params_`` attribute returns the dictionary with the best parameters combinations. Is this example, this dictionary is::

    > python gridsearch.py
    {'MLP__hidden_layer_sizes': 110, 'ar__n_prev': 2, 'dw__ratio': 0.1}

.. note::
    As randomness is not contemplated, best parameters dictionary may differ from the obtained in this example.

.. _GridSearchCV Scikit-learn: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html