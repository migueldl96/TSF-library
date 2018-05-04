.. _tsfpipeline:

TSFPipeline
***********
Pipelines are a great tool included in Scikit-learn_ that allows to concatenate several transformers and estimators in a single
object. TSF library include its own Pipeline class that extend the original and allows concatenating several TSF transformers.

Creating a TSFPipeline
======================
Concatenating algorithms is a great idea when preprocessing time series as it allows you to get a database from autorregresive
techniques. ``TSFPipeline`` works in the same way as original Pipeline does and it is compatible with all Scikit-learn_
environment.

.. code-block:: python

    from tsf.windows import SimpleAR, DinamicWindow
    from tsf.pipeline import TSFPipeline

    time_series =   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],			# Endogenous serie
                     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]		# Exogenous serie #1

    pipe = TSFPipeline([('ar', SimpleAR(n_prev=2)),
                        ('dw', DinamicWindow(stat='variance', ratio=0.1, metrics=['variance', 'mean'])),
                        ('', None)])

    X, y = pipe.transform(X=[], y=time_series)

    print X
    print y

In the example below, TSFPipeline is applied to make the transformations without a final estimator. The algorithms are applied
sequentially and results appended to the array passed to ``X`` parameter::

    > python pipeline.py
    [[ 0.          1.         10.         11.          0.25        0.5
       0.25       10.5       ]
     [ 1.          2.         11.         12.          0.66666667  1.
       0.66666667 11.        ]
     [ 2.          3.         12.         13.          0.66666667  2.
       0.66666667 12.        ]
     [ 3.          4.         13.         14.          0.66666667  3.
       0.66666667 13.        ]
     [ 4.          5.         14.         15.          0.66666667  4.
       0.66666667 14.        ]
     [ 5.          6.         15.         16.          0.66666667  5.
       0.66666667 15.        ]
     [ 6.          7.         16.         17.          0.66666667  6.
       0.66666667 16.        ]
     [ 7.          8.         17.         18.          0.66666667  7.
       0.66666667 17.        ]]
    [2 3 4 5 6 7 8 9]


.. note::
    TSFPipeline ``transform`` method returns ``X`` and ``y``.

TSFPipeline is useful to apply the same transformation to several time series.

Adding a final estimator to TSFPipeline
=======================================
As genuine Pipeline do, you can append an estimator to the steps of the Pipeline so you can use methods like ``predict``
and ``fit`` directly from TSFPipeline object.

.. code-block:: python

    from tsf.windows import SimpleAR, DinamicWindow
    from tsf.pipeline import TSFPipeline
    from sklearn.linear_model import LassoCV

    train_series = 	[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]


    pipe = TSFPipeline([('ar', SimpleAR(n_prev=2)),
                        ('dw', DinamicWindow(stat='variance', ratio=0.1, metrics=['variance', 'mean'])),
                        ('lasso', LassoCV())])

    pipe.fit(X=[], y=time_series)

.. _Scikit-learn: https://github.com/scikit-learn/scikit-learn/