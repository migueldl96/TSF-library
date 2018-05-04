Multivariate problems
*********************
Sometimes it is useful to combine several strongly linked time series to obtain better results when forecasting. For
example, SimpleAR transformer can be applied to a rains time serie to predict when you will have to take the umbrella,
but you would obtain better performance on your predictions if you could combine this information with temperature as they
are strongly linked climatic conditions.

This kind of problems are call **multivariate time series** and are very present in real world problems in which we
have one target serie that we call *endogenous* and others called *exogenous* relevant for our problem.
TSF library is prepared to deal with these kind of problems.

Dealing with several time series
================================
The ``y`` parameter on ``transform`` methods should receive the time serie. However, it can be a 1 dimension array (*vector*)
representing one time serie or a 2 dimensions array (*matrix*) in which every row will represent a single time serie. In this
case, the endogenous serie will be always the first row of the matrix, and the rest the exogenous ones.

When working with several time series, the window transformers will be applied to all of them. This time we'll use another
preprocessing algorithm included in TSF library: **DinamicWindow**.

.. code-block:: python

    from tsf.windows import DinamicWindow

    time_series =   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],                    # Endogenous serie
                     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],		# Exogenous serie #1
                     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]		# Exogenous serie #2

    dw = DinamicWindow(stat='variance', ratio=0.1, metrics=['variance', 'mean'])
    X = dw.transform(X=[], y=time_series)
    y = dw.offset_y(X, time_series)

    print X
    print y

The length for every window in DinamicWindow algorithm is determined by a limit depending on ``stat`` and ``ratio`` parameters.
In this case, the windows will grow while the window variance is less than 10% global variance. Once the limit is reached,
the window samples will be summarize in *variance* and *mean* (``metrics`` parameter).

Running this block of code, we'll get this output::

    > python multivariate.py
    [[ 0.          0.          0.         10.          0.         20.        ]
     [ 0.25        0.5         0.25       10.5         0.25       20.5       ]
     [ 0.66666667  1.          0.66666667 11.          0.66666667 21.        ]
     [ 0.66666667  2.          0.66666667 12.          0.66666667 22.        ]
     [ 0.66666667  3.          0.66666667 13.          0.66666667 23.        ]
     [ 0.66666667  4.          0.66666667 14.          0.66666667 24.        ]
     [ 0.66666667  5.          0.66666667 15.          0.66666667 25.        ]
     [ 0.66666667  6.          0.66666667 16.          0.66666667 26.        ]
     [ 0.66666667  7.          0.66666667 17.          0.66666667 27.        ]]
    [1 2 3 4 5 6 7 8 9]

As you can see, transforming these time series using DinamicWindow algorithm returns an input matrix with 6 features. By default,
all the time series are involved in the transformation, so the first two columns correspond to *variance* and *mean* for the first
serie, the next two columns for the second serie and the last two columns for the thirst one. The outputs are the elements of
our *endogenous* serie as it is our target.

Skip some time series
=====================
Maybe you don't want an algorithm to be applied in some series. It is useful when concatenating several transformers with
pipelines and working with ordinal time series where DinamicWindow doesn't make much sense. To avoid applying an algorithm
to a serie, every transformer in TSF library has a ``indexs`` parameter that allows you to indicate which series you
want to include in the preprocessing task. By default, this parameters is ``None``, meaning that all series are considered.
Let's do a little modification to our previous script:

.. code-block:: python

    from tsf.windows import DinamicWindow

    time_series =   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],			# Endogenous serie
                     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],		# Exogenous serie #1
                     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]		# Exogenous serie #2

    dw = DinamicWindow(stat='variance', ratio=0.1, metrics=['variance', 'mean'], indexs=[0, 2])
    X = dw.transform(X=[], y=time_series)
    y = dw.offset_y(X, time_series)

    print X
    print y

The code below suppose that we are not interested in getting DinamicWindow information for the second serie. Running it,
we'll get the following output::

    > python multivariate.py
    [[ 0.          0.          0.         20.        ]
     [ 0.25        0.5         0.25       20.5       ]
     [ 0.66666667  1.          0.66666667 21.        ]
     [ 0.66666667  2.          0.66666667 22.        ]
     [ 0.66666667  3.          0.66666667 23.        ]
     [ 0.66666667  4.          0.66666667 24.        ]
     [ 0.66666667  5.          0.66666667 25.        ]
     [ 0.66666667  6.          0.66666667 26.        ]
     [ 0.66666667  7.          0.66666667 27.        ]]
    [1 2 3 4 5 6 7 8 9]

We have ignored the second time serie in this example, so the third and forth column have disappeared.

.. note::
    ``indexs`` parameters should receive an array of ints indicating the indices of the rows from the time series matrix.
    If an index is out of bounds, you will get a ``UserWarning``, but program will continue its execution.