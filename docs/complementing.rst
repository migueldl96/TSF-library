Complement a database
*********************
TSF algorithms not only serves to create inputs matrixs from a time serie; they can also complement input data to
obtain better performance when forescasting. In previous chapter we called ``transform`` method passing an empty array
to X parameter. Let's suppose we have not only our time serie but also some features (e.g. some climatological indices)
that we want to keep and complement with autorregresive information from our serie:

.. code-block:: python

    from tsf.windows import SimpleAR

    features = [[-10, -20, -30, -40, -50, -60, -70, -80, -90],
                [-11, -21, -31, -41, -51, -61, -71, -81, -91],
                [-12, -22, -32, -42, -52, -62, -72, -82, -92],
                [-13, -23, -33, -43, -53, -63, -73, -83, -93],
                [-14, -24, -34, -44, -54, -64, -74, -84, -94],
                [-15, -25, -35, -45, -55, -65, -75, -85, -95],
                [-16, -26, -36, -46, -56, -66, -76, -86, -96],
                [-17, -27, -37, -47, -57, -67, -77, -87, -97],
                [-18, -28, -38, -48, -58, -68, -78, -88, -98],
                [-19, -29, -39, -49, -59, -69, -79, -89, -99]]
    time_serie = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ar = SimpleAR(n_prev=5)
    X = ar.transform(X=features, y=time_serie)
    y = ar.offset_y(X, time_serie)

    print X
    print y

In the example below, the negative integers simulate some extra features that we want to keep in our model. If we call
transform method passing this ``features`` matrix to X argument, the SimpleAR information will be appended to the
``features`` matrix::

    > python complementing.py
    [[-15 -25 -35 -45 -55 -65 -75 -85 -95   0   1   2   3   4]
     [-16 -26 -36 -46 -56 -66 -76 -86 -96   1   2   3   4   5]
     [-17 -27 -37 -47 -57 -67 -77 -87 -97   2   3   4   5   6]
     [-18 -28 -38 -48 -58 -68 -78 -88 -98   3   4   5   6   7]
     [-19 -29 -39 -49 -59 -69 -79 -89 -99   4   5   6   7   8]]
    [5 6 7 8 9]
