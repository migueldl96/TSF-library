Quickstart
**********

Installation
============
You can get TSF Library directly from PyPI like this::

    pip install tsf

Otherwise, you can clone the project directly from the Github repository using git clone::

    git clone https://github.com/migueldl96/TSF-library.git
    cd TSF-library
    [sudo] python setup.py install

First transformation: Look *n_prev* backward!
=============================================
The simplest preprocess algorithm is just to take some previous samples for every element of time serie. In TSF
this is call SimpleAR transformation and works with a fixed window size (we call window to the previous
samples used for forecast a sample of the serie). Let's jump right in with a synthetic time serie!

.. code-block:: python

    from tsf.windows import SimpleAR

    time_serie = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    ar = SimpleAR(n_prev=5)
    X = ar.transform(X=[], y=time_serie)

    print X

The code below import SimpleAR class from windows module, create an instance of it and call transform method. This method
will return the input matrix resulting from the transformation

Running this code we'll obtain::

    > python simplear.py
    [[1 2 3 4 5]
     [2 3 4 5 6]
     [3 4 5 6 7]
     [4 5 6 7 8]
     [5 6 7 8 9]]

That's out inputs matrix! Now we only need some the corresponding outputs to train any model. Luckily, all the TSF transformers have a ``offset_y`` method that returns
our precious outputs. Let's do some modifications to our previous script:

.. code-block:: python

    from tsf.windows import SimpleAR

    time_serie = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    ar = SimpleAR(n_prev=5)
    X = ar.transform(X=[], y=time_serie)
    y = ar.offset_y(X, time_serie)

    print X
    print y

If we run it now::

    > python simplear.py
    [[1 2 3 4 5]
     [2 3 4 5 6]
     [3 4 5 6 7]
     [4 5 6 7 8]
     [5 6 7 8 9]]
    [ 6  7  8  9 10]

That's it! We have our inputs matrix (X) and the output for every pattern (y).

.. note::
    From a time serie of 10 samples we have obtain 5 patterns. This is because we set ``n_prev`` to 5: the algorithm needs to take the first 5 samples to build the first pattern.
    All the autorregresive models always need to build patterns from previous samples, and always will return less patterns than the time serie length depending on the transformers parameters.


