|Travis|_ |AppTrevor|_ |Codecov|_ |Python27| |Python35|

.. |Travis| image:: https://travis-ci.org/migueldl96/TSF-library.svg?branch=master
.. _Travis: https://travis-ci.org/migueldl96/TSF-library

.. |AppTrevor| image:: https://ci.appveyor.com/api/projects/status/afjl2dkn4fb45d8p?svg=true
.. _AppTrevor : https://ci.appveyor.com/project/migueldl96/tsf-library/history

.. |Codecov| image:: https://codecov.io/gh/migueldl96/TSF-library/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/migueldl96/TSF-library

.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg


TSF library
===========

TSF is a library that extend Scikit-learn_ software composed by several time series preprocessing algorithms developed
at the University of Cordoba in the `Learning and Artificial Neural Networks (AYRNA)`_ research group.
This library is able to preprocess any time serie, either univariate or multivariate,
and create a inputs matrix for every sample of the time serie(s) so any model of Scikit-learn_ can be trained.

Times series data is preprocessed using windows-based models such as *Autorregresive Model*, which takes **n_prev**
samples for every sample of the time serie.
TSF comes with 3 different windows-based autorregressive models:

- SimpleAR
- DinamicWindow
- ClassChange

Further information about these models can be found on online documentation.

.. _Scikit-learn: https://github.com/scikit-learn/scikit-learn/
.. _Learning and Artificial Neural Networks (AYRNA): http://www.uco.es/grupos/ayrna/index.php/en


Installation
------------

Dependencies
~~~~~~~~~~~~

TSF use requires:

- Python (>= 2.7 or >= 3.4)
- NumPy (>= 1.8.2)
- SciPy (>= 0.13.3)
- Scikit-learn (>= 0.19)

User installation
~~~~~~~~~~~~~~~~~
The easiest way to install TSF and all the dependencies is using ``pip``::

    pip install tsf

You can also clone the repository and install the library manually:
   | git clone https://github.com/migueldl96/TSF-library.git
   | cd TSF-library
   | python setup.py install


Tests
-----
TSF code can be test using ``pytest`` on root directory. Once installed, please run the following command to
check the library was successfully installed::

    pytest tests/

.. note::
    ``Warnings`` should appear as tests change dynamically forecasting time horizon.


Documentation
-------------
Examples of use and installation guide can be found in the online documentation hosted by ReadTheDocs_.

.. _ReadTheDocs: https://tsf-library.readthedocs.io/en/latest/

References
----------

