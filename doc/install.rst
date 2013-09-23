.. _install:

************
Installation
************

Download
========

Stable Version
--------------
You can download the latest stable version of RLPy from http://acl.mit.edu/RLPy.
Extract the package in your desired location.

Development Version
-------------------
The development is maintained bitbucket at https://bitbucket.org/rlpy/rlpy.
The git-repository with the latest development version can be cloned via::

    git clone git://bitbucket.org/rlpy/rlpy.git RLPy

This will give you a copy of the repository in the directory `RLPy`. You might
want to change the location as you wish.

Dependencies
============

RLPy requires the following packages besides Python:

Graphviz
    (optional) For creating the graphical ouput of the code profiling tool.
GCC >= 4.6
    For compiling some C++ extensions which use the C++11 standard.


In addition, RLPy requires Python 2.7 to run. We do not support Python 3 at the
moment since most scientific libraries still require Python 2.
The following Python packages need to be available:

- Numpy >= 1.7
- Scipy
- Matplotlib >= 1.2
- Networkx
- PyTk
- scikit-learn
- Cython


Ubuntu / Debian
---------------
Install the non-python dependencies with::

    sudo apt-get install graphviz tk blt tcl gcc

To install the Python packages we recommend using Anaconda as described in
:ref:`the following section <anaconda>` . This will ensure you have the latest versions of
each package.
**Alternatively** you can install the python packages via apt. Note however that
these packages will usually be older.
You can install them by executing::

    sudo apt-get install python-dev python-setuptools python-sklearn python-numpy python-scipy python-matplotlib python-networkx graphviz python-pip tcl-dev tk-dev python-tk cython


.. _anaconda:

Anaconda
--------

We recommend using 
the `Anaconda Python distribution <https://store.continuum.io/cshop/anaconda/>`_. This software package comes with a current version of Python
and many libraries necessary for scientific computing. It simplifies installing
and updating Python libraries significantly on Windows, MacOS and Linux.
Please follow the original `installation instructions
<http://docs.continuum.io/anaconda/install.html>`_ of Anaconda.

After installing Anaconda, install the dependencies of RLPy by executing::

    conda install numpy scipy matplotlib networkx tk scikit-learn cython

Build Extensions of RLPy
------------------------

Build the C++ / Cython extensions of RLPy by executing in your RLPy directory::

    python setup.py build_ext --inplace


