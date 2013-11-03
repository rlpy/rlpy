.. _install:

************
Installation
************

The installation consists of three steps:

1. :ref:`Download the package <download>`
2. :ref:`Install required libraries <dependencies>`
3. :ref:`Compile C-Extenstions of RLPy <compile>`


.. _download:

1. Download
===========

Stable Version
--------------
You can download the latest stable version of RLPy from http://acl.mit.edu/RLPy/RLPy.zip.
Extract the package in your desired location.

Development Version
-------------------
The development is maintained bitbucket at https://bitbucket.org/rlpy/rlpy.
The git-repository with the latest development version can be cloned via::

    git clone https://bitbucket.org/rlpy/rlpy.git RLPy

This will give you a copy of the repository in the directory `RLPy`. You might
want to change the location as you wish.

.. _dependencies:

2. Dependencies
===============

RLPy requires the following packages besides Python:

GCC >= 4.6
    for compiling some C++ extensions which use the C++11 standard.
Graphviz (optional) 
    for creating the graphical ouput of the code profiling tool.
Tk
    as a backend for matplotlib and for visualizations of some domains.

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
- joblib
- hyperopt


Ubuntu / Debian
---------------
Install the non-python dependencies with::

    sudo apt-get install graphviz tk blt tcl gcc g++


To install the Python packages we recommend using Anaconda as described in
:ref:`the following section <anaconda>` . This will ensure you have the latest versions of
each package.

**Alternatively** you can install the python packages via apt. Note however that
these packages will usually be older.
You can install them by executing::

    sudo apt-get install python-dev python-setuptools python-sklearn python-numpy python-scipy python-matplotlib python-networkx graphviz python-pip tcl-dev tk-dev python-tk cython
    pip install joblib hyperopt pymongo



OS X
----

Check what version of gcc you have by executing::
    
    gcc --version

If it is older than 4.6, install a newer version. You can find compiled
packages at http://sourceforge.net/projects/hpc/files/hpc/gcc or use MacPorts.

To install the Python packages we recommend either using MacPorts and pip or Anaconda (as described in
:ref:`the Anaconda section <anaconda>`. This will ensure you have the latest versions of
each package.

.. warning::
    At the moment, the Python binaries shipped with Anaconda are built with old OS X versions for
    compatibility. Unfortunately, the employed environment does not support compiling C++11 code,
    which we use for the `Representation.KernelizediFDD` representation.

    If you use Anaconda, you will therefore get errors when building the C++11 extensions. However,
    everything in RLPy will work except the kernelized iFDD representation.

Windows
-------

We recommend using Anaconda for installing Python and all dependencies. Follow the instructions in 
:ref:`the following section <anaconda>`. On Windows, Anaconda also comes with a gcc compiler.

.. warning::
    Unfortunately, matplotlib shipped with Anaconda does not contain the `tkagg` backend, which we
    use by default. At the moment you need to install matplotlib manually with tkinter support for RLPy 
    to work properly. We hope this issue is fixed soon. See also 
    https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/G4McL1bclAs
    for updates.

    If you see an error complaining that the module `_tkagg` could not be imported, change 
    the `matplotlib_backend` variable in `Tools.GeneralTools` to \"qtagg\". While this workaround
    allows you to use matplotlib, it may result in interactive matplotlib plots to not be shown.
    
.. warning::
    A couple of problems arise when building our Cython / C++ Extensions on Windows. It requires therefore
    some workarounds to get all extensions running on Windows. For details see
    https://bitbucket.org/rlpy/rlpy/issue/31/windows-anaconda-installation-problems
    Unfortunately, the problems are caused by packages we rely on and are therefore not easy to resolve for us.
    
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
    pip install joblib hyperopt pymongo

.. _compile:

3. Build Extensions of RLPy
===========================

Build the C++ / Cython extensions of RLPy by executing in your RLPy directory::

    python setup.py build_ext --inplace

.. note:: 
    If you are using a MacOS and a MacPorts version of gcc and you get an
    error about the `-arch` parameters, try using::

        ARCHFLAGS="" python setup.py build_ext --inplace


.. tip::
    You can verify that your rlpy installation works well by running the testsuite in
    the `tests` directory. You can do so by executing from the rlpy directory::

        nosetests tests --exe

RLPy is now successfully installed. For an introduction on how to use the
framework have a look at :ref:`tutorial`.
