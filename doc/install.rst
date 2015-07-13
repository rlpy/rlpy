.. _install:

************
Installation
************

If you have a running Python distribution on your system, the external dependencies of RLPy are most likly installed already. Try to install rlpy with distutils as discribed in the following sections. If you do not have a Python distribution installed or run into trouble during installation, have a look at the Dependencies section below.

Stable Version
==============

RLPy is available on Pypi. The latest stable version can be installed directly with pip::

    pip install -U rlpy

This command downloads the latest release of RLPy and installs it into the default package location
of your python distribution. 
Only the library itself is installed. If you like to have the documentation and example scripts, have a look at how to install the development version below.

For more information of the disutils package systems, have a look at the `documentation <https://docs.python.org/2/install/index.html#install-index>`_. 

If you are using MacOS make sure you have the latest version of Xcode using::

    xcode-select --install

Alternatively, you can download RLPy manually, extract the package and execute the installer by hand::
    
    python setup.py install

All RLPy packages are now successfully installed in Python's site-package directory. 

.. note::

    Only the Python files necessary to use the toolbox are installed. 
    Documentation and example scripts how to use the toolbox are not installed by pip.
    They are, however, included in the package, which can be downloaded from
    https://pypi.python.org/pypi/rlpy. We recommend to download the archive and extract the 
    `examples` folder to get examples of how to use the toolbox. Please also check out 
    the tutorial at :ref:`tutorial`.




.. _devInstall:

Development Version
===================

The current development version of RLPy can be installed in distutils editable mode with::

    pip install -e git+https://github.com/rlpy/rlpy.git#egg=rlpy

This command clones the RLPy repository into the directory `src/rlpy`, compiles all C-extensions and tells the Python distribution where to find RLPy by creating a `.egg-link` file in the default package directory.

Alternatively, you can clone the RLPy directory manually by::

    git clone https://github.com/rlpy/rlpy.git RLPy

and make your Python distribution aware of RLPy by::

    python setup.py develop

.. note::
    
    If you install rlpy directly from the development repository, you need cython to build the
    cython extensions. You can get the latest version of cython by: "pip install cython -U"

.. _dependencies:

************
Dependencies
************

We recommend using 
the `Anaconda Python distribution <https://store.continuum.io/cshop/anaconda/>`_. This software package comes with a current version of Python
and many libraries necessary for scientific computing. It simplifies installing
and updating Python libraries significantly on Windows, MacOS and Linux.
Please follow the original `installation instructions
<http://docs.continuum.io/anaconda/install.html>`_ of Anaconda.


RLPy requires the following software besides Python:

Tk
    as a backend for matplotlib and for visualizations of some domains.
Graphviz (optional) 
    for creating the graphical ouput of the code profiling tool.

If you are using the Anaconda Python distribution, you can install Tk by executing::

    conda install tk

In addition, RLPy requires Python 2.7 to run. We do not support Python 3 at the
moment since most scientific libraries still require Python 2.
