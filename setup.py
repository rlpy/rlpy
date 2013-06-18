"""
This script can be used to compile further extensions that may
speed up computations.
Please run:
python setup.py build_ext --inplace
"""

from distutils.core import setup, Extension
import numpy
setup(name="_transformations",
      ext_modules=[Extension("Tools._transformatins",
                             ["Tools/transformations.c"],
                             include_dirs=[numpy.get_include()])])
