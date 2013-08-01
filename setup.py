"""
This script can be used to compile further extensions that may
speed up computations.
Please run:
python setup.py build_ext --inplace
"""

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy
setup(name="_transformations",
      cmdclass = {"build_ext": build_ext},
      ext_modules=[
          Extension("Domains.HIVTreatment_dynamics",
                             ["Domains/HIVTreatment_dynamics.pyx"],
                             include_dirs=[numpy.get_include()]),
          Extension("Representations.kernels",
                             ["Representations/kernels.pyx"],
                             include_dirs=[numpy.get_include()]),
          Extension("Tools._transformations",
                             ["Tools/transformations.c"],
                             include_dirs=[numpy.get_include()])])
