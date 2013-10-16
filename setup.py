"""
This script can be used to compile further extensions that may
speed up computations.
Please run:
python setup.py build_ext --inplace
"""

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

import sys
#if sys.platform == 'darwin':
#  extra_args = [] #"-std=c++0x", "-stdlib=libc++"]
#else:
#  extra_args = ["-std=c++0x"]
extra_args = []
setup(name="_transformations",
      cmdclass = {"build_ext": build_ext},
      ext_modules=[
          Extension("Representations.hashing",
                             ["Representations/hashing.pyx"],
                             include_dirs=[numpy.get_include(), "Representations"]),
          Extension("Domains.HIVTreatment_dynamics",
                             ["Domains/HIVTreatment_dynamics.pyx"],
                             include_dirs=[numpy.get_include(), "Representations"]),
          Extension("Representations.kernels",
                             ["Representations/kernels.pyx",
                              "Representations/c_kernels.cc",
                              "Representations/c_kernels.pxd"],
                             language="c++",
                             extra_compile_args=extra_args + ["-std=c++0x"],
                             include_dirs=[numpy.get_include(), "Representations"]),
          Extension("Tools._transformations",
                             ["Tools/transformations.c"],
                             include_dirs=[numpy.get_include()]),
          Extension("Representations.FastCythonKiFDD",
                             ["Representations/FastCythonKiFDD.pyx",
                              "Representations/c_kernels.pxd",
                              "Representations/c_kernels.cc",
                              "Representations/FastKiFDD.cc"],
                             language="c++",
                             extra_compile_args=["-std=c++0x"] + extra_args,
                             include_dirs=[numpy.get_include(), "Representations"],
                             ),])
