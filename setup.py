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
import os
import sys
if sys.platform == 'darwin':
    # by default use clang++ as this most likely to have c++11 support
    # on OSX
    if "CC" not in os.environ or os.environ["CC"] == "":
        os.environ["CC"] = "clang++"
        extra_args = ["-std=c++0x", "-stdlib=libc++"]
else:
    extra_args = []

setup(name="_transformations",
      cmdclass={"build_ext": build_ext},
      ext_modules=[
          Extension("rlpy.Representations.hashing",
                    ["rlpy/Representations/hashing.pyx"],
                    include_dirs=[numpy.get_include(
                    ), "rlpy/Representations"]),
          Extension("rlpy.Domains.HIVTreatment_dynamics",
                    ["rlpy/Domains/HIVTreatment_dynamics.pyx"],
                    include_dirs=[numpy.get_include(
                    ), "rlpy/Representations"]),
          Extension("rlpy.Representations.kernels",
                    ["rlpy/Representations/kernels.pyx",
                     "rlpy/Representations/c_kernels.cc",
                     "rlpy/Representations/c_kernels.pxd"],
                    language="c++",
                    #extra_compile_args=extra_args + ["-std=c++0x"],
                    include_dirs=[numpy.get_include(
                    ), "rlpy.Representations"]),
          Extension("rlpy.Tools._transformations",
                    ["rlpy/Tools/transformations.c"],
                    include_dirs=[numpy.get_include()]),
          Extension("rlpy.Representations.FastCythonKiFDD",
                    ["rlpy/Representations/FastCythonKiFDD.pyx",
                     "rlpy/Representations/c_kernels.pxd",
                     "rlpy/Representations/c_kernels.cc",
                     "rlpy/Representations/FastKiFDD.cc"],
                    language="c++",
                    extra_compile_args=["-std=c++0x"] + extra_args,
                    include_dirs=[
                        numpy.get_include(
                        ),
                        "rlpy/Representations"],
                    ), ])
