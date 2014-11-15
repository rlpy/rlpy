"""
Installation script for RLPy
"""

from setuptools import setup, Extension, find_packages
#from Cython.Distutils import build_ext
import numpy
import os
import sys

version = '1.3.2'

if sys.platform == 'darwin':
    # by default use clang++ as this most likely to have c++11 support
    # on OSX
    if "CC" not in os.environ or os.environ["CC"] == "":
        os.environ["CC"] = "clang++"
        extra_args = ["-std=c++0x", "-stdlib=libc++"]
else:
    extra_args = []

# only use Cython if explicitly told
USE_CYTHON =  os.getenv('USE_CYTHON', False)
# always cythonize if C-files are not present
USE_CYTHON = not os.path.exists("rlpy/Representations/hashing.c") or USE_CYTHON
extensions = [
          Extension("rlpy.Representations.hashing",
                    ["rlpy/Representations/hashing.pyx"],
                    include_dirs=[numpy.get_include(), "rlpy/Representations"]),
          Extension("rlpy.Domains.HIVTreatment_dynamics",
                    ["rlpy/Domains/HIVTreatment_dynamics.pyx"],
                    include_dirs=[numpy.get_include(), "rlpy/Representations"]),
          Extension("rlpy.Representations.kernels",
                    ["rlpy/Representations/kernels.pyx",
                     "rlpy/Representations/c_kernels.cc",
                     "rlpy/Representations/c_kernels.pxd"],
                    language="c++",
                    include_dirs=[numpy.get_include(), "rlpy.Representations"]),
          Extension("rlpy.Tools._transformations",
                    ["rlpy/Tools/transformations.c"],
                    include_dirs=[numpy.get_include()])]

def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            elif ext in ('.pxd'):
                continue
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)
else:
    extensions = no_cythonize(extensions)

setup(name="rlpy",
      version=version,
      maintainer="Christoph Dann",
      maintainer_email="cdann@cdann.de",
      license="BSD 3-clause",
      description="Value-Function-Based Reinforcement-Learning Library for"
                  + " Education and Research",
      url="http://acl.mit.edu/rlpy/",
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: C++',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                  ],
      long_description=open('README.rst').read(),
      packages=find_packages(exclude=['tests', 'tests.*']),
      include_package_data=True,
#      package_data={'rlpy': [
#          'Domains/GridWorldMaps/*.txt',
#          'Domains/IntruderMonitoringMaps/*.txt',
#          'Domains/PinballConfigs/*.cfg',
#          'Domains/PacmanPackage/layouts/*.lay',
#          'Domains/SystemAdministratorMaps/*.txt',
#          "Representations/c_kernels.h",
#      ]
#      },
      install_requires=[
          'numpy >= 1.7',
          'scipy',
          'matplotlib >= 1.2',
          'networkx',
          'scikit-learn',
          'joblib',
          'hyperopt',
          'pymongo'
      ],
      ext_modules=extensions,
      test_suite='tests'
      )
