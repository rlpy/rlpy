"""
Installation script for RLPy

Large parts of this file were taken from the pandas project
(https://github.com/pydata/pandas) which have been permitted for use under the
BSD license.
"""
import sys
try:
    from setuptools import setup, Command, find_packages
except ImportError:
    sys.exit("The setup requires setuptools, you can install it with 'easy_install setuptools'")

from distutils.extension import Extension
from distutils.command.build import build
from distutils.command.sdist import sdist
from distutils.command.build_ext import build_ext as _build_ext
import os
import shutil
import pkg_resources
from os.path import join as pjoin

version = '1.3.5'

if sys.platform == 'darwin':
    # by default use clang++ as this most likely to have c++11 support
    # on OSX
    if "CC" not in os.environ or os.environ["CC"] == "":
        os.environ["CC"] = "clang++"
        extra_args = ["-std=c++0x", "-stdlib=libc++"]
else:
    extra_args = []

try:
    from Cython.Distutils import build_ext as _build_ext
    # from Cython.Distutils import Extension # to get pyrex debugging symbols
    cython = True
except ImportError:
    cython = False


class build_ext(_build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if hasattr(ext, 'include_dirs') and not numpy_incl in ext.include_dirs:
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


class CheckingBuildExt(build_ext):
    """Subclass build_ext to get clearer report if Cython is necessary."""

    def check_cython_extensions(self, extensions):
        for ext in extensions:
            for src in ext.sources:
                if not os.path.exists(src):
                    raise Exception("""Cython-generated file '%s' not found.
                Cython is required to compile pandas from a development branch.
                Please install Cython or download a release package of rlpy.
                """ % src)

    def build_extensions(self):
        self.check_cython_extensions(self.extensions)
        build_ext.build_extensions(self)


class CythonCommand(build_ext):
    """Custom distutils command subclassed from Cython.Distutils.build_ext
    to compile pyx->c, and stop there. All this does is override the
    C-compile method build_extension() with a no-op."""
    def build_extension(self, ext):
        pass


class DummyBuildSrc(Command):
    """ numpy's build_src command interferes with Cython's build_ext.
    """
    user_options = []

    def initialize_options(self):
        self.py_modules_dict = {}

    def finalize_options(self):
        pass

    def run(self):
        pass


class CheckSDist(sdist):
    """Custom sdist that ensures Cython has compiled all pyx files to c."""

    _pyxfiles = ["rlpy/Representations/hashing.pyx",
                 "rlpy/Domains/HIVTreatment_dynamics.pyx",
                 "rlpy/Representations/kernels.pyx"]

    def initialize_options(self):
        sdist.initialize_options(self)

    def run(self):
        if 'cython' in cmdclass:
            self.run_command('cython')
        else:
            for pyxfile in self._pyxfiles:
                cfile = pyxfile[:-3] + 'c'
                cppfile = pyxfile[:-3] + 'cpp'
                msg = "C-source file '%s' not found." % (cfile) +\
                    " Run 'setup.py cython' before sdist."
                assert os.path.isfile(cfile) or os.path.isfile(cppfile), msg
        sdist.run(self)


class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = ['transformations.c',
                               ]

        for root, dirs, files in list(os.walk('rlpy')):
            for f in files:
                if f in self._clean_exclude:
                    continue

                if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o',
                                               '.pyo',
                                               '.pyd', '.c', '.orig'):
                    self._clean_me.append(pjoin(root, f))
            for d in dirs:
                if d == '__pycache__':
                    self._clean_trees.append(pjoin(root, d))

        for d in ('build',):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                shutil.rmtree(clean_tree)
            except Exception:
                pass

cmdclass = {'clean': CleanCommand,
            'build': build,
            'sdist': CheckSDist}

if cython:
    suffix = '.pyx'
    cmdclass['build_ext'] = CheckingBuildExt
    cmdclass['cython'] = CythonCommand
else:
    suffix = '.c'
    cmdclass['build_src'] = DummyBuildSrc
    cmdclass['build_ext'] = CheckingBuildExt

# only use Cython if explicitly told
USE_CYTHON =  os.getenv('USE_CYTHON', False)
# always cythonize if C-files are not present
USE_CYTHON = not os.path.exists("rlpy/Representations/hashing.c") or USE_CYTHON
extensions = [
          Extension("rlpy.Representations.hashing",
                    ["rlpy/Representations/hashing.pyx"],
                    include_dirs=["rlpy/Representations"]),
          Extension("rlpy.Domains.HIVTreatment_dynamics",
                    ["rlpy/Domains/HIVTreatment_dynamics.pyx"],
                    include_dirs=["rlpy/Representations"]),
          Extension("rlpy.Representations.kernels",
                    ["rlpy/Representations/kernels.pyx",
                     "rlpy/Representations/c_kernels.cc",
                     "rlpy/Representations/c_kernels.pxd"],
                    language="c++",
                    include_dirs=["rlpy.Representations"]),
          Extension("rlpy.Tools._transformations",
                    ["rlpy/Tools/transformations.c"],
                    include_dirs=[])]

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

if cython:
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
      zip_safe=False,
      cmdclass=cmdclass,
      long_description=open('README.rst').read(),
      packages=find_packages(exclude=['tests', 'tests.*']),
      include_package_data=True,
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
      setup_requires=['numpy >= 1.7'],
      ext_modules=extensions,
      test_suite='tests'
      )
