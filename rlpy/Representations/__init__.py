from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from .Tabular import Tabular
from .IncrementalTabular import IncrementalTabular
from .IndependentDiscretization import IndependentDiscretization
from .IndependentDiscretizationCompactBinary import IndependentDiscretizationCompactBinary
from .RBF import RBF
from .iFDD import iFDD, iFDDK
from .Fourier import Fourier
from .BEBF import BEBF
from .OMPTD import OMPTD
from .TileCoding import TileCoding

from .KernelizediFDD import linf_triangle_kernel, gaussian_kernel, KernelizediFDD
from .LocalBases import RandomLocalBases
from .LocalBases import NonparametricLocalBases
