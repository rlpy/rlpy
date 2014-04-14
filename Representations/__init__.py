

from Tabular import Tabular
from IncrementalTabular import IncrementalTabular
from IndependentDiscretization import IndependentDiscretization
from IndependentDiscretizationCompactBinary import IndependentDiscretizationCompactBinary
from RBF import RBF
from iFDD import iFDD
from Fourier import Fourier
from BEBF import BEBF
from OMPTD import OMPTD
from TileCoding import TileCoding

from KernelizediFDD import linf_triangle_kernel, gaussian_kernel, KernelizediFDD
try:
    from KernelizediFDD import FastKiFDD
except ImportError:
    print "C-Extensions not build, Fast Kernelized iFDD not available"
    FastKiFDD = KernelizediFDD
from LocalBases import RandomLocalBases
from LocalBases import NonparametricLocalBases
