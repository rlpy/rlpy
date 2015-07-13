"""Fourier representation"""

from .Representation import Representation
from numpy import indices, pi, cos, dot
from numpy.linalg import norm
import numpy


__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class Fourier(Representation):
    """ Fourier representation.
    Represents the value function using a Fourier series of the specified
    order (eg 3rd order, 5th order, etc).
    See Konidaris, Osentoski, and Thomas, "Value Function Approximation in
    Reinforcement Learning using Fourier Basis" (2011).
    http://lis.csail.mit.edu/pubs/konidaris-aaai11a.pdf

    """

    def __init__(self, domain, order=3, scaling=False):
        """
        :param domain: the problem :py:class:`~rlpy.Domains.Domain.Domain` to learn
        :param order: The degree of approximation to use in the Fourier series
            (eg 3rd order, 5th order, etc).  See reference paper in class API.

        """
        dims = domain.state_space_dims
        self.coeffs = indices((order,) * dims).reshape((dims, -1)).T
        self.features_num = self.coeffs.shape[0]

        if scaling:
            coeff_norms = numpy.array(map(norm, self.coeffs))
            coeff_norms[0] = 1.0
            self.alpha_scale = numpy.tile(1.0/coeff_norms, (domain.actions_num,))
        else:
            self.alpha_scale = 1.0

        super(Fourier, self).__init__(domain)

    def phi_nonTerminal(self, s):
        # normalize the state
        s_min, s_max = self.domain.statespace_limits.T
        norm_state = (s - s_min) / (s_max - s_min)
        return cos(pi * dot(self.coeffs, norm_state))

    def featureType(self):
        return float

    def featureLearningRate(self):
        return self.alpha_scale
