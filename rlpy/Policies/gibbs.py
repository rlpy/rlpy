"""Gibbs policy"""

from .Policy import DifferentiablePolicy
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"


class GibbsPolicy(DifferentiablePolicy):

    """
    Gibbs policy for finite number of actions

    Warning: assumes that the features for each action are stacked, i.e.,
    a feature vector consists of |A| identical stacked vectors.
    """

    def dlogpi(self, s, a):

        v = self.probabilities(s, False)
        n = self.representation.features_num
        phi = self.representation.phi(s, False)
        res = -np.outer(v, phi)
        res.shape = self.representation.weight_vec.shape
        res[a * n:(a + 1) * n] += phi
        assert not np.any(np.isnan(res))
        return res

    def probabilities(self, s, terminal):

        phi = self.representation.phi(s, terminal)
        n = self.representation.features_num
        v = np.exp(np.dot(self.representation.weight_vec.reshape(-1, n), phi))
        v[v > 1e50] = 1e50
        r = v / v.sum()
        assert not np.any(np.isnan(r))
        return r
