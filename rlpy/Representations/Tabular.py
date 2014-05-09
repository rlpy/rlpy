"""Tabular representation"""

from .Representation import Representation
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class Tabular(Representation):

    def __init__(self, domain, discretization=20):
        # Already performed in call to superclass
        self.setBinsPerDimension(domain, discretization)
        self.features_num = int(np.prod(self.bins_per_dim))
        super(Tabular, self).__init__(domain, discretization)

    def phi_nonTerminal(self, s):
        id = self.hashState(s)
        F_s = np.zeros(self.agg_states_num, bool)
        F_s[id] = 1
        return F_s

    def featureType(self):
        return bool
