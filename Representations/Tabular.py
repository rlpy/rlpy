"""Tabular representation"""
import numpy as np
from Representation import Representation, QFunRepresentation

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"

class Tabular(Representation):

    def __init__(self,domain,logger,discretization = 20):
        self.setBinsPerDimension(domain,discretization) # Already performed in call to superclass
        self.features_num = int(np.prod(self.bins_per_dim))
        super(Tabular,self).__init__(domain,logger,discretization)

    def phi_nonTerminal(self,s):
        id      = self.hashState(s)
        F_s     = np.zeros(self.agg_states_num,bool)
        F_s[id] = 1
        return F_s

    def featureType(self):
        return bool

class QTabular(Tabular, QFunRepresentation):
    pass
