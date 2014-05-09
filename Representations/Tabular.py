"""Tabular representation"""

from Representation import *

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"

class Tabular(Representation):
    """
    Tabular representation that assigns a binary feature function f_{d}() 
    to each possible discrete state *d* in the domain. (For bounded continuous
    dimensions of s, discretize.)
    f_{d}(s) = 1 when d=s, 0 elsewhere.  (ie, the vector of feature functions
    evaluated at *s* will have all zero elements except one).
    NOTE that this representation does not support unbounded dimensions

    """
    def __init__(self,domain,logger,discretization = 20):
        self.setBinsPerDimension(domain,discretization) # Already performed in call to superclass
        self.features_num = int(prod(self.bins_per_dim))
        super(Tabular,self).__init__(domain,logger,discretization)
    def phi_nonTerminal(self,s):
        id      = self.hashState(s)
        F_s     = zeros(self.agg_states_num,bool)
        F_s[id] = 1
        return F_s
    def featureType(self):
        return bool

