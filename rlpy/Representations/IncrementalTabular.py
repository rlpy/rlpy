"""Incrementally expanded Tabular Representation"""

from .Representation import Representation
import numpy as np
from copy import deepcopy

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class IncrementalTabular(Representation):
    """
    Identical to Tabular representation (ie assigns a binary feature function 
    f_{d}() to each possible discrete state *d* in the domain, with
    f_{d}(s) = 1 when d=s, 0 elsewhere.
    HOWEVER, unlike *Tabular*, feature functions are only created for *s* which
    have been encountered in the domain, not instantiated for every single 
    state at the outset.

    """
    hash = None

    def __init__(self, domain, discretization=20):
        self.hash = {}
        self.features_num = 0
        self.isDynamic = True
        super(
            IncrementalTabular,
            self).__init__(
            domain,
            discretization)

    def phi_nonTerminal(self, s):
        hash_id = self.hashState(s)
        hashVal = self.hash.get(hash_id)
        F_s = np.zeros(self.features_num, bool)
        if hashVal is not None:
            F_s[hashVal] = 1
        return F_s

    def pre_discover(self, s, terminal, a, sn, terminaln):
        return self._add_state(s) + self._add_state(sn)

    def _add_state(self, s):
        """
        :param s: the (possibly un-cached) state to hash.
        
        Accepts state ``s``; if it has been cached already, do nothing and 
        return 0; if not, add it to the hash table and return 1.
        
        """
        
        hash_id = self.hashState(s)
        hashVal = self.hash.get(hash_id)
        if hashVal is None:
            # New State
            self.features_num += 1
            # New id = feature_num - 1
            hashVal = self.features_num - 1
            self.hash[hash_id] = hashVal
            # Add a new element to the feature weight vector, theta
            self.addNewWeight()
            return 1
        return 0

    def __deepcopy__(self, memo):
        new_copy = IncrementalTabular(
            self.domain,
            self.discretization)
        new_copy.hash = deepcopy(self.hash)
        return new_copy

    def featureType(self):
        return bool
