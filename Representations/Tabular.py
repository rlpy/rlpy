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



class HashedTabular(Representation):
    """
    WARNING: EXPERIMENTAL!!


    Tabular Representation with Hashing Trick
    """

    BIG_INT = 2147483647

    def __init__(self, domain, logger, memory,
                 safety="super",
                 seed=0):
        """
        TODO
        """
        self.features_num = memory
        super(HashedTabular, self).__init__(domain, logger)
        # now only hashing stuff
        self.seed=seed
        self.safety = safety
        if safety == "super":
            size = self.domain.state_space_dims
            self.check_data = -np.ones((self.features_num, size), dtype=np.long)
        elif safety == "lazy":
            size = 1
        else:
            self.check_data = -np.ones((self.features_num), dtype=np.long)
        self.counts = np.zeros(self.features_num, dtype=np.long)
        self.collisions = 0
        self.R = np.random.RandomState(seed).randint(self.BIG_INT / 4  ,size=self.features_num).astype(np.int)

        if safety == "none":
            try:
                import hashing as h
                f = lambda self, A: h.physical_addr(A, self.R, self.check_data,
                                                   self.counts)[0]
                self._physical_addr = type(HashedTabular._physical_addr)(f, self, HashedTabular)
                print "Use cython extension for hashing trick"
            except Exception, e:
                print e
                print "Cython extension for hashing trick not available"

    def phi_nonTerminal(self, s):

        phi = np.zeros((self.features_num))
        sn = np.array(s, dtype="int")
        j = self._physical_addr(sn)
        phi[j] = 1
        return phi


    def _hash(self, A, increment=449, max=None):
        """
        hashing without collision detection
        """
        # TODO implement in cython if speed needs to be improved
        max = self.features_num if max == None else max
        return int(self.R[np.mod(A + np.arange(len(A))*increment, self.features_num)].sum()) % max

    def _physical_addr(self, A):
        """
        Map a virtual vector address A to a physical address i.e. the actual
        feature number.
        This is the actual hashing trick
        """
        h1 = self._hash(A)
        if self.safety == "super":
            # use full value to detect collisions
            check_val = A
        else:
            # use second hash
            check_val = self._hash(A, increment = 457, max = self.BIG_INT)

        if self.counts[h1] == 0:
            # first time, set up data
            self.check_data[h1] = check_val
            self.counts[h1] += 1
            return h1
        elif np.all(check_val == self.check_data[h1]):
            # clear hit, everything's fine
            self.counts[h1] += 1
            return h1
        elif self.safety == "none":
            self.collisions += 1
            return h1
        else:
            h2 = 1 + 2 * self._hash(A, max = self.BIG_INT / 4)
            for i in xrange(self.features_num):
                h1 = (h1 + h2) % self.features_num
                if self.counts[h1] == 0 or np.all(self.check_data[h1] == check_val):
                    self.check_data[h1] = check_val
                    self.counts[h1] += 1
                    return h1
            self.collisions += 1
            #self.logger.log("Tile memory too small")
            return h1

    def featureType(self):
        return bool

class QHashedTabular(HashedTabular, QFunRepresentation):
    pass


