from Tools import *
import numpy as np
from Representation import Representation

class TileCoding(Representation):
    """
    Tile Coding Representation with Hashing Trick
    based on http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tiles.html

    currently, no collision detection is implemented
    """

    BIG_INT = 2147483647

    def __init__(self, domain, logger, dim, scaling, num_tilings,
                 no_shift_dims=[], exclude_dims=[], safety="super",
                 seed=0, statistics=True):
        """
        dim: number of dimensions of the features, should be power of 2
        scaling: scaling factor for each dimension of the state space
        num_tilings: number of tilings
        exclude_dims: list of ids of dimensions that are not considered
        no_shift_dims: list of ids of dimensions where the tilings should
                       not be displaced
        seed: random seed for hashing
        safety: type of collision checking
            either super = detect and avoid all collisions
                   lazy  = faster but may have some false positives
                   none  = don't care about collisions
        each tile has width scaling[i] * num_tilings in the i-ths dimension

        """
        self.features_num = dim
        super(TileCoding, self).__init__(domain, logger)
        self.dim = dim
        self.scaling = scaling
        self.num_tilings = num_tilings
        self.exclude_dims = exclude_dims
        self.no_shift_dims = no_shift_dims
        self.seed=seed
        self.statistics = statistics
        self.safety = safety
        if safety == "super":
            size = self.domain.state_space_dims
        elif safety == "lazy":
            size = 1
        else:
            size = 0
        self.check_data = -np.ones((dim, size), dtype=np.int)
        self.counts = np.zeros((self.dim), dtype=np.long)
        self.collisions = 0
        self.R = np.random.RandomState(seed).randint(self.BIG_INT / 4  ,size=dim)

    def phi_nonTerminal(self, s):

        phi = np.zeros((self.features_num))
        sn = np.array((s - self.domain.statespace_limits[:,0]) / self.scaling, dtype=int)
        l = []
        for i in range(self.num_tilings):
            # compute "virtual" address
            A = sn - np.mod(sn - i, self.num_tilings)
            # avoid shifting by simply taking the (scaled) state values
            A[self.no_shift_dims] = sn[self.no_shift_dims]
            # exclude dimensions by setting constant 0, due to hashing the
            # constant dimensions to not matter afterwards
            A[self.exclude_dims] = 0
            # compute "physical" address
            j = self._physical_addr(A)
            phi[j] = 1
            l.append(j)
        return phi


    def _hash(self, A, increment=449, max=None):
        """
        hashing without collision detection
        """
        # TODO implement in cython if speed needs to be improved
        max = self.dim if max == None else max
        return int(self.R[np.mod(A + np.arange(len(A))*increment, self.dim)].sum()) % max

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
            check_val = self._bash(A, increment = 457, max = self.BIG_INT)

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
            for i in xrange(self.dim):
                h1 = (h1 + h2) % self.dim
                if self.counts[h1] == 0 or np.all(self.check_data[h1] == check_val):
                    self.check_data[h1] = check_val
                    self.counts[h1] += 1
                    return h1
            self.collisions += 1
            self.logger.log("Tile memory too small")
            return h1




if __name__ == "__main__":
    from Domains.acrobot import Acrobot
    domain = Acrobot(None)
    scaling = ((domain.statespace_limits[:,1] - domain.statespace_limits[:,0]) / 6.)
    t = TileCoding(scaling=scaling, logger=Logger(), dim=2000,
                   domain=domain, num_tilings=1)
    for i in np.linspace(-1, 1, 20):
        print i
        print np.nonzero(t.phi_nonTerminal(np.array([np.pi*i,0.,0.,0.])))[0][0]

