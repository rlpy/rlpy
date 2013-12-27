"""Radial Basis Function Representation"""

from Tools import perms
from Representation import Representation, QFunRepresentation
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class RBF(Representation):
    state_dimensions = None
    rbfs_mu         = None  #: The mean of RBFs
    rbfs_sigma      = None  #: The std dev of the RBFs (uniformly selected between [0, dimension width]

    def __init__(self,domain,logger, num_rbfs=None, id=1, state_dimensions=None,
                 const_feature=True, resolution_min=2., resolution_max=None,
                 seed=1, normalize=False, grid_bins=None, include_border=False):

        if resolution_max is None:
            resolution_max = resolution_min

        if state_dimensions is not None:
            dims = len(state_dimensions)
        else: # just consider all dimensions
            state_dimensions = range(domain.state_space_dims)
            dims = domain.state_space_dims

        if grid_bins is not None:
            # uniform grid of rbfs
            self.rbfs_mu, num_rbfs = self.uniformRBFs(grid_bins, include_border)
            self.rbfs_sigma = np.ones((num_rbfs, dims)) * (resolution_max + resolution_min) / 2
        else:
            # uniformly scattered
            assert(num_rbfs is not None)
            self.rbfs_mu        = np.zeros((num_rbfs, dims))
            self.rbfs_sigma     = np.zeros((num_rbfs, dims))
            dim_widths          = (domain.statespace_limits[state_dimensions,1]
                                   - domain.statespace_limits[state_dimensions,0])
            #super(RBF,self).__init__(domain,logger)
            rand_stream = np.random.RandomState(seed=seed)
            for i in np.arange(num_rbfs):
                for d in state_dimensions:
                    self.rbfs_mu[i,d] = rand_stream.uniform(domain.statespace_limits[d,0],
                                                        domain.statespace_limits[d,1])
                    self.rbfs_sigma[i,d] = rand_stream.uniform(dim_widths[d]/ resolution_max, dim_widths[d]/ resolution_min)
        self.const_feature = const_feature
        self.features_num   = num_rbfs
        if const_feature:
            self.features_num += 1 # adds a constant 1 to each feature vector
        self.state_dimensions = state_dimensions
        self.normalize = normalize
        super(RBF, self).__init__(domain, logger)

    def phi_nonTerminal(self, s):
        F_s = np.ones(self.features_num)
        if self.state_dimensions is not None:
            s = s[self.state_dimensions]

        exponent = np.sum(0.5 * ((s - self.rbfs_mu) / self.rbfs_sigma)**2, axis=1)
        if self.const_feature:
            F_s[:-1] = np.exp(-exponent)
        else:
            F_s[:] = np.exp(-exponent)

        if self.normalize and F_s.sum() != 0.:
            F_s /= F_s.sum()
        return F_s

    def uniformRBFs(self,bins_per_dimension, IncludeBorders = False):
        # Positions RBF Centers uniformly across the state space and returns the centers as RBFs-by-dims matrix
        # Each row is a center of an RBF
        # Example: 2D domain where each dimension is in [0,3]
        # with bins = [2,3], False => we get 1 center in the first dimension and 2 centers in the second dimension, hence the combination is:
        # 1.5    1
        # 1.5    2
        # with parameter [2,3], True => we get 3 center in the first dimension and 5 centers in the second dimension, hence the combination is:
        # 0      0
        # 0      1
        # 0      2
        # 0      3
        # 1.5    0
        # 1.5    1
        # 1.5    2
        # 1.5    3
        # 3      0
        # 3      1
        # 3      2
        # 3      3
        # The second output is the total number of rbfs
        #Find centers in each dimension
        domain      = self.domain
        dims        = domain.state_space_dims
        rbfs_num    = np.prod(bins_per_dimension[:]+1) if IncludeBorders else np.prod(bins_per_dimension[:]-1)
        all_centers = []
        for d in np.arange(dims):
            centers = np.linspace(domain.statespace_limits[d, 0],
                               domain.statespace_limits[d,1],
                               bins_per_dimension[d]+1)
            if not IncludeBorders:
                centers = centers[1:-1] #Exclude the beginning and ending
            all_centers.append(centers.tolist())
        #print all_centers
        # Find all pair combinations of them:
        result = perms(all_centers)
        #print result.shape
        return result, rbfs_num

    def featureType(self):
        return float

class QRBF(RBF, QFunRepresentation):
    pass
