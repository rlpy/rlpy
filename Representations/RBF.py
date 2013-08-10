#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

######################################################
# Developed by Alborz Geramiard Nov 9th 2012 at MIT #
######################################################
import os
from Tools import *
from Domains import *
from Domains.Pendulum import Pendulum
from Representation import Representation
import numpy as np

class RBF(Representation):
    state_dimensions = None
    rbfs_mu         = None  #: The mean of RBFs
    rbfs_sigma      = None  #: The std dev of the RBFs (uniformly selected between [0, dimension width]

    def __init__(self,domain,logger, num_rbfs=20, id=1, state_dimensions=None,
                 const_feature=True, resolution_min=2., resolution_max=10.,
                 seed=1, normalize=False):
        if isinstance(domain,Pendulum):
            # Put 9 equal in the intersections + 1 extra
            self.domain             = domain
            dims                    = domain.state_space_dims
            bins                    = 4
            includeBorders          = False
            self.rbfs_mu, rbfs      = self.uniformRBFs(array([bins,bins]),includeBorders)
            logger.log('Using 3x3 uniform RBFs => 9 RBFs + 1 constant.')
            self.rbfs_sigma         = ones((num_rbfs,dims))
        elif isinstance(domain,GridWorld):
            # Put 20 equal in the intersections + 1 extra
            self.domain             = domain
            dims                    = domain.state_space_dims
            bins                    = 5
            includeBorders          = True
            self.rbfs_mu, rbfs      = self.uniformRBFs(array([bins,bins]),includeBorders)
            logger.log('Using %dx%d uniform RBFs => %d RBFs + 1 constant.' % (bins+1,bins+1,rbfs))
            dim_widths              = (domain.statespace_limits[:,1]-domain.statespace_limits[:,0])
            self.rbfs_sigma         = empty((num_rbfs,dims))
            for d in arange(dims):
                self.rbfs_sigma[:,d] = dim_widths[d]/3.0
        else:
            self.features_num   = num_rbfs
            if const_feature:
                self.features_num += 1 # adds a constant 1 to each feature vector
            if state_dimensions is not None:
                dims = len(state_dimensions)
            else:
                state_dimensions = range(domain.state_space_dims)
                dims                = domain.state_space_dims
            self.rbfs_mu        = np.zeros((num_rbfs, dims))
            self.rbfs_sigma     = np.zeros((num_rbfs, dims))
            dim_widths          = (domain.statespace_limits[state_dimensions,1]
                                   - domain.statespace_limits[state_dimensions,0])
            super(RBF,self).__init__(domain,logger)
            rand_stream = np.random.RandomState(seed=seed)
            for i in arange(num_rbfs):
                for d in state_dimensions:
                    self.rbfs_mu[i,d] = rand_stream.uniform(domain.statespace_limits[d,0],
                                                        domain.statespace_limits[d,1])
                    self.rbfs_sigma[i,d] = rand_stream.uniform(dim_widths[d]/ resolution_max, dim_widths[d]/ resolution_min)
        self.const_feature = const_feature
        self.state_dimensions = state_dimensions
        self.normalize = normalize
        super(RBF, self).__init__(domain, logger)

    def phi_nonTerminal(self, s):
        F_s = ones(self.features_num)
        if self.state_dimensions is not None:
            s = s[self.state_dimensions]

        exponent = sum(0.5 * ((s - self.rbfs_mu) / self.rbfs_sigma)**2, axis=1)
        if self.const_feature:
            F_s[:-1] = exp(-exponent)
        else:
            F_s[:] = exp(-exponent)

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
        rbfs_num    = prod(bins_per_dimension[:]+1) if IncludeBorders else prod(bins_per_dimension[:]-1)
        all_centers = []
        for d in arange(dims):
            centers = linspace(domain.statespace_limits[d, 0],
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

