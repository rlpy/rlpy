"""Independent Discretization"""

from .Representation import Representation, QFunRepresentation
import numpy as np
from Tools import union1d, setdiff1d, intersect1d
from copy import copy

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class IndependentDiscretization(Representation):

    def __init__(self,domain,logger,discretization = 20):
        self.setBinsPerDimension(domain,discretization)
        self.features_num = int(sum(self.bins_per_dim))
        self.maxFeatureIDperDimension = np.cumsum(self.bins_per_dim)-1
        super(IndependentDiscretization,self).__init__(domain,logger,discretization)

    def phi_nonTerminal(self,s):
        F_s = np.zeros(self.features_num,'bool')
        F_s[self.activeInitialFeatures(s)]  = 1
        return F_s

    def getDimNumber(self,f):
        # Returns the dimension number corresponding to this feature
        dim     = searchsorted(self.maxFeatureIDperDimension,f)
        return dim

    def getFeatureName(self,id):
        if hasattr(self.domain, 'DimNames'):
            dim     = np.searchsorted(self.maxFeatureIDperDimension,id)
            #Find the index of the feature in the corresponding dimension
            index_in_dim = id
            if dim != 0:
                index_in_dim = id - self.maxFeatureIDperDimension[dim-1]
            print self.domain.DimNames[dim]

    def featureType(self):
        return bool


class QIndependentDiscretization(IndependentDiscretization, QFunRepresentation):
    pass


class IndependentDiscretizationCompact(Representation):
    """This representation is identical to IndependentDiscretization except when binary features exist in
    the state-space
    In such case the feature corresponding to the 0 values of binary dimension are excluded.
    Furthermore an extra feature is added to the representation which is activated only if all
    dimensions are binary and non of them are active

    Based on preliminary mathematical formulation both this representation and the non-compact
    representation will have the same representational power in the limit.
    """
    def __init__(self,domain,logger,discretization = 20):
        # Identify binary dimensions
        self.setBinsPerDimension(domain,discretization)
        nontwobuckets_dims  = np.where(self.bins_per_dim != 2)[0]
        self.nonbinary_dims = union1d(nontwobuckets_dims,domain.continuous_dims)
        self.binary_dims    = setdiff1d(np.arange(domain.state_space_dims),self.nonbinary_dims)
        self.features_num   = int(np.sum(self.bins_per_dim)) - len(self.binary_dims) + 1
        #Calculate the maximum id number
        temp_bin_number = copy(self.bins_per_dim)
        temp_bin_number[self.binary_dims] -= 1
        self.maxFeatureIDperDimension = np.cumsum(temp_bin_number)-1

        super(IndependentDiscretizationCompact,self).__init__(domain,logger,discretization)
        if self.logger:
            self.logger.log("Binary Dimensions:\t%s"% str(self.binary_dims))

    def phi_nonTerminal(self,s):
        F_s = np.zeros(self.features_num,'bool')
        activeInitialFeatures = self.activeInitialFeaturesCompactBinary(s)
        if len(activeInitialFeatures):
            F_s[self.activeInitialFeaturesCompactBinary(s)] = 1
        else:
            F_s[-1] = 1 # Activate the last feature
        return F_s

    def activeInitialFeaturesCompactBinary(self,s):
        #Same as activeInitialFeatures of the parent except that for binary dimensions only the True value will have a corresponding feature
        bs              = self.binState(s)
        zero_index      = np.where(bs == 0)[0]
        remove_index    = intersect1d(zero_index,self.binary_dims) # Has zero value and is binary dimension
        remain_index    = setdiff1d(np.arange(self.domain.state_space_dims),remove_index)
        # Create a new bin vector where the number of bins are 1 for binary
        temp_bin_number = copy(self.bins_per_dim)
        temp_bin_number[self.binary_dims] -= 1
        bs[self.binary_dims] -= 1 # Because activation now is mapped to the first bin which is 0
        shifts          = np.hstack((0, np.cumsum(temp_bin_number)[:-1]))
        index           = bs + shifts
        # Remove the corresponding features highlighted by remove_index
        return      index[remain_index].astype('uint32')

    def getDimNumber(self,f):
        # Returns the dimension number corresponding to this feature
        dim     = np.searchsorted(self.maxFeatureIDperDimension,f)
        return dim

    def featureType(self):
        return bool


class QIndependentDiscretizationCompact(IndependentDiscretizationCompact, QFunRepresentation):
    pass
