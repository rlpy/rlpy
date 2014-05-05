"""Independent Discretization with compact handling of binary features."""

from .Representation import Representation
from rlpy.Tools import *

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class IndependentDiscretizationCompactBinary(Representation):
    # This representation is identical to IndependentDiscretization except when binary features exist in the state-space
    # In such case the feature corresponding to the 0 values of binary dimension are excluded.
    # Furthermore an extra feature is added to the representation which is activated only if all dimensions are binary and non of them are active
    # Based on preliminary mathematical formulation both this representation
    # and the non-compact representation will have the same representational
    # power in the limit.

    def __init__(self, domain, logger, discretization=20):
        # Identify binary dimensions
        self.setBinsPerDimension(domain, discretization)
        nontwobuckets_dims = np.where(self.bins_per_dim != 2)[0]
        self.nonbinary_dims = union1d(
            nontwobuckets_dims,
            domain.continuous_dims)
        self.binary_dims = setdiff1d(
            arange(domain.state_space_dims),
            self.nonbinary_dims)
        self.features_num   = int(sum(self.bins_per_dim)) - \
            len(self.binary_dims) + 1
        # Calculate the maximum id number
        temp_bin_number = copy(self.bins_per_dim)
        temp_bin_number[self.binary_dims] -= 1
        self.maxFeatureIDperDimension = cumsum(temp_bin_number) - 1

        super(
            IndependentDiscretizationCompactBinary,
            self).__init__(
            domain,
            logger,
            discretization)
        if self.logger:
            self.logger.log("Binary Dimensions:\t%s" % str(self.binary_dims))

    def phi_nonTerminal(self, s):
        F_s = zeros(self.features_num, 'bool')
        activeInitialFeatures = self.activeInitialFeaturesCompactBinary(
            s)
        if len(activeInitialFeatures):
            F_s[self.activeInitialFeaturesCompactBinary(s)] = 1
        else:
            F_s[-1] = 1  # Activate the last feature
        return F_s

    def activeInitialFeaturesCompactBinary(self, s):
        # Same as activeInitialFeatures of the parent except that for binary
        # dimensions only the True value will have a corresponding feature
        bs = self.binState(s)
        zero_index = where(bs == 0)[0]
        # Has zero value and is binary dimension
        remove_index = intersect1d(zero_index, self.binary_dims)
        remain_index = setdiff1d(
            arange(self.domain.state_space_dims),
            remove_index)
        # Create a new bin vector where the number of bins are 1 for binary
        temp_bin_number = copy(self.bins_per_dim)
        temp_bin_number[self.binary_dims] -= 1
        # Because activation now is mapped to the first bin which is 0
        bs[self.binary_dims] -= 1
        # print self.bins_per_dim
        # print "MIN:", hstack((0, cumsum(temp_bin_number)[:-1]))
        # print "MAX:", hstack((0, cumsum(temp_bin_number)[:-1]))+temp_bin_number-1
        # print bs
        shifts = hstack((0, cumsum(temp_bin_number)[:-1]))
        index = bs + shifts
        # Remove the corresponding features highlighted by remove_index
        return index[remain_index].astype('uint32')

    def getDimNumber(self, f):
        # Returns the dimension number corresponding to this feature
        dim = searchsorted(self.maxFeatureIDperDimension, f)
        return dim

    def featureType(self):
        return bool
