"""Independent Discretization with compact handling of binary features."""

from .Representation import Representation
import numpy as np
from copy import copy

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class IndependentDiscretizationCompactBinary(Representation):

    """
    Compact tabular representation with linearly independent basis functions.

    This representation is identical to IndependentDiscretization except when binary features exist in the state-space
    In such case the feature corresponding to the 0 values of binary dimension are excluded.
    Furthermore an extra feature is added to the representation which is activated only if all dimensions are binary and non of them are active
    Based on preliminary mathematical formulation both this representation
    and the non-compact representation will have the same representational
    power in the limit.

    """

    def __init__(self, domain, discretization=20):
        # See superclass __init__ definition
        self.setBinsPerDimension(domain, discretization)
        nontwobuckets_dims = np.where(self.bins_per_dim != 2)[0]
        self.nonbinary_dims = np.union1d(
            nontwobuckets_dims,
            domain.continuous_dims)
        self.binary_dims = np.setdiff1d(
            np.arange(domain.state_space_dims),
            self.nonbinary_dims)
        self.features_num   = int(sum(self.bins_per_dim)) - \
            len(self.binary_dims) + 1
        # Calculate the maximum id number
        temp_bin_number = copy(self.bins_per_dim)
        temp_bin_number[self.binary_dims] -= 1
        self.maxFeatureIDperDimension = np.cumsum(temp_bin_number) - 1

        super(
            IndependentDiscretizationCompactBinary,
            self).__init__(
            domain,
            discretization)

    def phi_nonTerminal(self, s):
        F_s = np.zeros(self.features_num, 'bool')
        activeInitialFeatures = self.activeInitialFeaturesCompactBinary(
            s)
        if len(activeInitialFeatures):
            F_s[self.activeInitialFeaturesCompactBinary(s)] = 1
        else:
            F_s[-1] = 1  # Activate the last feature
        return F_s

    def activeInitialFeaturesCompactBinary(self, s):
        """
        Same as :py:meth:`~rlpy.Representations.Representation.activeInitialFeatures`
        except that for binary dimensions (taking values 0,1) only the 
        ``1`` value will have a corresponding feature; ``0`` is expressed by 
        that feature being inactive.
        
        """
        bs = self.binState(s)
        zero_index = np.where(bs == 0)[0]
        # Has zero value and is binary dimension
        remove_index = np.intersect1d(zero_index, self.binary_dims)
        remain_index = np.setdiff1d(
            np.arange(self.domain.state_space_dims),
            remove_index)
        # Create a new bin vector where the number of bins are 1 for binary
        temp_bin_number = copy(self.bins_per_dim)
        temp_bin_number[self.binary_dims] -= 1
        # Because activation now is mapped to the first bin which is 0
        bs[self.binary_dims] -= 1
        shifts = np.hstack((0, np.cumsum(temp_bin_number)[:-1]))
        index = bs + shifts
        # Remove the corresponding features highlighted by remove_index
        return index[remain_index].astype('uint32')

    def getDimNumber(self, f):
        """ Returns the dimension number corresponding to feature ``f``. """
        dim = np.searchsorted(self.maxFeatureIDperDimension, f)
        return dim

    def featureType(self):
        return bool
