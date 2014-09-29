"""Independent Discretization"""

from .Representation import Representation
import numpy as np

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class IndependentDiscretization(Representation):
    """
    Creates a feature for each discrete bin in each dimension; the feature 
    vector for a given state is comprised of binary features, where only the 
    single feature in a particular dimension is 1, all others 0.
    I.e., in a particular state, the sum of all elements of a feature vector
    equals the number of dimensions in the state space.

    Note that This is the minimum number of binary features required to 
    uniquely represent a state in a given finite discrete domain.

    """

    def __init__(self, domain, discretization=20):
        self.setBinsPerDimension(domain, discretization)
        self.features_num = int(sum(self.bins_per_dim))
        self.maxFeatureIDperDimension = np.cumsum(self.bins_per_dim) - 1
        super(
            IndependentDiscretization,
            self).__init__(
            domain,
            discretization)

    def phi_nonTerminal(self, s):
        F_s = np.zeros(
            self.features_num,
            'bool')
        F_s[self.activeInitialFeatures(s)] = 1
        return F_s

    def getDimNumber(self, f):
        # Returns the dimension number corresponding to this feature
        dim = np.searchsorted(self.maxFeatureIDperDimension, f)
        return dim

    def getFeatureName(self, feat_id):
        if hasattr(self.domain, 'DimNames'):
            dim = np.searchsorted(self.maxFeatureIDperDimension, feat_id)
            # Find the index of the feature in the corresponding dimension
            index_in_dim = feat_id
            if dim != 0:
                index_in_dim = feat_id - self.maxFeatureIDperDimension[dim - 1]
            print self.domain.DimNames[dim]
            f_name = self.domain.DimNames[dim] + '=' + str(index_in_dim)

    def featureType(self):
        return bool
