"""Bellman-Error Basis Function Representation."""

#from rlpy.Tools import
import numpy as np
from .Representation import Representation
from rlpy.Tools import svm

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Robert H. Klein"


class BEBF(Representation):

    """Bellman-Error Basis Function Representation

    REQUIRES the implementation of locally-weighted
    projection regression (LWPR), available at:
    http://wcms.inf.ed.ac.uk/ipab/slmc/research/software-lwpr

    Parameters set according to:
    Parr et al. 'Analyzing Feature Generation for Function Approximation'

    Bellman-Error Basis Function Representation.
    1. Initial basis function based on immediate reward.
    2. Evaluate r + Q(s', \pi{s'}) - Q(s,a) for all samples.
    3. Train function approximator on bellman error of present solution above
    4. Add the above as a new basis function.
    5. Repeat the process using the new basis until the most
    recently added basis function has norm <= batchThreshold, which
    Parr et al. used as 10^-5.
    """

    debug = 0
    # Number of features to be expanded in the batch setting; here 1 since
    # each BEBF will be identical on a given iteration
    maxBatchDicovery = 1
    # from sklearn: "epsilon in the epsilon-SVR model. It specifies the
    # epsilon-tube within which no penalty is associated in the training loss
    # function with points predicted within a distance epsilon from the actual
    # value."
    svm_epsilon = None
    # Array of pointers to feature functions, indexed by order created
    features = []
    # Minimum value of feature relevance for the batch setting (10^-5 per Parr
    # et al.
    batchThreshold = None
    # Initial number of features, initialized in __init__
    initial_features_num = 0

    def __init__(self, domain, discretization=20,
                 debug=0, batchThreshold=10 ** -3, svm_epsilon=.1):
        self.setBinsPerDimension(domain, discretization)
        # Effectively initialize with IndependentDiscretization
        self.initial_features_num = int(sum(self.bins_per_dim))
        # Starting number of features equals the above, changes during
        # execution
        self.features_num = self.initial_features_num
       # self.features_num           = 0
        self.debug = debug
        self.svm_epsilon = svm_epsilon
        self.batchThreshold = batchThreshold
        self.addInitialFeatures()
        super(BEBF, self).__init__(domain, discretization)
        self.isDynamic = True
        # @return: a function object corresponding to the

    def getFunctionApproximation(self, X, y):
        # bebfApprox = svm.SVR(kernel='rbf', degree=3, C=1.0, epsilon = 0.0005) # support vector regression
                                                 # C = penalty parameter of
                                                 # error term, default 1
        bebfApprox = svm.SVR(
            kernel='rbf',
            degree=3,
            C=1.0,
            epsilon=self.svm_epsilon)
        bebfApprox.fit(X, y)
        return bebfApprox

    def addInitialFeatures(self):
        pass

    def phi_nonTerminal(self, s):
        F_s = np.zeros(self.features_num)
        # From IndependentDiscretization
        F_s[self.activeInitialFeatures(s)] = 1
        bebf_features_num = self.features_num - self.initial_features_num
        for features_ind, F_s_ind in enumerate(np.arange(bebf_features_num) + self.initial_features_num):
            F_s[F_s_ind] = self.features[features_ind].predict(s)
#        print 's,F_s',s,F_s
#        shout('F_s:',F_s)
        return F_s

    # Adds new features based on the Bellman Error in batch setting.
    # @param td_errors: p-by-1 (How much error observed for each sample)
    # @param all_phi_s: n-by-p features corresponding to all samples (each column corresponds to one sample)
    # @param s: List of states corresponding to each td_error in td_errors (note that the same state may appear multiple times because of different actions taken while there)
    # self.batchThreshold is threshold below which no more BEBFs are added.
    def batchDiscover(self, td_errors, all_phi_s, s):
        # need states here instead?
        addedFeature = False
        # PLACEHOLDER for norm of function
        norm = max(abs(td_errors))  # Norm of function
        for j in xrange(self.maxBatchDicovery):
            self.features.append(self.getFunctionApproximation(s, td_errors))
            if norm > self.batchThreshold:
                self.addNewWeight()
                addedFeature = True
                self.features_num += 1
                self.logger.debug(
                    'Added feature. \t %d total feats' %
                    self.features_num)
            else:
                break
        return addedFeature

    def featureType(self):
        return float
