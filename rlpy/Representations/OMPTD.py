"""OMP-TD implementation based on ICML 2012 paper of Wakefield and Parr."""

from .Representation import Representation
import numpy as np
from .iFDD import iFDD
from rlpy.Tools import className, plt
from copy import deepcopy

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"


class OMPTD(Representation):

    """OMP-TD implementation based on ICML 2012 paper of Wakefield and Parr.

    This implementation assumes an initial representation exists and the bag 
    of features is the conjunctions of existing features.
    OMP-TD uses iFDD to represents its features, yet its discovery method is 
    different; while iFDD looks at the fringe of the tree of expanded features,
    OMPTD only looks through a predefined set of features.

    The set of features used by OMPTD aside from the initial_features are 
    represented by self.expandedFeatures

    """

    # Maximum number of features to be expanded on each iteration
    maxBatchDiscovery = 0
    batchThreshold = 0      # Minimum threshold to add features
    # List of selected features. In this implementation initial features are
    # selected initially by default
    selectedFeatures = None
    remainingFeatures = None  # Array of remaining features

    def __init__(
            self, domain, initial_representation, discretization=20,
            maxBatchDiscovery=1, batchThreshold=0, bagSize=100000, sparsify=False):
        """
        :param domain: the :py:class`~rlpy.Domains.Domain.Domain` associated 
            with the value function we want to learn.
        :param initial_representation: The initial set of features available. 
            OMP-TD does not dynamically introduce any features of its own, 
            instead it takes conjunctions of initial_representation feats until 
            all permutations have been created or bagSize has been reached.
            OMP-TD uses an (ever-growing) subset,termed the \"active\" features.
        :param discretization: Number of bins used for each continuous dimension.
            For discrete dimensions, this parameter is ignored.
        :param maxBatchDiscovery: Maximum number of features to be expanded on 
            each iteration
        :param batchThreshold: Minimum features \"relevance\" required to add 
            a feature to the active set.
        :param bagSize: The maximum number of features available for 
            consideration.
        :param sparsify: (Boolean)
            See :py:class`~rlpy.Representations.iFDD.iFDD`.
        
        """
        
        self.selectedFeatures = []
        # This is dummy since omptd will not use ifdd in the online fashion
        self.iFDD_ONLINETHRESHOLD = 1
        self.maxBatchDiscovery = maxBatchDiscovery
        self.batchThreshold = batchThreshold
        self.initial_representation = initial_representation
        self.iFDD = iFDD(
            domain,
            self.iFDD_ONLINETHRESHOLD,
            initial_representation,
            sparsify=0,
            discretization=discretization,
            useCache=1)
        self.bagSize = bagSize
        self.features_num = self.initial_representation.features_num
        self.isDynamic = True

        super(OMPTD, self).__init__(domain, discretization)

        self.fillBag()
        self.totalFeatureSize = self.bagSize
        # Add initial features to the selected list
        self.selectedFeatures = range(
            self.initial_representation.features_num)
        # Array of indicies of features that have not been selected
        self.remainingFeatures = np.arange(self.features_num, self.bagSize)

    def phi_nonTerminal(self, s):
        F_s = self.iFDD.phi_nonTerminal(s)
        return F_s[self.selectedFeatures]

    def show(self):
        self.logger.info('Features:\t\t%d' % self.features_num)
        self.logger.info('Remaining Bag Size:\t%d' %
                         len(self.remainingFeatures))

    def showBag(self):
        """
        Displays the non-active features that OMP-TD can select from to add 
        to its representation.
        
        """
        print "Remaining Items in the feature bag:"
        for f in self.remainingFeatures:
            print "%d: %s" % (f, str(sorted(list(self.iFDD.getFeature(f).f_set))))

    def calculateFullPhiNormalized(self, states):
        """
        In general for OMPTD it is faster to cache the normalized feature matrix
        at once.  Note this is only valid if possible states do not change over
        execution.  (In the feature matrix, each column is a feature function, 
        each row is a state; thus the matrix has rows phi(s1)', phi(s2)', ...).

        """
        p = len(states)
        self.fullphi = np.empty((p, self.totalFeatureSize))
        o_s = self.domain.state
        for i, s in enumerate(states):
            self.domain.state = s
            if not self.domain.isTerminal(s):
                self.fullphi[i, :] = self.iFDD.phi_nonTerminal(s)
        self.domain.state = o_s
        # Normalize features
        for f in xrange(self.totalFeatureSize):
            phi_f = self.fullphi[:, f]
            norm_phi_f = np.linalg.norm(phi_f)    # L2-Norm of phi_f
            if norm_phi_f == 0:
                norm_phi_f = 1          # This helps to avoid divide by zero
            self.fullphi[:, f] = phi_f / norm_phi_f

    def batchDiscover(self, td_errors, phi, states):
        """
        :param td_errors: p-by-1 vector, error associated with each state
        :param phi: p-by-n matrix, vector-valued feature function evaluated at 
            each state.
        :param states: p-by-(statedimension) matrix, each state under test.
        
        Discovers features using OMPTD
        1. Find the index of remaining features in the bag \n
        2. Calculate the inner product of each feature with the TD_Error vector \n
        3. Add the top maxBatchDiscovery features to the selected features \n
        
        OUTPUT: Boolean indicating expansion of features
        
        """
        if len(self.remainingFeatures) == 0:
            # No More features to Expand
            return False

        SHOW_RELEVANCES = 0      # Plot the relevances
        self.calculateFullPhiNormalized(states)

        relevances = np.zeros(len(self.remainingFeatures))
        for i, f in enumerate(self.remainingFeatures):
            phi_f = self.fullphi[:, f]
            relevances[i] = np.abs(np.dot(phi_f, td_errors))

        if SHOW_RELEVANCES:
            e_vec = relevances.flatten()
            e_vec = e_vec[e_vec != 0]
            e_vec = np.sort(e_vec)
            plt.plot(e_vec, linewidth=3)
            plt.ioff()
            plt.show()
            plt.ion()

        # Sort based on relevances
        # We want high to low hence the reverse: [::-1]
        sortedIndices = np.argsort(relevances)[::-1]
        max_relevance = relevances[sortedIndices[0]]

        # Add top <maxDiscovery> features
        self.logger.debug("OMPTD Batch: Max Relevance = %0.3f" % max_relevance)
        added_feature = False
        to_be_deleted = []  # Record the indices of items to be removed
        for j in xrange(min(self.maxBatchDiscovery, len(relevances))):
            max_index = sortedIndices[j]
            f = self.remainingFeatures[max_index]
            relevance = relevances[max_index]
            # print "Inspecting %s" % str(list(self.iFDD.getFeature(f).f_set))
            if relevance >= self.batchThreshold:
                self.logger.debug(
                    'New Feature %d: %s, Relevance = %0.3f' %
                    (self.features_num, str(np.sort(list(self.iFDD.getFeature(f).f_set))), relevances[max_index]))
                to_be_deleted.append(max_index)
                self.selectedFeatures.append(f)
                self.features_num += 1
                added_feature = True
            else:
                # Because the list is sorted, there is no use to look at the
                # others
                break
        self.remainingFeatures = np.delete(self.remainingFeatures, to_be_deleted)
        return added_feature

    def fillBag(self):
        """
        Generates potential features by taking conjunctions of existing ones.
        Adds these to the bag of features available to OMPTD in a breadth-first
        fashion until the ``bagSize`` limit is reached.
        
        """
        level_1_features = np.arange(
            self.initial_representation.features_num)
        # We store the dimension corresponding to each feature so we avoid
        # adding pairs of features in the same dimension
        level_1_features_dim = {}
        for i in xrange(self.initial_representation.features_num):
            level_1_features_dim[i] = np.array(
                [self.initial_representation.getDimNumber(i)])
            # print i,level_1_features_dim[i]
        level_n_features = np.array(level_1_features)
        level_n_features_dim = deepcopy(level_1_features_dim)
        new_id = self.initial_representation.features_num
        self.logger.debug(
            "Added %d size 1 features to the feature bag." %
            (self.initial_representation.features_num))

        # Loop over possible layers that conjunctions can be add. Notice that
        # layer one was already built
        for f_size in np.arange(2, self.domain.state_space_dims + 1):
            added = 0
            next_features = []
            next_features_dim = {}
            for f in level_1_features:
                f_dim = level_1_features_dim[f][0]
                for g in level_n_features:
                    g_dims = level_n_features_dim[g]
                    if not f_dim in g_dims:
                        # We pass inf to make sure iFDD will add the
                        # combination of these two features
                        added_new_feature = self.iFDD.inspectPair(f, g, np.inf)
                        if added_new_feature:
                            # print '%d: [%s,%s]' % (new_id, str(f),str(g))
                            next_features.append(new_id)
                            next_features_dim[new_id] = g_dims + f_dim
                            new_id += 1
                            added += 1
                            if new_id == self.bagSize:
                                self.logger.debug(
                                    "Added %d size %d features to the feature bag." %
                                    (added, f_size))
                                return
            level_n_features = next_features
            level_n_features_dim = next_features_dim
            self.logger.debug(
                "Added %d size %d features to the feature bag." %
                (added, f_size))
        self.bagSize = new_id

    def featureType(self):
        return self.initial_representation.featureType()
