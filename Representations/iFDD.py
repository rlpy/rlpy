"""Incremental Feature Dependency Discovery"""
# iFDD implementation based on ICML 2011 paper

from copy import deepcopy
import numpy as np
from collections import defaultdict
from Tools import *
from .Representation import Representation, QFunRepresentation

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"

class iFDD_feature(object):
    index   = None        #Unique index Corresponding to its location in a vector (0 based!)
    f_set   = None        #A set of basic features that this feature corresponds to
    p1      = None        #Parent 1 feature index *Basic features have -1 for both nodes
    p2      = None        #Parent 2 feature index *Basic features have -1 for both nodes

    def __init__(self,potential):
        if isinstance(potential,iFDD_potential):
            self.index  = potential.index
            self.f_set  = deepcopy(potential.f_set)
            self.p1     = potential.p1
            self.p2     = potential.p2
        else:
            # This is the case where a single integer is passed to generate an initial feature
            index           = potential
            self.index      = index
            self.f_set      = frozenset([index])
            self.p1         = -1
            self.p2         = -1

    def __deepcopy__(self,memo):
        new_f       = iFDD_feature(self.index)
        new_f.p1    = self.p1
        new_f.p2    = self.p2
        new_f.f_set = deepcopy(self.f_set)
        return new_f

    def show(self):
        printClass(self)


class iFDD_potential(object):
    relevance = None     # Sum of errors corresponding to this feature
    f_set     = None     # Set of features it corresponds to [Immutable]
    index     = None     # If this feature has been discovered set this to its index else 0
    p1        = None     # Parent 1 feature index
    p2        = None     # Parent 2 feature index
    count     = None     # Number of times this feature was visited
    def __init__(self, f_set, parent1, parent2):
        self.f_set      = deepcopy(f_set)
        self.index      = -1 # -1 means it has not been discovered yet
        self.p1         = parent1
        self.p2         = parent2
        self.cumtderr   = 0
        self.cumabstderr= 0
        self.count      = 1
    def __deepcopy__(self,memo):
        new_p       = iFDD_potential(self.f_set,self.p1,self.p2)
        new_p.cumtderr = self.cumtderr
        new_p.cumabstderr = self.cumabstderr
        return new_p
    def show(self):
        printClass(self)


class iFDD(Representation):
    PRINT_MAX_RELEVANCE     = False # It is a good starting point to see how relevances grow if threshold is set to infinity.
    discovery_threshold     = None  # psi in the paper
    sparsify                = None  # boolean specifying the use of the trick mentioned in the paper so that features are getting sparser with more feature discoveries (i.e. use greedy algorithm for feature activation)
    iFDD_features           = None    # dictionary mapping initial feature sets to iFDD_feature
    iFDD_potentials         = None    # dictionary mapping initial feature sets to iFDD_potential
    featureIndex2feature    = None    # dictionary mapping each feature index (ID) to its feature object
    debug                   = 0     # Print more stuff
    cache                   = None    # dictionary mapping  initial active feature set phi_0(s) to its corresponding active features at phi(s). Based on Tuna's Trick to speed up iFDD
    useCache                = 0     # this should only increase speed. If results are different something is wrong
    maxBatchDicovery        = 0     # Number of features to be expanded in the batch setting
    batchThreshold          = 0     # Minimum value of feature relevance for the batch setting
    iFDDPlus                = 0     # ICML 11 iFDD would add sum of abs(TD-errors) while the iFDD plus uses the abs(sum(TD-Error))/sqrt(potential feature presence count)
    sortediFDDFeatures      = None  # This is a priority queue based on the size of the features (Largest -> Smallest). For same size features, it is also sorted based on the newest -> oldest. Each element is the pointer to feature object.
    initial_representation  = None  # A Representation that provides the initial set of features for iFDD
    maxRelevance            = -inf  # Helper parameter to get a sense of appropriate threshold on the relevance for discovery
    use_chirstoph_ordered_features = True # As Christoph mentioned adding new features may affect the phi for all states. This idea was to make sure both conditions for generating active features generate the same result.

    def __init__(self,domain,logger,discovery_threshold, initial_representation,
                 sparsify = True, discretization = 20,debug = 0,useCache = 0,
                 maxBatchDicovery = 1, batchThreshold = 0,iFDDPlus = 1):
        self.iFDD_features          = {}
        self.iFDD_potentials        = {}
        self.featureIndex2feature   = {}
        self.cache                  = {}
        self.discovery_threshold    = discovery_threshold
        self.sparsify               = sparsify
        self.setBinsPerDimension(domain,discretization)
        self.features_num           = initial_representation.features_num
        self.debug                  = debug
        self.useCache               = useCache
        self.maxBatchDicovery       = maxBatchDicovery
        self.batchThreshold         = batchThreshold
        self.sortediFDDFeatures     = PriorityQueueWithNovelty()
        self.initial_representation = initial_representation
        self.iFDDPlus               = iFDDPlus
        self.isDynamic              = True
        self.addInitialFeatures()
        super(iFDD,self).__init__(domain,logger,discretization)
        if self.logger:
            self.logger.log("Initial Representation:\t%s"% className(self.initial_representation))
            self.logger.log("Plus:\t\t\t%d" % self.iFDDPlus)
            self.logger.log("Sparsify:\t\t%d"% self.sparsify)
            self.logger.log("Cached:\t\t\t%d"% self.useCache)
            self.logger.log("Online Threshold:\t%0.3f" % self.discovery_threshold)
            self.logger.log("Batch Threshold:\t\t%0.3f"% self.batchThreshold)
            self.logger.log("Max Batch Discovery:\t%d"% self.maxBatchDicovery)

    def phi_nonTerminal(self,s):
        """ Based on Tuna's Master Thesis 2012 """
        F_s                     = zeros(self.features_num,'bool')
        F_s_0                   = self.initial_representation.phi_nonTerminal(s)
        activeIndices           = where(F_s_0 != 0)[0]
        if self.useCache:
            finalActiveIndices     = self.cache.get(frozenset(activeIndices))
            if finalActiveIndices is None:
                # run regular and update the cache
                finalActiveIndices     = self.findFinalActiveFeatures(activeIndices)
        else:
            finalActiveIndices         = self.findFinalActiveFeatures(activeIndices)
        F_s[finalActiveIndices] = 1
        return F_s

    def findFinalActiveFeatures(self,intialActiveFeatures):
        """
        Given the active indices of phi_0(s) find the final active indices of phi(s) based on discovered features
        """
        finalActiveFeatures = []
        k = len(intialActiveFeatures)
        initialSet          = set(intialActiveFeatures)


        if 2**k <= self.features_num:
            # k can be big which can cause this part to be very slow
            # if k is large then find active features by enumerating on the
            # discovered features.
            if self.use_chirstoph_ordered_features:
                for i in range(k, 0, -1):
                    if len(initialSet) == 0:
                        break
                    # generate list of all combinations with i elements
                    cand_i = [(c, self.iFDD_features[frozenset(c)].index)
                                for c in combinations(initialSet, i)
                                    if frozenset(c) in self.iFDD_features]
                    # sort (recent features (big ids) first)
                    cand_i.sort(key=lambda x: x[1], reverse=True)
                    #idx = -1
                    for candidate, ind in cand_i:
                        # the next block is for testing only
                        #cur_idx = self.iFDD_features[frozenset(candidate)].index
                        #if idx > 0:
                        #    assert(idx > cur_idx)
                        #idx = cur_idx

                        if len(initialSet) == 0:
                            break # No more initial features to be mapped to extended ones

                        if initialSet.issuperset(set(candidate)): # This was missing from ICML 2011 paper algorithm. Example: [0,1,20], [0,20] is discovered, but if [0] is checked before [1] it will be added even though it is already covered by [0,20]
                            feature = self.iFDD_features.get(frozenset(candidate))
                            if feature is not None:
                                finalActiveFeatures.append(feature.index)
                                if self.sparsify:
                                    #print "Sets:", initialSet, feature.f_set
                                    initialSet   = initialSet - feature.f_set
                                    #print "Remaining Set:", initialSet
            else:
                for candidate in powerset(initialSet,ascending=0):
                    if len(initialSet) == 0:
                        break # No more initial features to be mapped to extended ones

                    if initialSet.issuperset(set(candidate)): # This was missing from ICML 2011 paper algorithm. Example: [0,1,20], [0,20] is discovered, but if [0] is checked before [1] it will be added even though it is already covered by [0,20]
                        feature = self.iFDD_features.get(frozenset(candidate))
                        if feature != None:
                            finalActiveFeatures.append(feature.index)
                            if self.sparsify:
                                #print "Sets:", initialSet, feature.f_set
                                initialSet   = initialSet - feature.f_set
                                #print "Remaining Set:", initialSet
        else:
            #print "********** Using Alternative: %d > %d" % (2**k, self.features_num)
            # Loop on all features sorted on their size and then novelty and activate features
            for feature in self.sortediFDDFeatures.toList():
                if len(initialSet) == 0:
                    break # No more initial features to be mapped to extended ones

                if initialSet.issuperset(set(feature.f_set)):
                    finalActiveFeatures.append(feature.index)
                    if self.sparsify:
                        #print "Sets:", initialSet, feature.f_set
                        initialSet   = initialSet - feature.f_set
                        #print "Remaining Set:", initialSet

        if self.useCache:
            self.cache[frozenset(intialActiveFeatures)] = finalActiveFeatures
        return finalActiveFeatures

    def post_discover(self, s, terminal, a, td_error, phi_s, rho=1):
        """
        returns the number of added features
        """


        activeFeatures = phi_s.nonzero()[0] # Indices of non-zero elements of vector phi_s
        discovered = 0
        for g_index,h_index in combinations(activeFeatures,2):
            discovered += self.inspectPair(g_index,h_index,td_error)
        return discovered

    def inspectPair(self,g_index,h_index,td_error):
        # Inspect feature f = g union h where g_index and h_index are the indices of features g and h
        # If the relevance is > Threshold add it to the list of features
        # Returns True if a new feature is added
        g  = self.featureIndex2feature[g_index].f_set
        h  = self.featureIndex2feature[h_index].f_set
        f           = g.union(h)
        feature     = self.iFDD_features.get(f)
        if not self.iFDDPlus: td_error = abs(td_error)

        if feature is not None:
            #Already exists
            return False

        #Look it up in potentials
        potential = self.iFDD_potentials.get(f)
        if potential is None:
            # Generate a new potential and put it in the dictionary
            potential = iFDD_potential(f, g_index, h_index)
            self.iFDD_potentials[f] = potential

        potential.cumtderr += td_error
        potential.cumabstderr += abs(td_error)
        potential.count     += 1
        # Check for discovery

        if np.random.rand() < self.iFDDPlus:
            relevance = abs(potential.cumtderr) / np.sqrt(potential.count)
        else:
            relevance = potential.cumabstderr

        if relevance >= self.discovery_threshold:
            self.maxRelevance = -inf
            self.addFeature(potential)
            return True
        else:
            self.updateMaxRelevance(relevance)
            return False

    def show(self):
        self.showFeatures()
        self.showPotentials()
        self.showCache()

    def updateWeight(self,p1_index,p2_index):
        if self.sparsify:
            newElem = np.atleast_2d(self.theta[p1_index] + self.theta[p2_index])
        else:
            newElem =  None
        self.theta      = addNewElementForAllActions(self.theta,1,newElem)
        self.hashed_s   = None # We dont want to reuse the hased phi because phi function is changed!

    def addInitialFeatures(self):
        for i in arange(self.initial_representation.features_num):
            feature = iFDD_feature(i)
            #shout(self,self.iFDD_features[frozenset([i])].index)
            self.iFDD_features[frozenset([i])] = feature
            self.featureIndex2feature[feature.index] = feature
            priority = 1 # priority is 1/number of initial features corresponding to the feature
            self.sortediFDDFeatures.push(priority,feature)

    def addFeature(self,potential):
        # Add it to the list of features
        potential.index                     = self.features_num # Features_num is always one more than the max index (0-based)
        self.features_num                   += 1
        feature                             = iFDD_feature(potential)
        self.iFDD_features[potential.f_set] = feature
        #Expand the size of the theta
        self.updateWeight(feature.p1,feature.p2)
        # Update the index to feature dictionary
        self.featureIndex2feature[feature.index] = feature
        # print "IN IFDD, New Feature = %d => Total Features = %d" % (feature.index, self.features_num)
        # Update the sorted list of features
        priority = 1/(len(potential.f_set)*1.) # priority is 1/number of initial features corresponding to the feature
        self.sortediFDDFeatures.push(priority,feature)

        #If you use cache, you should invalidate entries that their initial set contains the set corresponding to the new feature
        if self.useCache:
            for initialActiveFeatures in self.cache.keys():
                if initialActiveFeatures.issuperset(feature.f_set):
                    if self.sparsify:
                        self.cache.pop(initialActiveFeatures)
                    else:
                        # If sparsification is not used, simply add the new feature id to all cached values that have feature set which is a super set of the features corresponding to the new discovered feature
                        self.cache[initialActiveFeatures].append(feature.index)
        if self.debug: self.show()

    def batchDiscover(self,td_errors, phi, states):
        # Discovers features using iFDD in batch setting.
        # TD_Error: p-by-1 (How much error observed for each sample)
        # phi: n-by-p features corresponding to all samples (each column corresponds to one sample)
        # self.batchThreshold is the minimum relevance value for the feature to be expanded
        SHOW_PLOT       = 0      #Shows the histogram of relevances
        maxDiscovery    = self.maxBatchDicovery
        n               = self.features_num #number of features
        p               = len(td_errors)     #Number of samples
        counts          = zeros((n,n))
        relevances      = zeros((n,n))
        for i in arange(p):
            phiphiT     = outer(phi[i,:],phi[i,:])
            if self.iFDDPlus:
                relevances  += phiphiT*td_errors[i]
            else:
                relevances  += phiphiT*abs(td_errors[i])
            counts      += phiphiT
        #Remove Diagonal and upper part of the relevances as they are useless
        relevances      = triu(relevances,1)
        non_zero_index  = nonzero(relevances)
        if self.iFDDPlus:
            relevances[non_zero_index] = divide(abs(relevances[non_zero_index]), sqrt(counts[non_zero_index])) #Calculate relevances based on theoretical results of ICML 2013 potential submission
        else:
            relevances[non_zero_index] = relevances[non_zero_index] #Based on Geramifard11_ICML Paper

        #Find indexes to non-zero excited pairs
        (F1, F2)            = relevances.nonzero() # F1 and F2 are the parents of the potentials
        relevances          = relevances[F1,F2]
        if len(relevances) == 0:
            # No feature to add
            self.logger.log("iFDD Batch: Max Relevance = 0")
            return False

        if SHOW_PLOT:
            e_vec  = relevances.flatten()
            e_vec  = e_vec[e_vec != 0]
            e_vec  = sort(e_vec)
            pl.ioff()
            pl.plot(e_vec,linewidth=3)
            pl.show()

        #Sort based on relevances
        sortedIndices  = argsort(relevances)[::-1] # We want high to low hence the reverse: [::-1]
        max_relevance   = relevances[sortedIndices[0]]
        #Add top <maxDiscovery> features
        self.logger.log("iFDD Batch: Max Relevance = {0:g}".format(max_relevance))
        added_feature = False
        new_features = 0
        for j in arange(len(relevances)):
            if new_features >= maxDiscovery:
                break
            max_index   = sortedIndices[j]
            f1          = F1[max_index]
            f2          = F2[max_index]
            relevance   = relevances[max_index]
            if relevance > self.batchThreshold:
                #print "Inspecting", f1,f2,'=>',self.getStrFeatureSet(f1),self.getStrFeatureSet(f2)
                if self.inspectPair(f1, f2, inf):
                    self.logger.log('New Feature %d: %s, Relevance = %0.3f' % (self.features_num-1, self.getStrFeatureSet(self.features_num-1),relevances[max_index]))
                    new_features += 1
                    added_feature = True
            else:
                #Because the list is sorted, there is no use to look at the others
                break
        return added_feature # A signal to see if the representation has been expanded or not

    def showFeatures(self):
        print "Features:"
        print join(["-"]*30)
        print " index\t| f_set\t| p1\t| p2\t | Weights (per action)"
        print join(["-"]*30)
        for feature in reversed(self.sortediFDDFeatures.toList()):
        #for feature in self.iFDD_features.itervalues():
            #print " %d\t| %s\t| %s\t| %s\t| %s" % (feature.index,str(list(feature.f_set)),feature.p1,feature.p2,str(self.theta[feature.index::self.features_num]))
            print " %d\t| %s\t| %s\t| %s\t| Omitted" % (feature.index,self.getStrFeatureSet(feature.index),feature.p1,feature.p2)

    def showPotentials(self):
        print "Potentials:"
        print join(["-"]*30)
        print " index\t| f_set\t| relevance\t| count\t| p1\t| p2"
        print join(["-"]*30)
        for _,potential in self.iFDD_potentials.iteritems():
            print " %d\t| %s\t| %0.2f\t| %d\t| %s\t| %s" % (potential.index,str(sort(list(potential.f_set))),potential.relevance,potential.count,potential.p1,potential.p2)

    def showCache(self):
        if self.useCache:
            print "Cache:"
            if len(self.cache) == 0:
                print 'EMPTY!'
                return
            print join(["-"]*30)
            print " initial\t| Final"
            print join(["-"]*30)
            for initial,active in self.cache.iteritems():
                print " %s\t| %s" % (str(list(initial)), active)

    def updateMaxRelevance(self, newRelevance):
        # Update a global max relevance and outputs it if it is updated
        if self.maxRelevance < newRelevance:
            self.maxRelevance = newRelevance
            if self.PRINT_MAX_RELEVANCE:
                self.logger.log("iFDD Batch: Max Relevance = {0:g}".format(newRelevance))

    def getFeature(self,f_id):
        #returns a feature given a feature id
        if f_id in self.featureIndex2feature.keys():
            return self.featureIndex2feature[f_id]
        else:
            print "F_id %d is not valid" % f_id
            return None

    def getStrFeatureSet(self,f_id):
        #returns a string that corresponds to the set of features specified by the given feature_id
        if f_id in self.featureIndex2feature.keys():
            return str(sorted(list(self.featureIndex2feature[f_id].f_set)))
        else:
            print "F_id %d is not valid" % f_id
            return None

    def featureType(self):
        return bool

    def __deepcopy__(self,memo):
        ifdd = iFDD(self.domain,self.logger,self.discovery_threshold, self.initial_representation, self.sparsify, self.discretization, self.debug, self.useCache,self.maxBatchDicovery, self.batchThreshold, self.iFDDPlus)
        for s,f in self.iFDD_features.items():
            new_f = deepcopy(f)
            new_s = deepcopy(s)
            ifdd.iFDD_features[new_s] = new_f
            ifdd.featureIndex2feature[new_f.index] = new_f
        for s,p in self.iFDD_potentials.items():
            new_s = deepcopy(s)
            new_p = deepcopy(p)
            ifdd.iFDD_potentials[new_s] = deepcopy(new_p)
        ifdd.cache = deepcopy(self.cache)
        ifdd.sortediFDDFeatures = deepcopy(self.sortediFDDFeatures)
        ifdd.features_num = self.features_num
        ifdd.theta = deepcopy(self.theta)
        return ifdd

class QiFDD(iFDD, QFunRepresentation):

    def updateWeight(self,p1_index,p2_index):

        # Add a new weight corresponding to the new added feature for all actions.
        # The new weight is set to zero if sparsify = False, and equal to the sum of weights corresponding to the parents if sparsify = True
        a = self.domain.actions_num
        f = self.features_num-1 # Number of feature before adding the new one
        if self.sparsify:
            newElem = (self.theta[p1_index::f] + self.theta[p2_index::f]).reshape((-1,1))
        else:
            newElem =  None
        self.theta      = addNewElementForAllActions(self.theta,a,newElem)
        self.hashed_s   = None # We dont want to reuse the hased phi because phi function is changed!


class iFDDK_potential(iFDD_potential):
    f_set     = None     # Set of features it corresponds to [Immutable]
    index     = None     # If this feature has been discovered set this to its index else 0
    p1        = None     # Parent 1 feature index
    p2        = None     # Parent 2 feature index
    a = 0.   # tE[phi |\delta|] estimate
    b = 0.   # tE[phi \delta] estimate
    c = 0.   # || phi ||^2_d estimate
    n_crho = 0  # rho episode index of last update
    e = 0.   # eligibility trace

    nu = 0.  # w value of last statistics update
    x_a = 0.  # y_a value of last statistics update
    x_b = 0.  # y_b value of last stistics update
    l = 0 # t value of last statistics update

    def relevance(self, kappa=None, plus=None):
        if plus is None:
            assert(kappa is not None)
            plus = np.random.rand() >= kappa

        if plus:
            return np.abs(self.b) / np.sqrt(self.c)
        else:
            return self.a / np.sqrt(self.c)


    def update_statistics(self, rho, td_error, lambda_, gamma, phi_s, n_rho):
        phi = phi_s[self.p1] * phi_s[self.p2]
        if n_rho > self.n_crho:
            self.e = 0
        else:
            self.e = rho * (lambda_ * gamma * self.e + phi)

        self.a += np.abs(td_error) * self.e
        self.b += td_error * self.e
        self.c += phi**2

        self.n_crho = n_rho

    def update_lazy_statistics(self, rho, td_error, lambda_, gamma, phi_s, y_a, y_b, t_rho, w, t, n_rho):
        phi = phi_s[self.p1] * phi_s[self.p2]
        # catch up on old updates
        gl = np.power(gamma * lambda_,t_rho[n_rho] - self.l)
        self.a += self.e * (y_a[self.n_crho] - self.x_a) * np.exp(- self.nu) * gl
        self.b += self.e * (y_b[self.n_crho] - self.x_b) * np.exp(- self.nu) * gl
        if n_rho > self.n_crho:
            self.e = 0
            # TODO clean up y_a and y_b
        else:
            np.seterr(under="warn")
            self.e *= gl * (lambda_ * gamma)**(t - 1 - t_rho[n_rho]) * np.exp(w - self.nu)
            np.seterr(under="raise")
        # updates based on current transition
        np.seterr(under="warn")
        self.e = rho * (lambda_ * gamma * self.e + phi)

        self.a += np.abs(td_error) * self.e
        self.b += td_error * self.e
        np.seterr(under="raise")
        self.c += phi**2
        # save current values for next catch-up update
        self.l = t
        self.x_a = y_a[n_rho]
        self.x_b = y_b[n_rho]
        self.nu = w
        self.n_crho = n_rho

    def __init__(self, f_set, parent1, parent2):
        self.f_set      = deepcopy(f_set)
        self.index      = -1  # -1 means it has not been discovered yet
        self.p1         = parent1
        self.p2         = parent2

    def __deepcopy__(self, memo):
        new_p = iFDD_potential(self.f_set, self.p1, self.p2)
        for i in ["a", "b", "c", "n_crho", "e", "nu", "x_a", "x_b", "l"]:
            new_p.__dict__[i] = self.__dict__[i]
        return new_p


class iFDDK(iFDD):
    """iFDD(kappa) algorithm with support for elibility traces"""

    w = 0  # log(rho) trace
    n_rho = 0  # index for rho episodes
    t = 0
    y_a = defaultdict(float)
    y_b = defaultdict(float)
    t_rho = defaultdict(int)

    def __init__(self, domain, logger, discovery_threshold, initial_representation, sparsify=True,
                 discretization=20, debug=0, useCache=0, kappa=1e-5, lambda_=0., lazy=False):
        self.lambda_ = lambda_
        self.kappa = kappa
        self.gamma = domain.gamma
        self.lazy = lazy  # lazy updating?
        super(iFDDK, self).__init__(domain, logger, discovery_threshold, initial_representation,
                                    sparsify=sparsify, discretization=discretization, debug=debug,
                                    useCache=useCache)

    def episodeTerminated(self):
        self.n_rho += 1
        self.w = 0
        self.t_rho[self.n_rho] = self.t

    def post_discover(self, s, terminal, a, td_error, phi_s, rho=1):
        """
        returns the number of added features
        """
        self.t += 1
        discovered = 0
        plus = np.random.rand() >= self.kappa

        activeFeatures = phi_s.nonzero()[0]  # Indices of non-zero elements of vector phi_s
        if not self.lazy:
            for g_index, h_index in combinations(activeFeatures, 2):
                # create potential if necessary
                self.get_potential(g_index, h_index)

            for f, potential in self.iFDD_potentials.items():
                potential.update_statistics(rho, td_error, self.lambda_, self.gamma, phi_s)
                #print plus, potential.relevance(plus=plus)
                if potential.relevance(plus=plus) >= self.discovery_threshold:
                    self.addFeature(potential)
                    del self.iFDD_potentials[f]
                    discovered += 1
            return discovered


        for g_index,h_index in combinations(activeFeatures,2):
            discovered += self.inspectPair(g_index,h_index,td_error, phi_s, rho, plus)

        if rho > 0:
            self.w += np.log(rho)
        else:
            # cut e-traces
            self.n_rho += 1
            self.w = 0
            self.t_rho[self.n_rho] = self.t
        # "rescale" to avoid numerical problems
        if self.t_rho[self.n_rho] + 300 < self.t:
            self.y_a[self.n_rho] *= (self.gamma * self.lambda_) ** (-300)
            self.y_b[self.n_rho] *= (self.gamma * self.lambda_) ** (-300)
            self.t_rho[self.n_rho] += 300

        self.y_a[self.n_rho] += np.exp(self.w) * (self.gamma * self.lambda_) ** (self.t - self.t_rho[self.n_rho]) * np.abs(td_error)
        self.y_b[self.n_rho] += np.exp(self.w) * (self.gamma * self.lambda_) ** (self.t - self.t_rho[self.n_rho]) * td_error
        return discovered

    def get_potential(self, g_index, h_index):
        g  = self.featureIndex2feature[g_index].f_set
        h  = self.featureIndex2feature[h_index].f_set
        f           = g.union(h)
        feature     = self.iFDD_features.get(f)

        if feature is not None:
            #Already exists
            return None

        #Look it up in potentials
        potential = self.iFDD_potentials.get(f)
        if potential is None:
            # Generate a new potential and put it in the dictionary
            potential = iFDDK_potential(f, g_index, h_index)
            self.iFDD_potentials[f] = potential
        return potential

    def inspectPair(self,g_index,h_index,td_error, phi_s, rho, plus):
        # Inspect feature f = g union h where g_index and h_index are the indices of features g and h
        # If the relevance is > Threshold add it to the list of features
        # Returns True if a new feature is added
        potential = self.get_potential(g_index, h_index)
        potential.update_lazy_statistics(rho, td_error, lambda_, gamma, phi_s, self.y_a, self.y_b, self.t_rho, self.w, self.t, self.n_rho)
        # Check for discovery

        relevance = potential.relevance(plus=plus)
        if relevance >= self.discovery_threshold:
            self.maxRelevance = -np.inf
            self.addFeature(potential)
            del self.iFDD_potentials[potential.f_set]
            return 1
        else:
            self.updateMaxRelevance(relevance)
            return 0

class QiFDDK(iFDDK, QFunRepresentation):

    def updateWeight(self, p1_index, p2_index):

        # Add a new weight corresponding to the new added feature for all actions.
        # The new weight is set to zero if sparsify = False, and equal to the sum of weights corresponding to the parents if sparsify = True
        a = self.domain.actions_num
        f = self.features_num-1 # Number of feature before adding the new one
        if self.sparsify:
            newElem = (self.theta[p1_index::f] + self.theta[p2_index::f]).reshape((-1,1))
        else:
            newElem =  None
        self.theta      = addNewElementForAllActions(self.theta,a,newElem)
        self.hashed_s   = None # We dont want to reuse the hased phi because phi function is changed!


