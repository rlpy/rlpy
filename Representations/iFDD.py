######################################################
# Developed by Alborz Geramiard Nov 14th 2012 at MIT #
######################################################
# iFDD implementation based on ICML 2011 paper
import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domains import MountainCar
from Representation import *

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
            self.f_set      = ImmutableSet([index])
            self.p1         = -1
            self.p2         = -1
    def show(self):
        printClass(self)
class iFDD_potential(object):
    relevance = None     # Sum of errors corresponding to this feature
    f_set     = None     # Set of features it corresponds to [Immutable]
    index     = None     # If this feature has been discovered set this to its index else 0
    p1        = None     # Parent 1 feature index
    p2        = None     # Parent 2 feature index
    count     = None     # Number of times this feature was visited
    def __init__(self,f_set,relevance,parent1,parent2):
        self.f_set      = deepcopy(f_set)
        self.index      = -1 # -1 means it has not been discovered yet
        self.p1         = parent1
        self.p2         = parent2
        self.relevance  = relevance
        self.count      = 1
    def show(self):
        printClass(self)
class iFDD(Representation):
    discovery_threshold     = None  # psi in the paper
    sparsify                = None  # boolean specifying the use of the trick mentioned in the paper so that features are getting sparser with more feature discoveries (i.e. use greedy algorithm for feature activation)
    iFDD_features           = {}    # dictionary mapping initial feature sets to iFDD_feature 
    iFDD_potentials         = {}    # dictionary mapping initial feature sets to iFDD_potential
    featureIndex2feature    = {}    # dictionary mapping each feature index (ID) to its feature object
    debug                   = 0     # Print more stuff
    cache                   = {}    # dictionary mapping  initial active feature set phi_0(s) to its corresponding active features at phi(s). Based on Tuna's Trick to speed up iFDD
    useCache                = 0     # this should only increase speed. If results are different something is wrong
    def __init__(self,domain,discovery_threshold, sparsify = True, discretization = 20,debug = 0,useCache = 0):
        self.discovery_threshold    = discovery_threshold
        self.sparsify               = sparsify
        self.setBinsPerDimension(domain,discretization)
        self.features_num           = int(sum(self.bins_per_dim))
        self.debug                  = debug
        self.useCache               = useCache
        self.addInitialFeatures()
        super(iFDD,self).__init__(domain,discretization)
    def phi_nonTerminal(self,s):
        # Based on Tuna's Master Thesis 2012
        F_s                     = zeros(self.features_num,'bool')
        activeIndecies          = self.activeInitialFeatures(s)
        if self.useCache:
            key                     = ImmutableSet(activeIndecies)
            finalActiveIndecies     = self.cache.get(key)
            if finalActiveIndecies is None:        
                # run regular and update the cache
                finalActiveIndecies     = self.findFinalActiveFeatures(activeIndecies)
                self.cache[key]         = finalActiveIndecies
        else:
            finalActiveIndecies         = self.findFinalActiveFeatures(activeIndecies)
        F_s[finalActiveIndecies] = 1
        return F_s
    def findFinalActiveFeatures(self,intialActiveFeatures):
        # Given the active indices of phi_0(s) find the final active indices of phi(s) based on discovered features
        finalActiveFeatures = []
        for candidate in powerset(intialActiveFeatures,ascending=0):
            if len(intialActiveFeatures) == 0:
                break
            feature = self.iFDD_features.get(ImmutableSet(candidate))
            if feature != None:
                finalActiveFeatures.append(feature.index)
                if self.sparsify:
                    intialActiveFeatures   = intialActiveFeatures - feature.f_set
        return finalActiveFeatures     
    def discover(self,phi_s,td_error):            
        activeFeatures = phi_s.nonzero()[0] # Indecies of non-zero elements of vector phi_s
        for g_index,h_index in combinations(activeFeatures,2):
            self.inspectPair(g_index,h_index,td_error)
    def inspectPair(self,g_index,h_index,td_error):
        # Inspect feature f = g union h where g_index and h_index are the indices of features g and h        
        # If the relevance is > Threshold add it to the list of features
        g  = self.featureIndex2feature[g_index].f_set 
        h  = self.featureIndex2feature[h_index].f_set
        f           = g.union(h)
        feature     = self.iFDD_features.get(f)
        if feature is None:
            #Look it up in potentials
            potential = self.iFDD_potentials.get(f)
            if potential is None:
                # Generate a new potential and put it in the dictionary
                potential = iFDD_potential(f,td_error,g_index,h_index)
                self.iFDD_potentials[f] = potential
            else:
                potential.relevance += td_error
                potential.count     += 1
            # Check for discovery
            if abs(potential.relevance/sqrt(potential.count)) > self.discovery_threshold:
                self.addFeature(potential)
    def show(self):
        print "Features:"
        print join(["-"]*30)
        print " index\t| f_set\t| p1\t| p2\t | Weights"
        print join(["-"]*30)
        for _,feature in self.iFDD_features.iteritems():
            print " %d\t| %s\t| %s\t| %s\t| %s" % (feature.index,str(list(feature.f_set)),feature.p1,feature.p2,str(self.theta[feature.index::self.features_num]))
        print "Potentials:"
        print join(["-"]*30)
        print " index\t| f_set\t| relevance\t| count\t| p1\t| p2"
        print join(["-"]*30)
        for _,potential in self.iFDD_potentials.iteritems():
            print " %d\t| %s\t| %0.2f\t| %d\t| %s\t| %s" % (potential.index,str(list(potential.f_set)),potential.relevance,potential.count,potential.p1,potential.p2)
        if self.useCache: 
            print "Cache:"
            print join(["-"]*30)
            print " initial\t| Final"
            print join(["-"]*30)
            for initial,active in self.cache.iteritems():
                print " %s\t| %s" % (str(list(initial)), active)
    def saveRepStat(self):
        pass
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
    def addInitialFeatures(self):
        for i in range(self.features_num):
            feature = iFDD_feature(i)
            #shout(self,self.iFDD_features[ImmutableSet([i])].index)
            self.iFDD_features[ImmutableSet([i])] = feature
            self.featureIndex2feature[feature.index] = feature
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
        
        #If you use cache, you should invalidate entries that their initial set contains the set corresponding to the new feature 
        if self.useCache:
            for initialActiveFeatures,_ in self.cache:
                if initialActiveFeatures.issuperset(feature.f_set):
                    self.cache.pop(initialActiveFeatures)
        
        if self.debug: self.show()
    def batchDiscover(self,td_errors, phi, maxDiscovery = 1, minRelevance  = 1e-2):
        # Discovers features using iFDD in batch setting.
        # TD_Error: p-by-1 (How much error observed for each sample)
        # phi: n-by-p features corresponding to all samples (each column corresponds to one sample)
        # minRelevance is the minimum relevance value for the feature to be expanded
        SHOW_HISTOGRAM  = 0      #Shows the histogram of relevances 
        max_excitement  = 0
        n               = self.features_num #number of features
        p               = len(td_errors)     #Number of samples
        counts          = zeros((n,n))
        relevances      = zeros((n,n))
        
        for i in range(p):
            phiphiT     = outer(phi[i,:],phi[i,:])
            relevances  += phiphiT*td_errors[i]
            counts      += phiphiT
        
        #Remove Diagonal and upper part of the relevances as they are useless
        relevances      = triu(relevances,1)
        non_zero_index  = nonzero(relevances)
        relevances[non_zero_index] = divide(abs(relevances[non_zero_index]), sqrt(counts[non_zero_index])) #Calculate relevances based on theoretical results of ICML 2013 potential submission 

        if SHOW_HISTOGRAM:
            e_vec  = relevances.flatten()
            e_vec  = e_vec[e_vec != 0]
            e_vec  = sort(e_vec)
            drawHist(e_vec)
            pl.show(block=False)

        #Find indexes to non-zero excited pairs
        (F1, F2)            = relevances.nonzero() # F1 and F2 are the parents of the potentials
        relevances          = relevances[F1,F2]
        #Sort based on relevances
        sortedIndecies  = argsort(relevances)[::-1] # We want high to low hence the reverse: [::-1]
        max_relevance   = relevances[sortedIndecies[0]];
        #Add top <maxDiscovery> features
        for j in range(min(maxDiscovery,len(relevances))):
            max_index   = sortedIndecies[j]
            f1          = F1[max_index]
            f2          = F2[max_index]
            relevance   = relevances[max_index]
            if relevance > minRelevance:
                print 'Added Feature [%d,%d]' % (f1,f2)
                self.inspectPair(f1, f2, inf)

if __name__ == '__main__':
    discovery_threshold = 1
    domain      = MountainCar()
    rep         = iFDD(domain,discovery_threshold,debug=1,useCache=1)
    rep.theta   = arange(rep.features_num*domain.actions_num)*10
    print rep.findFinalActiveFeatures([0,1,20])
    rep.show()
    print 'discover 0,20'
    phi_s = zeros(rep.features_num)
    phi_s[0] = 1
    phi_s[20] = 1
    rep.discover(phi_s, discovery_threshold+1)
    print rep.findFinalActiveFeatures([0,1,20])
    # Change the weight for new feature 40
    rep.theta[40] = -100
    print 'discover 0,1,20'
    rep.inspectPair(1,40, discovery_threshold+1)
    rep.show()
    print rep.theta
    s = domain.statespace_limits[:,0]
    print rep.findFinalActiveFeatures([0,1,20])
    rep.show()
    
    
    
    