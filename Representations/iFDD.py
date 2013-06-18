######################################################
# Developed by Alborz Geramiard Nov 14th 2012 at MIT #
######################################################
# iFDD implementation based on ICML 2011 paper
from Representation import *

class iFDD_feature(object):
    index   = None        #Unique index Corresponding to its location in a vector (0 based!)
    f_set   = None      #A set of basic features that this feature corresponds to
    p1      = None         #Parent 1 feature index *Basic features have -1 for both nodes
    p2      = None         #Parent 2 feature index *Basic features have -1 for both nodes
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
            self.p1         = []
            self.p2         = []
    def show(self):
        printClass(self)
        
class iFDD_potential(object):
    relevance = None     # Sum of errors corresponding to this feature
    f_set     = None     # Set of features it corresponds to
    index     = None     # If this feature has been discovered set this to its index else 0
    p1        = None     # Parent 1 feature index
    p2        = None     # Parent 2 feature index
    count     = None     # Number of times this feature was visited
    def __init__(self,f_set,relevance,parent1,parent2):
        self.f_set      = f_set
        self.index      = -1 # -1 means it has not been discovered yet
        self.p1         = ImmutableSet([parent1])
        self.p2         = ImmutableSet([parent2])
        self.relevance  = relevance
        self.count      = 1
class iFDD(Representation):
    discovery_threshold = None # psi in the paper
    sparsify            = None # boolean specifying the use of the trick mentioned in the paper so that features are getting sparser with more feature discoveries (i.e. use greedy algorithm for feature activation)
    iFDD_features       = {}   # dictionary mapping initial feature sets to iFDD_feature 
    iFDD_potentials     = {}   # dictionary mapping initial feature sets to iFDD_potential
    def __init__(self,domain,discovery_threshold, sparsify = True, discretization = 20):
        self.discovery_threshold    = discovery_threshold
        self.sparsify               = sparsify
        self.setBinsPerDimension(domain,discretization)
        self.features_num = int(sum(self.bins_per_dim))
        self.add_initialFeatures()
        super(iFDD,self).__init__(domain,discretization)
    def phi_nonTerminal(self,s): #refer to Algorithm 2 in [Geramifard et al. 2011 ICML]
        F_s                     = zeros(sum(self.features_num),'bool')
        activeIndecies          = ImmutableSet(self.activeInitialFeatures(s))
        for candidate in powerset(activeIndecies,ascending=0):
            
            if len(activeIndecies) == 0:
                break
            
            feature = self.iFDD_features.get(ImmutableSet(candidate))
            if feature != None:
                F_s[feature.index]  = 1
                if self.sparsify:
                    activeIndecies   = activeIndecies - feature.f_set
        return F_s 
    def discover(self,s,td_error):            
        initial_features = self.activeInitialFeatures(s)
        for pair in combinations(initial_features,2):
            g,h     = pair
            f       = ImmutableSet(hstack((g,h)))
            feature  = self.iFDD_features.get(f)
            if feature is None:
                #Look it up in potentials
                potential = self.iFDD_potentials.get(f)
                if potential is None:
                    # Generate a new potential and put it in the dictionary
                    potential = iFDD_potential(f,td_error,g,h)
                    self.iFDD_potentials[f] = potential
                else:
                    potential.relevance += td_error
                    potential.count     += 1
                # Check for discovery
                if abs(potential.relevance/sqrt(potential.count)) > self.discovery_threshold:
                    
                    # Add it to the list of features
                    potential.index         = self.features_num # Features_num is always one more than the max index (0-based)
                    self.features_num       += 1
                    feature                 = iFDD_feature(potential)
                    self.iFDD_features[f]   = feature
                    
                    #Expand the size of the theta
                    p1_index = self.iFDD_features[feature.p1].index
                    p2_index = self.iFDD_features[feature.p2].index
                    self.updateWeight(p1_index,p2_index)
                     # print "New Feature:", list(f)
                     # self.show()
    def show(self):
        print "Features:"
        print join(["-"]*30)
        print " index\t| f_set\t| p1\t| p2"
        print join(["-"]*30)
        for _,feature in self.iFDD_features.iteritems():
            print " %d\t| %s\t| %s\t| %s" % (feature.index,str(list(feature.f_set)),str(list(feature.p1)),str(list(feature.p2)))
        print "Potentials:"
        print join(["-"]*30)
        print " index\t| f_set\t| relevance\t| count\t| p1\t| p2"
        print join(["-"]*30)
        for _,potential in self.iFDD_potentials.iteritems():
            print " %d\t| %s\t| %0.2f\t| %d\t| %s\t| %s" % (potential.index,str(list(potential.f_set)),potential.relevance,potential.count,str(list(potential.p1)),str(list(potential.p2)))
    def saveRepStat(self):
        pass
    def updateWeight(self,p1_index,p2_index):
        # Add a new weight corresponding to the new added feature for all actions.
        # The new weight is set to zero if sparsify = False, and equal to the sum of weights corresponding to the parents if sparsify = True
        a = self.domain.actions_num
        f = self.features_num-1 # Number of feature before adding the new one
        if self.sparsify:
            newElem = (self.theta[p1_index::f] + self.theta[p1_index::f]).reshape((-1,1)) 
        else:
            newElem =  None
        self.theta      = addNewElementForAllActions(self.theta,a,newElem)
        self.hashed_s   = None # We dont want to reuse the hased phi because phi function is changed!
    def add_initialFeatures(self):
        for i in range(self.features_num):
            feature = iFDD_feature(i)
            self.iFDD_features[ImmutableSet([i])] = feature
            #shout(self,self.iFDD_features[ImmutableSet([i])].index)
        