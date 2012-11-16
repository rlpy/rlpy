######################################################
# Developed by Alborz Geramiard Nov 14th 2012 at MIT #
######################################################
# iFDD implementation based on ICML 2011 paper
from Representation import *

class iFDD_feature(object):
    id = None         #Unique id Corresponding to its location in a vector (0 based!)
    f_set = None      #A set of basic features that this feature corresponds to
    p1 = None         #Parent 1 feature id *Basic features have -1 for both nodes
    p2 = None         #Parent 2 feature id *Basic features have -1 for both nodes
    def __init__(self,potential):
        for prop in ['id','f_set','p1','p2']:
            vars(self)[prop] = vars(potential)[prop]
    def __init__(self,id):
        self.id     = id
        self.f_set  = ImmutableSet([id])
        self.p1     = []
        self.p2     = []
class iFDD_potential(object):
    relevance = None     # Sum of errors corresponding to this feature
    f_set     = None     # Set of features it corresponds to
    id        = None     # If this feature has been discovered set this to its id else 0
    p1        = None     # Parent 1 feature id
    p2        = None     # Parent 2 feature id
    count     = None     # Number of times this feature was visited
    def __init__(self,f_set,relevance,parent1,parent2):
        self.fset       = fset
        self.id         = -1
        self.p1         = parent1
        self.p2         = parent2
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
    def phi(self,s): #refer to Algorithm 2 in [Geramifard et al. 2011 ICML]
        F_s                     = zeros(sum(self.bins_per_dim),'bool')
        activeIndecies          = ImmutableSet(self.activeInitialFeatures(s))
        for candidate in powerset(activeIndecies,ascending=0):
            
            if len(activeIndecies) == 0:
                break
            
            feature = self.iFDD_features.get(ImmutableSet(candidate))
            if feature != None:
                F_s[feature.id]  = 1
                activeIndecies   = activeIndecies - feature.f_set
        return F_s 
    def discover(self,initial_features,td_error):            
        for pair in combinations(initial_features,2):
            g,h     = pair
            f       = ImmutableSet(hstack((g,h)))
            fature  = self.iFDD_features.get(f)
            if fature is None:
                #Look it up in potentials
                potential = self.iFDD_potentials.get(f)
                if potential is None:
                    # Generate a new potential and put it in the dictionary
                    potential = iFDD_potential(f,td_error,g,h)
                    iFDD_potentials[f] = potential
                else:
                    potential.relevance += td_error
                    potential.count     += 1
                # Check for discovery
                if abs(potential.relevance/sqrt(potential.count)) > self.discovery_threshold:
                    # Add it to the list of features
                    self.features_num += 1
                    potential.id = self.features_num
                    feature = iFDD_feature(potential)
                    iFDD_feature[f] = feature
                    #Expand the size of the theta
                    self.addNewWeight(iFDD_feature[feature.p1].id,iFDD_feature[feature.p2].id)
        return F_s
    def show(self):
        print "Features:"
        print join(["-"]*30)
        print " ID\t| f_set\t| p1\t| p2"
        print join(["-"]*30)
        for _,feature in self.iFDD_features.iteritems():
            print " %d\t| %s\t| %s\t| %s" % (feature.id,str(list(feature.f_set)),str(list(feature.p1)),str(list(feature.p2)))
        print "Potentials:"
        print join(["-"]*30)
        print " ID\t| f_set\t| relevance\t| count\t| p1\t| p2"
        print join(["-"]*30)
        for _,potential in self.iFDD_potentials.iteritems():
            print " %d\t| %s\t| %0.2f\t| %d\t| %s\t| %s" % (potential.id,str(list(potential.f_set)),potential.relevance,potential.count,str(list(potential.p1)),str(list(potential.p2)))
    def saveRepStat(self):
        pass
    def addNewWeight(self,p1,p2):
        # Add a new weight corresponding to the new added feature for all actions.
        # The new weight is set to zero if sparsify = False, and equal to the sum of weights corresponding to the parents if sparsify = True
        x               = self.theta.reshape(self.domain.actions_num,-1) # -1 means figure the other dimension yourself
        if self.sparsify:
            parents_sum     = x[:,p1] + x[:,p2]
            x               = hstack((x,parents_sum))
        else:
            x               = hstack((x,zeros((self.domain.actions_num,1))))
        self.theta      = x.reshape(1,-1).flatten()
        self.hashed_s   = None # We dont want to reuse the hased phi because phi function is changed!
    def add_initialFeatures(self):
        for i in range(self.features_num):
            feature = iFDD_feature(i)
            self.iFDD_features[ImmutableSet([i])] = feature
        self.show()