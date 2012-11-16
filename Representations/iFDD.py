######################################################
# Developed by Alborz Geramiard Nov 14th 2012 at MIT #
######################################################
# iFDD implementation based on ICML 2011 paper
 

from Representation import *
class iFDD_feature(object):
    id          #Unique id Corresponding to its location in a vector (0 based!)
    f_set       #A set of basic features that this feature corresponds to
    name        #Name of the feature. If it is not basic the name = ""
    p1 = -1;    #Parent 1 feature id *Basic features have -1 for both nodes
    p2 = -1;    #Parent 2 feature id *Basic features have -1 for both nodes

class iFDD_potential(object):
    relevance      # Sum of errors corresponding to this feature
    f_set          # Set of features it corresponds to
    id             # If this feature has been discovered set this to its id else 0
    queued         # True if a reference to this potential exists in the new feature priority queue
    p1 = -1        # Parent 1 feature id
    p2 = -1        # Parent 2 feature id
    count          # Number of times this feature was visited
    
    
class iFDD(Representation):
    discovery_threshold = None # psi in the paper
    sparsify = None # boolean specifying the use of the trick mentioned in the paper so that features are getting sparser with more feature discoveries (i.e. use greedy algorithm for feature activation)
    def __init__(self,domain,discovery_threshold, sparsify = True, discretization = 20):
        self.setBinsPerDimension(domain,discretization)
        self.features_num           = int(sum(self.bins_per_dim))
        self.discovery_threshold    = discovery_threshold
        self.sparsify               = sparsify
        super(iFDD,self).__init__(domain,discretization)
    def phi(self,s):
        F_s                         = zeros(sum(self.bins_per_dim),'uint8')
    def discover(self,initial_features):
        pass
    def show(self):
        pass
    def saveRepStat(self):
        pass
    def initialFeatures(self,s):
        #Return the index of active initial features for state s 
        ds          = self.discretized(s)
        shifts      = hstack((0, cumsum(self.bins_per_dim)[:-1]))
        index       = ds-self.domain.statespace_limits[:,0]+shifts
        return index.astype('uint8')