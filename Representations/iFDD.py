######################################################
# Developed by Alborz Geramiard Nov 14th 2012 at MIT #
######################################################
# iFDD implementation based on ICML 2011 paper
 

from Representation import *
class iFDD(Representation):
    discovery_threshold # psi in the paper
    sparsify # boolean specifying the use of the trick mentioned in the paper so that features are getting sparser with more feature discoveries (i.e. use greedy algorithm for feature activation)
    def __init__(self,domain,discovery_threshold, sparsify = True, discretization = 20, ):

        # Find bins per dimension
        self.bins_per_dim = empty(domain.state_space_dims)
        for d in arange(domain.state_space_dims):
             if d in domain.continous_dims:
                 self.bins_per_dim[d] = discretization
             else:
                 self.bins_per_dim[d] = domain.statespace_limits[d,1] - domain.statespace_limits[d,0]+1
        self.discovery_threshold    = discovery_threshold
        self.sparsify               = sparsify
        self.features_num           = int(sum(self.bins_per_dim))
        
        super(iFDD,self).__init__(domain,discretization)
    def phi(self,s):
        F_s                         = zeros(sum(self.bins_per_dim),'uint8')

    def initialFeatures(self,s):
        #Return the index of active initial features for state s 
        ds          = self.discretized(s)
        shifts      = hstack((0, cumsum(self.bins_per_dim)[:-1]))
        index       = ds-self.domain.statespace_limits[:,0]+shifts
        return index.astype('uint8')