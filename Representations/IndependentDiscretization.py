######################################################
# Developed by Alborz Geramiard Nov 9th 2012 at MIT #
######################################################
from Representation import *
class IndependentDiscretization(Representation):
    def __init__(self,domain,discretization = 20):

        # Find buckets per dimension
        self.buckets_per_dim = empty(domain.state_space_dims)
        for d in arange(domain.state_space_dims):
             if d in domain.continous_dims:
                 self.buckets_per_dim[d] = discretization
             else:
                 self.buckets_per_dim[d] = domain.statespace_limits[d,1] - domain.statespace_limits[d,0]+1
        self.features_num   = int(sum(self.buckets_per_dim))
        super(IndependentDiscretization,self).__init__(domain,discretization)
    def phi(self,s):
        ds          = self.discretized(s)
        F_s         = zeros(sum(self.buckets_per_dim),'uint8')
        shifts      = hstack((0, cumsum(self.buckets_per_dim)[:-1]))
        index       = ds-self.domain.statespace_limits[:,0]+shifts
        F_s[index.astype('uint8')] = 1
        return F_s  
        
