######################################################
# Developed by Alborz Geramiard Nov 9th 2012 at MIT #
######################################################
from Representation import *
class IndependentDiscretization(Representation):
    def __init__(self,domain,discretization = 20):
        super(IndependentDiscretization,self).__init__(domain,discretization)
        phi_s_size          = sum(self.buckets_per_dim)
        self.theta          = zeros(phi_s_size*self.domain.actions_num)
        self.features_num   = int(phi_s_size)
    def phi(self,s):
        ds          = self.discretized(s)
        F_s         = zeros(sum(self.buckets_per_dim),'uint8')
        shifts      = hstack((0, cumsum(self.buckets_per_dim)[:-1]))
        index       = ds-self.domain.statespace_limits[:,0]+shifts
        F_s[index.astype('uint8')] = 1
        return F_s  
        
