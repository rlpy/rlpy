######################################################
# Developed by Alborz Geramiard Nov 9th 2012 at MIT #
######################################################
from Representation import *
class IndependentDiscretization(Representation):
    def __init__(self,domain,discretization = 20):

        # Find bins per dimension
        self.bins_per_dim = empty(domain.state_space_dims)
        for d in arange(domain.state_space_dims):
             if d in domain.continous_dims:
                 self.bins_per_dim[d] = discretization
             else:
                 self.bins_per_dim[d] = domain.statespace_limits[d,1] - domain.statespace_limits[d,0]
        self.features_num   = int(sum(self.bins_per_dim))
        super(IndependentDiscretization,self).__init__(domain,discretization)
    def phi(self,s):
        bs          = self.binState(s)
        F_s         = zeros(self.features_num,'bool')
        shifts      = hstack((0, cumsum(self.bins_per_dim)[:-1]))
        index       = bs+shifts
        #print "IN PHI:", s,bs, 'shift',shifts,'index:', index, 'Bins:', self.bins_per_dim
        F_s[index.astype('uint8')] = 1
        return F_s  