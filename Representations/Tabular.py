######################################################
# Developed by Alborz Geramiard Nov 14th 2012 at MIT #
######################################################
from Representation import *
class Tabular(Representation):
    def __init__(self,domain,discretization = 20):
        self.buckets_per_dim = empty(domain.state_space_dims)
        for d in arange(domain.state_space_dims):
             if d in domain.continous_dims:
                 self.buckets_per_dim[d] = discretization
             else:
                 self.buckets_per_dim[d] = domain.statespace_limits[d,1] - domain.statespace_limits[d,0]+1
        self.features_num = int(prod(self.buckets_per_dim))
        super(Tabular,self).__init__(domain,discretization)
    def phi(self,s):
        ds      = self.discretized(s)
        id      = self.hashState(s)
        F_s     = zeros(self.domain.states_num)
        F_s[id] = 1
        return F_s

