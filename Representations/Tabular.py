######################################################
# Developed by Alborz Geramiard Nov 14th 2012 at MIT #
######################################################
from Representation import *
class Tabular(Representation):
    def __init__(self,domain,logger,discretization = 20):
        self.setBinsPerDimension(domain,discretization) # Already performed in call to superclass
        self.features_num = int(prod(self.bins_per_dim))
        super(Tabular,self).__init__(domain,logger,discretization)
    def phi_nonTerminal(self,s):
        id      = self.hashState(s)
        F_s     = zeros(self.agg_states_num,bool)
        F_s[id] = 1
        return F_s
def featureType(self):
        return bool

