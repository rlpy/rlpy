######################################################
# Developed by Alborz Geramiard Nov 9th 2012 at MIT #
######################################################
from Representation import *
class IndependentDiscretization(Representation):
    def __init__(self,domain,logger,discretization = 20):
        self.setBinsPerDimension(domain,discretization)
        self.features_num = int(sum(self.bins_per_dim))
        self.maxFeatureIDperDimension = cumsum(self.bins_per_dim)-1
        super(IndependentDiscretization,self).__init__(domain,logger,discretization)
    def phi_nonTerminal(self,s):
        #print "IN PHI:", s,bs, 'shift',shifts,'index:', index, 'Bins:', self.bins_per_dim
        F_s                                 = zeros(self.features_num,'bool')
        F_s[self.activeInitialFeatures(s)]  = 1
        return F_s
    def getDimNumber(self,f):
        # Returns the dimension number corresponding to this feature
        dim     = searchsorted(self.maxFeatureIDperDimension,f) 
        return dim