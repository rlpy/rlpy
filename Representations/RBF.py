######################################################
# Developed by Alborz Geramiard Nov 9th 2012 at MIT #
######################################################
from Representation import *
class RBF(Representation):
    rbfs_mu     = None          # The mean of RBFs
    rbfs_sigma  = None          # The variance of the RBFs (uniformly selected between [0, dimension width]
    def __init__(self,domain,logger,rbfs = 20):
        self.features_num   = rbfs+1 # adds a constant 1 to each feature vector
        dims                = domain.state_space_dims
        self.rbfs_mu        = zeros((rbfs,dims))
        self.rbfs_sigma     = zeros((rbfs,dims))
        dim_widths          = (domain.statespace_limits[:,1]-domain.statespace_limits[:,0])
        for i in arange(rbfs):
            for d in arange(dims):
                self.rbfs_mu[i,d]        = random.uniform(domain.statespace_limits[d,0],
                                                        domain.statespace_limits[d,1])
                self.rbfs_sigma[i,d]     = random.uniform(dim_widths[d]/2.0,dim_widths[d])
#        pl.plot(self.rbfs_mu[:,1],self.rbfs_mu[:,0],'.k')
#        pl.show()
#        raw_input()
        super(RBF,self).__init__(domain,logger)
    def phi_nonTerminal(self,s):
        F_s         = ones(self.features_num)
        for i in arange(0,self.features_num-1):
            F_s[i] = prod(normpdf(s,self.rbfs_mu[i,:], self.rbfs_sigma[i,:]))
        return normalize(F_s)
    def featureType(self):
        return float
        
