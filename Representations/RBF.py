######################################################
# Developed by Alborz Geramiard Nov 9th 2012 at MIT #
######################################################
from Representation import *
class RBF(Representation):
    rbfs_mu     = None          # The mean of RBFs
    rbfs_sigma  = None          # The variance of the RBFs (uniformly selected between [0, dimension width]
    def __init__(self,domain,rbfs = 20):
        super(RBF,self).__init__(domain)
        self.features_num   = rbfs+1 # adds a constant 1 to each feature vector
        self.theta          = zeros(self.features_num*self.domain.actions_num)
        dims                = self.domain.state_space_dims
        self.rbfs_mu        = zeros((rbfs,dims))
        self.rbfs_sigma     = zeros((rbfs,dims))
        for i in arange(rbfs):
            for d in arange(dims):
                self.rbfs_mu[i,d]        = random.uniform(self.domain.statespace_limits[d,0],
                                                        self.domain.statespace_limits[d,1])
                self.rbfs_sigma[i,d]     = random.uniform(0,(self.domain.statespace_limits[d,1]-self.domain.statespace_limits[d,0])/2)
#        pl.plot(self.rbfs_mu[:,1],self.rbfs_mu[:,0],'.k')
#        pl.show()
#        raw_input()
    def phi(self,s):
        F_s         = ones(self.features_num)
        for i in range(0,self.features_num-1):
            F_s[i] = prod(normpdf(s,self.rbfs_mu[i,:], self.rbfs_sigma[i,:]))
        return F_s
        
