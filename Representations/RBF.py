######################################################
# Developed by Alborz Geramiard Nov 9th 2012 at MIT #
######################################################
from Representation import *
class RBF(Representation):
    rbfs_mu     = None          # The mean of RBFs
    rbfs_sigma  = None          # The variance of the RBFs (uniformly selected between [0, dimension width]
    def __init__(self,domain,logger,rbfs = 20, id = 1):
        self.features_num   = rbfs+1 # adds a constant 1 to each feature vector
        self.rbfFile        = '%d-rbfs.npy' % id # used to save or load RBFs
        dims                = domain.state_space_dims
        self.rbfs_mu        = zeros((rbfs,dims))
        self.rbfs_sigma     = zeros((rbfs,dims))
        dim_widths          = (domain.statespace_limits[:,1]-domain.statespace_limits[:,0])
        super(RBF,self).__init__(domain,logger)
        if os.path.exists(self.rbfFile):
            data = load(self.rbfFile)
            #First row corresponds to the mean and second row corresponds to variance.
            self.logger.log('Loaded the RBFs from: %s' % self.rbfFile)
            self.rbfs_mu    = data[0]
            self.rbfs_sigma = data[1]
        else:
            for i in arange(rbfs):
                for d in arange(dims):
                    self.rbfs_mu[i,d]        = random.uniform(domain.statespace_limits[d,0],
                                                            domain.statespace_limits[d,1])
                    self.rbfs_sigma[i,d]     = random.uniform(dim_widths[d]/2.0,dim_widths[d])
            data    = zeros((2,rbfs,dims))
            data[0] = self.rbfs_mu
            data[1] = self.rbfs_sigma
            save(self.rbfFile,data)
            self.logger.log('Saved the RBFs to: %s' % self.rbfFile)
                #random.uniform(dim_widths[d]/2.0,dim_widths[d]) 
#        pl.figure()
#        pl.plot(self.rbfs_mu[:,1],self.rbfs_mu[:,0],'.k')
#        pl.ioff
#        pl.show()
#        raw_input()
    def phi_nonTerminal(self,s):
        F_s         = ones(self.features_num)
        for i in arange(0,self.features_num-1):
            exponent = sum((s-self.rbfs_mu[i,:])**2/(2.0*self.rbfs_sigma[i,:]))
            F_s[i] = exp(-exponent)
        return F_s 
        #return normalize(F_s) DO NOT normalize the rbfs as it can make the learning much slower if you dont increase alpha proportionally. 
    def featureType(self):
        return float
        
