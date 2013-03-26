######################################################
# Developed by Alborz Geramiard Nov 9th 2012 at MIT #
######################################################
import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domains import *
from Representation import *

class RBF(Representation):
    SHOW_CENTERS    = False
    rbfs_mu         = None          # The mean of RBFs
    rbfs_sigma      = None          # The variance of the RBFs (uniformly selected between [0, dimension width]
    def __init__(self,domain,logger,rbfs = 20, id = 1):
        if isinstance(domain,Pendulum):
            # Put 9 equal in the intersections + 1 extra
            self.domain             = domain
            dims                    = domain.state_space_dims
            bins                    = 4
            self.rbfs_mu, rbfs      = self.uniformRBFs([bins,bins])
            logger.log('Using 3x3 uniform RBFs => 9 RBFs + 1 constant.') 
            self.rbfs_sigma         = ones((rbfs,dims))
        elif isinstance(domain,PitMaze):
            # Put 20 equal in the intersections + 1 extra
            self.domain             = domain
            dims                    = domain.state_space_dims
            bins                    = 5
            self.rbfs_mu, rbfs      = self.uniformRBFs([bins,bins],True)
            logger.log('Using %dx%d uniform RBFs => %d RBFs + 1 constant.' % (bins+1,bins+1,rbfs)) 
            dim_widths              = (domain.statespace_limits[:,1]-domain.statespace_limits[:,0])
            self.rbfs_sigma         = empty((rbfs,dims))
            for d in arange(dims):
                self.rbfs_sigma[:,d] = dim_widths[d]/3.0
        else:
            #id = 2 # Best Performing.
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
                        self.rbfs_sigma[i,d]     = random.uniform(dim_widths[d]/4.0,dim_widths[d]/3.0)
                data    = zeros((2,rbfs,dims))
                data[0] = self.rbfs_mu
                data[1] = self.rbfs_sigma
                save(self.rbfFile,data)
                self.logger.log('Saved the RBFs to: %s' % self.rbfFile)
                #random.uniform(dim_widths[d]/2.0,dim_widths[d]) 
        if self.SHOW_CENTERS:
            pl.plot(self.rbfs_mu[:,1],self.rbfs_mu[:,0],'.k')
            pl.xlim(self.domain.statespace_limits[0])
            pl.ylim(self.domain.statespace_limits[1])
            pl.draw()
            raw_input()
        self.features_num   = rbfs+1 # adds a constant 1 to each feature vector
        super(RBF,self).__init__(domain,logger)
    def phi_nonTerminal(self,s):
        F_s         = ones(self.features_num)
        for i in arange(0,self.features_num-1):
            exponent = sum((s-self.rbfs_mu[i,:])**2/(2.0*self.rbfs_sigma[i,:]))
            F_s[i] = exp(-exponent)
        return F_s 
        #return normalize(F_s) DO NOT normalize the rbfs as it can make the learning much slower if you dont increase alpha proportionally. 
    def uniformRBFs(self,bins_per_dimension, IncludeBorders = False):
        # Positions RBF Centers uniformly across the state space and returns the centers as RBFs-by-dims matrix
        # Each row is a center of an RBF
        # Example: 2D domain where each dimension is in [0,3]
        # with parameter [1,2], False => we get 2 centers hence the output:
        # 1.5    1
        # 1.5    2
        # with parameter [1,2], True => we get 2 centers hence the output:
        # 1.5    1
        # 1.5    2
        # If IncludeBorders is True => Centers can also be put on the borders of dimensions
        # The second output is the total number of rbfs which is prod(RBF_per_dimension)
        #Find centers in each dimension
        domain      = self.domain
        dims        = domain.state_space_dims
        rbfs_num    = prod(RBF_per_dimension)
        bins        = RBF_per_dimension + 2
        if IncludeBorders: bins 
        all_centers = []
        for d in arange(dims):
            centers = linspace(domain.statespace_limits[d,0],domain.statespace_limits[d,1],RBF_per_dimension[d]+2)
            if not IncludeBorders:
                centers = centers[1:-1] #Exclude the beginning and ending
            all_centers.append(centers.tolist())
        #print all_centers
        # Find all pair combinations of them:
        result = perms(all_centers)
        #print result.shape
        return result, rbfs_num
    def featureType(self):
        return float
        
