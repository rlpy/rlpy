# INPROGRESS
######################################################
# Developed by Alborz Geramiard Dec 12th 2012 at MIT #
######################################################
# Kernelized version of the iFDD
# - Instead of piece-wise constant any function can be used as kernels
# - Samples are used as the basis of the kernels with some sparsification mechanism 

import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domains import MountainCar
from iFDD import iFDD
class KernelizediFDD(iFDD):
    kernel                  = None  # The function that returns a real value corresponding to distance for every pair in the state space
    centers                 = {}    # The centers of the kernels
    def __init__(self,domain,logger,discovery_threshold, sparsify = True, discretization = 20,debug = 0,useCache = 0,maxBatchDicovery = 1, batchThreshold = 0, kernel = lambda x,y: norm(x-y), maxMemory = 1000):
        self.kernel                 = kernel
        self.discovery_threshold    = discovery_threshold
        self.maxMemory              = maxMemory
        self.sparsify               = sparsify
        self.debug                  = debug
        self.useCache               = useCache
        self.maxBatchDicovery       = maxBatchDicovery
        self.batchThreshold         = batchThreshold
        self.features_num           = 0 
        super(iFDD,self).__init__(domain,logger,discretization)
        self.logger.log("Kernel:\t\t\t%s" % self.kernel.__name__)
        self.logger.log("Maximum Features:\t%d" % self.maxMemory)
        self.logger.log("Threshold:\t\t%0.3f" % self.discovery_threshold)
        self.logger.log("Sparsify:\t\t%d"% self.sparsify)
        self.logger.log("Cached:\t\t\t%d"% self.useCache)
        self.logger.log("Max Batch Discovery:\t%d"% self.maxBatchDicovery)
        self.logger.log("Batch Threshold:\t\t%0.3f"% self.batchThreshold)
    def phi_nonTerminal(self,s):
        # Find the n closest points
if __name__ == '__main__':
    SIGMA               = 1
    kernel              = lambda x,y: prod(normpdf(x,y,SIGMA)) 
    STDOUT_FILE         = 'out.txt'
    JOB_ID              = 1
    OUT_PATH            = 'Results/Temp'
    logger              = Logger('%s/%d-%s'%(OUT_PATH,JOB_ID,STDOUT_FILE))
    discovery_threshold = 1
    domain      = MountainCar()
    rep         = KernelizediFDD(domain,logger,discovery_threshold,debug=0,useCache=1,kernel=kernel)
    rep.theta   = arange(rep.features_num*domain.actions_num)*10
    print 'Initial [0,1,20] => ',
    print rep.findFinalActiveFeatures([0,1,20])
    print 'Initial [0,20] => ',
    print rep.findFinalActiveFeatures([0,20])
    rep.showCache()
    print 'discover 0,20'
    phi_s = zeros(rep.features_num)
    phi_s[0] = 1
    phi_s[20] = 1
    rep.discover(phi_s, discovery_threshold+1)
    rep.showFeatures()
    rep.showCache()
    print 'Initial [0,20] => ',
    print rep.findFinalActiveFeatures([0,20])
    print 'Initial [0,1,20] => ',
    print rep.findFinalActiveFeatures([0,1,20])
    rep.showCache()
    # Change the weight for new feature 40
    rep.theta[40] = -100
    print 'Initial [0,20] => ',
    print rep.findFinalActiveFeatures([0,20])
    print 'discover 0,1,20'
    rep.inspectPair(1,40, discovery_threshold+1)
    rep.showFeatures()
    rep.showCache()
    print 'Initial [0,1,20] => ',
    print rep.findFinalActiveFeatures([0,1,20])
    rep.showCache()
    
    
    
    