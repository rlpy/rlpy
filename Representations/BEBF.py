######################################################
#       Robert H. Klein, Dec. 19, 2012 at MIT        #
######################################################
#
# REQUIRES the implementation of locally-weighted
# projection regression (LWPR), available at:
# http://wcms.inf.ed.ac.uk/ipab/slmc/research/software-lwpr
#
# Parameters set according to:
# Parr et al. 'Analyzing Feature Generation for Function Approximation'
#
# Bellman-Error Basis Function Representation.
# 1. Initial basis function based on immediate reward.
# 2. Evaluate r + Q(s', \pi{s'}) - Q(s,a) for all samples.
# 3. Train function approximator on bellman error of present solution above
# 4. Add the above as a new basis function.
# 5. Repeat the process using the new basis until the most
# recently added basis function has norm <= normThreshold, which
# Parr et al. used as 10^-5.

import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))

#from scipy.interpolate import Rbf
from sklearn import svm
from Tools import *
from Domains import PitMaze
from Representation import *


class BEBF(Representation):
    debug                   = 0
#    memoized_phi            = {}    # dictionary mapping each feature index 
    featureIndex2feature    = {}    # dictionary mapping each feature index (ID) to its feature (function) object
#    useCache                = 0     # this should only increase speed. If results are different something is wrong
    maxBatchDicovery        = 0     # Number of features to be expanded in the batch setting
    normThreshold           = 0     # Minimum value of feature relevance for the batch setting (10^-5 per Parr et al.)
    features                = None # Array of pointers to feature functions, indexed by order created
    def __init__(self,domain,logger, discretization = 20, debug = 0,maxBatchDicovery = 1, normThreshold = 10 ** -3):
        self.setBinsPerDimension(domain,discretization)
        # self.features_num           = int(sum(self.bins_per_dim))
        self.features_num           = 1
        self.debug                  = debug
        self.maxBatchDicovery       = maxBatchDicovery
        self.normThreshold          = normThreshold
        self.addInitialFeatures()
        super(BEBF,self).__init__(domain,logger,discretization)
        self.logger.log("Max Batch Discovery:\t%d"% self.maxBatchDicovery)
        self.logger.log("Norm Threshold:\t\t%0.3f"% self.normThreshold)
    ## Adds an initial basis function estimated using only immediate reward.
    
        ## @return: a function object corresponding to the 
    def getFunctionApproximation(self,X,y):
        #return Rbf(X,y,function='multiquadric') # function = gaussian
        bebfApprox = svm.SVR(kernel='rbf', degree=3, C=1.0) # support vector regression
                                                 # C = penalty parameter of error term, default 1
        bebfApprox.fit(X,y)
        return bebfApprox
    
    def addInitialFeatures(self):
        numDims = 2
        numSamples = 100
        y = random.randn(numSamples)
        X = random.randn(numSamples,numDims)
        self.features = array([self.getFunctionApproximation(X, y)])
#        for i in range(self.features_num):
#            feature = iFDD_feature(i)
#            #shout(self,self.iFDD_features[ImmutableSet([i])].index)
#            self.iFDD_features[ImmutableSet([i])] = feature
#            self.featureIndex2feature[feature.index] = feature
    def phi_nonTerminal(self,s):
        # Based on Tuna's Master Thesis 2012
        F_s = zeros(self.features_num)
        for i in range(self.features_num):
            F_s[i] = self.features[i].predict(s)
#        print 's,F_s',s,F_s
#        shout('F_s:',F_s)
        return F_s
    
    ## Adds new features based on the Bellman Error in batch setting.
    # @param td_errors: p-by-1 (How much error observed for each sample)
    # @param phi: n-by-p features corresponding to all samples (each column corresponds to one sample)
    # self.normThreshold is threshold below which no more BEBFs are added.
    def batchDiscover(self,td_errors, all_phi_s, s):
        # need states here instead?
        addedFeature = False
        for j in range(self.maxBatchDicovery):
#            print 's'
#            print 'tderr',td_errors[0:20]
            newFeature = self.getFunctionApproximation(s, td_errors)
            self.features = append(self.features,newFeature)
            if self.debug:
                shout('s',s)
                shout('td_errors',td_errors)
            # PLACEHOLDER for norm of function
            norm = max(abs(td_errors))# Norm of function
            print 'norm',norm
            if norm > self.normThreshold:
#                print 'added feature'
                self.addNewWeight()
                addedFeature = True
                self.features_num+= 1
                
            else: break
        return addedFeature
    
    
if __name__ == '__main__':
    STDOUT_FILE         = 'out.txt'
    JOB_ID              = 1
    OUT_PATH            = 'Results/Temp'
#    logger              = Logger('%s/%d-%s'%(OUT_PATH,JOB_ID,STDOUT_FILE))
    logger              = Logger()
    discovery_threshold = 1
    domain      = PitMaze()
    rep         = BEBF(domain,logger,debug=1,normThreshold = 10 ** -5)
    rep.theta   = arange(rep.features_num*domain.actions_num)*10
    print 'initial features'
    print rep.features_num,'---',rep.features
    s           = domain.s0() 
    a           = domain.possibleActions(s)
    a = a[0]
    r,ns,terminal   = domain.step(s, a)
    print 'step 2 r,ns',r,ns
    
    
    